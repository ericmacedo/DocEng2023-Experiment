from models.clustering import Report, Clustering, clusterer
from utils import load_model, print_header, Logger
from models.document import DocumentModels
from models import ModelType, models
from functools import partialmethod
from models.metrics import Metrics
from models.word import WordModels
from tabulate import tabulate
from yaspin import yaspin
from pathlib import Path
from models import Data
from tqdm import tqdm
import pandas as pd
import warnings
import argparse
import sys

# ============================================================================
#   Arguments
# ============================================================================
parser = argparse.ArgumentParser()

parser.add_argument("--name", "-n", type=str, required=True, dest="name",
                    help="Name of the dataset")
parser.add_argument("--path", "-p", type=str, required=True, dest="path",
                    help="Path to the CSV file")
parser.add_argument("--sample-size", "-s", type=int, default=100,
                    dest="sample_size", required=True,
                    help="Number of observations to generate")
parser.add_argument("--id", "-id", type=str, dest="id",
                    help="The column name that contain the document identifier")
parser.add_argument("--label", "-l", type=str, required=True, dest="label",
                    help="The column name that contain the document label")
parser.add_argument("--content", "-c", type=str, nargs="+",
                    required=True, dest="content",
                    help="The column names that contain the document content")

parser.add_argument("--train-models", "-t", type=str, nargs="+",
                    dest="train_models", default="all",
                    choices=[m['name'] for m in models] + ["all", "none"],
                    help="Name of the models to train (default 'all'))")

args = parser.parse_args()

dataset = args.name
path = args.path

id_field = args.id
label_field = args.label
content_fields = args.content

train_models = args.train_models
if train_models == "all":
    train_models = [m['name'] for m in models]
elif train_models == "none":
    train_models = []

# ============================================================================
#   Setup
# ============================================================================
data_path = Path(f"./data/{dataset}").resolve()
data_path.mkdir(parents=True, exist_ok=True)

log_path = Path(f"./data/{dataset}/report.log").resolve()

if log_path.exists():
    log_path.unlink()

log_path.touch()

sys.stdout = Logger(str(log_path))
sys.stdout.isatty = lambda : False
sys.stdout.encoding = "utf-8"

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

warnings.filterwarnings("ignore")

# ============================================================================
#   Collect and process data
# ============================================================================
df = pd.read_csv(filepath_or_buffer=f"{Path(path).resolve()}",
                 engine="python", encoding="utf-8",
                 usecols=[] + [id_field] + [label_field] + content_fields)

data = Data(data=df,
            name=dataset,
            id_field=id_field,
            content_fields=content_fields,
            label_field=label_field)

# ============================================================================
#   Execute experiment
# ============================================================================
repeat = args.sample_size

print_header(f"Running experiment on dataset {data.name} with {repeat} observations")

with yaspin(color="cyan") as sp:
    print("Training models".upper())
    for model_cfg in models:
        if not model_cfg['name'] in train_models:
            continue
        Path(
            f"{data_path}/{model_cfg['model_type'].value}/{model_cfg['name']}"
        ).mkdir(parents=True, exist_ok=True)
        model_cfg["dataset"] = data.name
        
        sp.write(f"> Traning {model_cfg['model_type'].value} model {model_cfg['name']}")
        
        model = load_model(**model_cfg)
        for id in range(1, repeat):
            sp.text = f"Training model {id}"
            if id > 1 and model_cfg["name"] == "BagOfWords":
                model.save_model(id=id,
                                 dataset=data.name,
                                 model=model.load_model(dataset=dataset, id=1))
            else:
                model.train_model(id=id, dataset=data.name, corpus=data.processed)

        del model

    sp.ok("✔ ")


with yaspin(text="Clustering", color="cyan") as sp:
    print("Clustering models".upper())
    word_models = [*filter(
        lambda m: m['model_type'] == ModelType.WORD, models)]
    doc_models = [*filter(
        lambda m: m['model_type'] == ModelType.DOCUMENT, models)]

    metrics_list = [m.value for m in Metrics]

    pd_index = list(zip(*[
        [ i for m in WordModels for i in [m.value] * len(metrics_list) ],
        metrics_list * 3
    ]))

    pd_index = pd.MultiIndex.from_tuples(pd_index)

    pd_cols = [m.value for m in DocumentModels]

    report_df = pd.DataFrame(index=pd_index, columns=pd_cols, dtype=str)
    
    for word_model_cfg in word_models:
        word_model_cfg["dataset"] = data.name
        word_model = load_model(**word_model_cfg)
        
        for doc_model_cfg in doc_models:
            doc_model_cfg["dataset"] = data.name
            doc_model = load_model(**doc_model_cfg)

            sp.write("> Running clustering algorithm {0} x {1}".format(
                word_model_cfg['name'], doc_model_cfg['name']))
            report = Report(name="_".join(data.name,
                                          word_model_cfg['name'],
                                          doc_model_cfg['name']))
            for id in range(1, repeat):
                sp.text = f"Clustering iteration {id}"
                word_vectors = word_model.load_model(id=id, dataset=data.name)
                doc_vectors = doc_model.load_model(id=id, dataset=data.name)
                embeddings = doc_vectors.get_vectors(id=id,
                                                     dataset=data.name,
                                                     corpus=data.processed)
                labels_pred, time = clusterer(
                    dataset=data.name, 
                    id=id,
                    word_vectors=word_vectors,
                    doc_vectors=doc_vectors,
                    embeddings=embeddings,
                    k=data.k)

                report.append(Clustering(
                    name=f"{report.name}_{id}",
                    X=embeddings,
                    labels_pred=labels_pred,
                    labels_true=data.labels,
                    time=time))

                del word_vectors, doc_vectors, embeddings, time, labels_pred

            for metric in Metrics:
                report_df[
                    f"{word_model_cfg['name']}", f"{metric.value}"
                ][f"{doc_model_cfg['name']}"] = report.get(metric=metric)

            report.save(folder=f"{data_path}")

            del report, doc_model
        
        del word_model

    report_df.to_csv(path_or_buf=f"{data_path}/report.csv",
                     encoding="utf-8")
    print(tabulate(report_df))

    sp.ok("✔ ")
    
    print("Finished successfully!.. Exiting.")

    del report_df, sys.out
    sys.exit(0)