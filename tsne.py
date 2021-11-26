from models.projection import Report, TSNEData, Projection
from models.document import DocumentModels
from models.clustering import Clustering
from models import ModelType, models
from models.word import WordModels
from tabulate import tabulate
from utils import load_model
from yaspin import yaspin
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import sys

# ============================================================================
#   Arguments
# ============================================================================
parser = argparse.ArgumentParser()

parser.add_argument("--name", "-n", type=str, dest="name", required=True,
                    help="Name of the dataset (available located in ./data)")
parser.add_argument("--perplexity", "-p", type=int, dest="perplexity",
                    help="Perplexity for TSNE", default=30)
parser.add_argument("--sample-size", "-s", type=int, default=100,
                    dest="sample_size",
                    help="Number of observations to generate")
parser.add_argument("--skip-training", "-t", type=bool, default=False,
                    dest="skip_training", help="Skip training t-SNE")

args = parser.parse_args()

perplexity = args.perplexity
dataset = args.name
repeat = args.sample_size
skip_training = args.skip_training

data = pd.read_csv(f"./data/{dataset}/data.csv", encoding="utf-8")

if not skip_training:
    with yaspin(text="Running t-SNE", color="cyan") as sp:
        doc_models = [*filter(
            lambda m: m['model_type'] == ModelType.DOCUMENT, models)]

        for doc_model_cfg in doc_models:
            doc_model_cfg["dataset"] = dataset
            doc_model = load_model(**doc_model_cfg)
            Path(
                f"./data/{dataset}/projection/{doc_model_cfg['name']}"
            ).resolve().mkdir(parents=True, exist_ok=True)

            sp.write(f"> Running t-SNE on {doc_model_cfg['name']}")
            report = Report(name="TSNE",
                            doc_model=doc_model_cfg['name'],
                            dataset=dataset)
            for id in range(repeat):
                sp.text = f"Training t-SNE iteration {id}"

                embeddings = doc_model.get_vectors(id=id,
                                                   dataset=dataset,
                                                   data=None)
                tsne = TSNEData(embeddings=np.array(embeddings),
                                perplexity=perplexity,
                                dataset=dataset,
                                doc_model=doc_model_cfg['name'],
                                id=id)
                tsne.save()

                del embeddings, tsne

    sp.ok("✔ ")

with yaspin(text="Clustering", color="cyan") as sp:
    print("Clustering models".upper())
    word_models = [*filter(
        lambda m: m['model_type'] == ModelType.WORD, models)]
    doc_models = [*filter(
        lambda m: m['model_type'] == ModelType.DOCUMENT, models)]

    metrics_list = ["true", "pred", "time"]

    pd_index = list(zip(*[
        [i for m in WordModels for i in [m.value] * len(metrics_list)],
        metrics_list * 3
    ]))

    pd_index = pd.MultiIndex.from_tuples(pd_index)

    pd_cols = [m.value for m in DocumentModels]

    report_df = pd.DataFrame(index=pd_index, columns=pd_cols, dtype=str)

    for word_model_cfg in word_models:
        for doc_model_cfg in doc_models:

            sp.write("> Calculating metrics for {0} x {1}".format(
                word_model_cfg['name'], doc_model_cfg['name']))
            report = Report(name="TSNE",
                            doc_model=doc_model_cfg['name'],
                            dataset=dataset)
            for id in range(repeat):
                sp.text = f"Clustering iteration {id}"
                clustering_path = Path("./data/{0}/clustering/{1}_{2}/".format(
                    dataset, word_model_cfg['name'], doc_model_cfg['name']
                )).resolve()

                clustering = Clustering.load(folder=clustering_path, id=id)

                tsne = TSNEData.load(dataset=dataset,
                                     doc_model=doc_model_cfg['name'],
                                     id=id)

                report.append(Projection(
                    id=id,
                    X=tsne.embedding,
                    labels_pred=clustering.labels_pred,
                    labels_true=clustering.labels_true,
                    time=tsne.time))

                del clustering_path, clustering, tsne

            for metric in metrics_list:
                report_df.loc[
                    f"{word_model_cfg['name']}", f"{metric}"
                ][f"{doc_model_cfg['name']}"] = report.get(metric=metric)

            del report

    report_df.to_csv(path_or_buf=f"./data/{dataset}/projection.csv",
                     encoding="utf-8")
    print(tabulate(report_df))

    sp.ok("✔ ")


print("Finished successfully!.. Exiting.")

del report_df, sys.stdout
sys.exit(0)
