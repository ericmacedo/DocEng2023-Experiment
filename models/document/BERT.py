from os.path import basename, splitext, isfile
from typing import List, Iterable, Dict
from sklearn.cluster import KMeans
from models.document import Bert
from pathlib import Path
import numpy as np


name = splitext(basename(__file__))[0]


def model_path(dataset: str, id: int) -> str:
    return str(Path(
        f"./data/{dataset}/document/{name}/{id:03}.bin"
    ).resolve())


def load_model(dataset: str, id: int) -> Bert:
    path = model_path(dataset=dataset, id=id)
    if isfile(path):
        return Bert.load(path)
    return None


def save_model(dataset: str, id: int, model: Bert):
    model.save(model_path(dataset=dataset, id=id))


def train_model(dataset: str, id: int, corpus: Iterable[str]):
    model = Bert(model_name="allenai-specter")

    model.train(corpus=corpus)
    save_model(dataset=dataset, id=id, model=model)


def get_vectors(dataset: str,
                id: int,
                data: Iterable[str]) -> List[List[float]]:
    model = load_model(dataset=dataset, id=id)

    return model.predict(data=data) if data else model.embeddings


def cluster(dataset: str, id: int,
            seed_paragraphs: Iterable[Dict[str, Iterable]],
            k: int,
            embeddings: List) -> Iterable[int]:
    doc_seeds = get_vectors(
        dataset=dataset,
        id=id,
        data=[" ".join(p["paragraph"]) for p in seed_paragraphs])

    return KMeans(
        n_clusters=k,
        init=np.array(doc_seeds, dtype=np.float32),
        n_init=10,
        tol=1e-5
    ).fit_predict(embeddings)
