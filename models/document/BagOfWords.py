from os.path import basename, splitext, isfile
from scipy.spatial.distance import cosine
from typing import Iterable, List, Dict
from sklearn.cluster import KMeans
from models import BagOfWords
from pathlib import Path
import numpy as np

name = splitext(basename(__file__))[0]

# default parameters
confidenceUser = 50
documentPercentileInit = 20.0 * 2 ** (2*(1-confidenceUser/50.0))


def model_path(dataset: str, id: int) -> str:
    return str(Path(
        f"./data/{dataset}/document/{name}/{id:03}.bin"
    ).resolve())


def load_model(dataset: str, id: int) -> BagOfWords:
    path = model_path(dataset=dataset, id=id)
    if isfile(path):
        return BagOfWords.load(path)
    return None


def save_model(dataset: str, id: int, model: BagOfWords):
    model.save(model_path(dataset=dataset, id=id))


def train_model(dataset: str, id: int, corpus: Iterable[str]):
    model = BagOfWords()
    model.train(corpus=corpus)
    save_model(dataset=dataset, id=id, model=model)


def get_vectors(dataset: str,
                id: int,
                data: Iterable[str] = None) -> List[List[float]]:
    model = load_model(dataset=dataset, id=id)

    return model.predict(data=data) if data else model.matrix.tolist()


def cluster(dataset: str, id: int,
            seed_paragraphs: Iterable[Dict[str, Iterable]],
            k: int,
            embeddings: List) -> Iterable[int]:
    # iKMeans clustering
    embeddings = np.array(embeddings)
    n, m = embeddings.shape

    seedDocumentsTerms = [i["vector"] for i in seed_paragraphs]

    # select documents related to the selected terms and calculate the centroid of documents
    seedDocumentsTermsCosine = np.zeros((k, n))
    for index, centroid in enumerate(seedDocumentsTerms):
        for document_index in range(0, n):
            seedDocumentsTermsCosine[index, document_index] = cosine(
                centroid, embeddings[document_index, :])

    # upper bound for the number of terms of each cluster
    documentPercentile = documentPercentileInit * n / 100
    seedDocuments = np.zeros((k, m))
    for index, center in enumerate(seedDocumentsTermsCosine):
        average = np.average(center)
        minDistance = center.min()
        counter = 0
        while minDistance < average:
            min_idx = center.argmin()
            seedDocuments[index] += embeddings[min_idx, :]
            counter += 1
            if counter > documentPercentile:
                break
            minDistance = center[min_idx]
            center[min_idx] = 2
        seedDocuments[index] /= counter

    # run kmeans
    return KMeans(
        n_clusters=k,
        init=seedDocuments,
        n_init=1
    ).fit_predict(embeddings)
