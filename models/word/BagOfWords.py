from os.path import basename, splitext, isfile
from utils.text import process_text, synonyms
from scipy.spatial.distance import cosine
from models.document import BestCMeans
from models import BagOfWords
from typing import Iterable
from pathlib import Path
import numpy as np


name = splitext(basename(__file__))[0]

# Default variables
confidenceUser = 50
termPercentileInit = 10.0 * 2 ** (2*(1-confidenceUser/50.0))


def model_path(dataset: str, id: int) -> str:
    return str(Path(
        f"./data/{dataset}/word/{name}/{id:03}.bin"
    ).resolve())


def load_model(dataset: str, id: int) -> BagOfWords:
    path = model_path(dataset=dataset, id=id)
    if isfile(path):
        return BagOfWords.load(path)
    return None


def save_model(dataset: str, id: int, model: BagOfWords):
    model.save(model_path(dataset=dataset, id=id))


def train_model(dataset: str, id: int, corpus: Iterable[str]) -> BagOfWords:
    model = BagOfWords()
    model.train(corpus=corpus)
    save_model(dataset=dataset, id=id, model=model)
    return model


def get_vectors(dataset: str, id: int, data: Iterable[str]) -> Iterable[Iterable[float]]:
    model = load_model(dataset=dataset, id=id)
    if not model:
        model = train_model(dataset=dataset, id=id, corpus=data)

    return np.transpose(model.matrix)


def cluster(dataset: str, id: int,
            k: int,
            seed: dict = None) -> Iterable[Iterable[float]]:
    def handle_unseen_words(words: list) -> list:
        words_filtered = [
            *filter(lambda word: word in model.vocabulary, words)]
        if len(words) != 0 and len(words_filtered) == 0:
            synonyms = []
            for word in words:
                synonyms += [process_text(syn) for syn in synonyms(word)]
            synonyms = [
                *filter(lambda word: word in model.vocabulary, synonyms)]
            if len(synonyms) == 0:
                raise Exception(
                    "Neither the words nor its synonyms in the vocabulary")
            else:
                words_filtered = [*synonyms]

        return words_filtered

    # iKMeans clustering
    model = load_model(dataset=dataset, id=id)

    cmeansWords = 5

    clusterTerms = []
    if seed:
        for i in range(k):
            clusterTerms.append(handle_unseen_words([
                word["word"]
                for word in seed["cluster_words"][i]
                if word["weight"] > 0]))
    else:
        _, u, _, _, _, _, _ = BestCMeans().fit_predict(
            data=model.matrix,
            c=k,
            m=1.1,
            error=0.005,
            maxiter=50,
            init=None)

        for cluster in u:
            temp = []
            for j in range(0, cmeansWords):
                temp.append(model.vocabulary[cluster.argmax()])
                cluster[cluster.argmax()] = -1
            clusterTerms.append(temp)

    # calculate centroid of selected terms
    termCentroids = np.zeros((k, model.n))
    for index, clusterTerm in enumerate(clusterTerms):
        center = np.zeros(model.n)
        for term in clusterTerm:
            center += model.matrix[:, model.vocabulary.index(term)]
        center /= k
        termCentroids[index] = center

    return termCentroids


def seed_paragraph(dataset: str, id: int, centroid: Iterable, topn: int = 50) -> dict:
    model = load_model(dataset=dataset, id=id)

    # expand terms using Cosine distance over the column of document-term matrix
    termCentroidCosine = np.zeros(model.m)

    for termIndex in range(0, model.m):
        termCentroidCosine[termIndex] = cosine(
            centroid, model.matrix[:, termIndex])

    # upper bound for the number of terms of each cluster
    termPercentile = termPercentileInit * model.m / 100
    seedDocumentsTerms = np.zeros(model.m)

    center = termCentroidCosine
    average = np.average(center)
    minDistance = center.min()
    counter = 0
    indexes = list()
    while minDistance < average:
        min_idx = center.argmin()
        seedDocumentsTerms[min_idx] = center[min_idx]
        indexes.append(min_idx)
        counter += 1
        if counter > termPercentile:
            break
        minDistance = center[min_idx]
        center[min_idx] = 2

    return dict(
        paragraph=[model.vocabulary[i] for i in indexes],
        vector=seedDocumentsTerms)


def most_similar(dataset: str, id: int, positive: list, topn: int = 10) -> list:
    model = load_model(dataset=dataset, id=id)

    words_filtered = [*filter(lambda word: word in model.vocabulary, positive)]
    if len(positive) != 0 and len(words_filtered) == 0:
        syns = []
        for word in positive:
            syns += [process_text(syn) for syn in synonyms(word)]
        syns = [*filter(lambda word: word in model.vocabulary, syns)]
        if len(syns) == 0:
            raise Exception(
                "Neither the words nor its synonyms in the vocabulary")
        else:
            words_filtered = [*syns]

        positive = words_filtered

    centroid = np.mean([model[term] for term in positive], axis=0)

    termCentroidCosine = np.zeros(model.m)

    for index, term in enumerate(model.vocabulary):
        termCentroidCosine[index] = cosine(centroid, model[term])

    indexes = np.argsort(termCentroidCosine)

    result, i = [], 0
    while len(result) < 10:
        if not model.vocabulary[indexes[i]] in positive:
            result.append({
                "word": model.vocabulary[indexes[i]],
                "value": float(1.0 - termCentroidCosine[indexes[i]])})
        i += 1

    return result