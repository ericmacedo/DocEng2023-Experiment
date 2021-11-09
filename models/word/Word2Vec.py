from os.path import basename, splitext, isfile
from utils.text import process_text, synonyms
from multiprocessing import cpu_count
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from sklearn.utils import shuffle
from utils import calculateSample
from typing import Iterable
from pathlib import Path
import numpy as np

name = splitext(basename(__file__))[0]


def model_path(dataset: str, id: int) -> str:
    return str(Path(
        f"./data/{dataset}/word/{name}/{id:03}.bin"
    ).resolve())


def load_model(dataset: str, id: int) -> Word2Vec:
    path = model_path(dataset=dataset, id=id)
    if isfile(path):
        model = Word2Vec.load(path)
        return model
    return None


def save_model(dataset: str, id: int, model: Word2Vec):
    model.save(model_path(dataset=dataset, id=id))


def train_model(dataset: str, id: int, corpus: Iterable[str]) -> Word2Vec:
    sentences = [doc.split(" ") for doc in corpus]

    corpus_size = len(corpus)

    model = Word2Vec(
        window=8,
        min_count=5,
        size=100,
        alpha=0.025,
        min_alpha=0.0007,
        sample=calculateSample(corpus_size),
        hs=1,
        sg=1,
        negative=15,
        ns_exponent=0.75,
        workers=cpu_count(),
        iter=40)

    model.build_vocab(sentences=sentences)

    model.train(
        shuffle(sentences),
        total_examples=corpus_size,
        epochs=40)

    model.wv.init_sims(replace=True)
    save_model(dataset=dataset, id=id, model=model)
    return model


def get_vectors(dataset: str, id: int,
                data: Iterable[str] = None) -> Iterable[Iterable[float]]:
    model = load_model(dataset=dataset, id=id)
    return model.wv.vectors.tolist()


def cluster(dataset: str, id: int,
            k: int,
            seed: dict = None) -> Iterable[Iterable[float]]:
    def handle_unseen_words(words: list) -> list:
        words_filtered = [
            *filter(lambda word: word in model.wv, words)]
        if len(words) != 0 and len(words_filtered) == 0:
            synonyms = []
            for word in words:
                synonyms += [process_text(syn) for syn in synonyms(word)]
            synonyms = [
                *filter(lambda word: word in model.wv, synonyms)]
            if len(synonyms) == 0:
                raise Exception(
                    "Neither the words nor its synonyms in the vocabulary")
            else:
                words_filtered = [*synonyms]

        return words_filtered

    model = load_model(dataset=dataset, id=id)

    if seed:  # UPDATE CLUSTERS GIVEN USER SEEDS
        init_mode = np.zeros((k, model.vector_size))
        for i in range(k):
            positive = []
            negative = []
            for word in seed["cluster_words"][i]:
                if word["weight"] > 0:
                    positive.append(word["word"])
                else:
                    negative.append(word["word"])

            # Handling unseen words for Word2Vec
            # FastText can handle unseen words
            positive = handle_unseen_words(positive)
            negative = handle_unseen_words(negative)

            seed_terms = positive + [
                term[0]
                for term in model.wv.most_similar(
                    positive=(positive if positive else None),
                    negative=(negative if negative else None),
                    topn=(50 - len(positive)))
            ]

            init_mode[i] = np.mean([
                model.wv.word_vec(term)
                for term in seed_terms
            ], axis=0)
    else:  # NEW RANDOM CLUSTERS
        init_mode = "k-means++"

    # Clustering word vectors
    k_means = KMeans(
        n_clusters=k,
        init=init_mode,
        n_init=10,
        tol=1e-5
    ).fit(model.wv.vectors)

    return k_means.cluster_centers_


def seed_paragraph(dataset: str, id: int, centroid: Iterable, topn: int = 50) -> dict:
    model = load_model(dataset=dataset, id=id)

    return dict(
        paragraph=[
            term[0]
            for term in model.wv.similar_by_vector(centroid, topn=topn)]
    )


def most_similar(dataset: str, id: int, positive: list, topn: int = 10) -> list:
    model = load_model(dataset=dataset, id=id)

    words_filtered = [*filter(lambda word: word in model.wv, positive)]
    if len(positive) != 0 and len(words_filtered) == 0:
        syns = []
        for word in positive:
            syns += [process_text(syn) for syn in synonyms(word)]
        syns = [*filter(lambda word: word in model.wv, syns)]
        if len(syns) == 0:
            raise Exception(
                "Neither the words nor its synonyms in the vocabulary")
        else:
            words_filtered = [*syns]

        positive = words_filtered

    sim_wors = model.wv.most_similar(positive=positive, topn=topn)

    return [
        {"word": word[0], "value": word[1]}
        for word in sim_wors]