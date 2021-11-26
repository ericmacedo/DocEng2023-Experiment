from sklearn.metrics import silhouette_score as Silhouette
from dataclasses import dataclass, field
from typing import Sequence, List, Dict
from numpy import mean, std
from openTSNE import TSNE
from pathlib import Path
from utils import timing
import numpy as np
import pickle

from utils import timing


@dataclass
class TSNEData:
    embedding: np.array
    time: float
    perplexity: int

    def __init__(self, embeddings: list,
                 dataset: str, doc_model: str, id: int,
                 perplexity: int = 30):
        self.perplexity = perplexity
        self.embedding, self.time = self.__train(embeddings=embeddings)

        self.__path = Path(
            f"./data/{dataset}/projection/{doc_model}/{id:03}.bin"
        ).resolve()

    @timing
    def __train(self, embeddings: Sequence) -> np.array:
        return TSNE(
            n_components=2,
            perplexity=self.perplexity,
            metric="euclidean",
            n_jobs=-1
        ).fit(embeddings)

    def save(self):
        with open(self.__path, "wb") as pkl_file:
            pickle.dump(
                obj=self,
                file=pkl_file,
                protocol=pickle.DEFAULT_PROTOCOL,
                fix_imports=True)

    @classmethod
    def load(cls, dataset: str, doc_model: str, id: int):
        path = Path(
            f"./data/{dataset}/projection/{doc_model}/{id:03}.bin"
        ).resolve()
        return pickle.load(open(path, "rb"))


@dataclass
class Evaluation:
    SILHOUETTE_true: float
    SILHOUETTE_pred: float
    TIME: float
    __metrics: Dict[str, any] = field(default_factory=dict())

    def __init__(self, X: Sequence[Sequence[float]],
                 labels_true: List[int],
                 labels_pred: List[int],
                 time: float):

        self.SILHOUETTE_true = Silhouette(X=X, labels=labels_true)
        self.SILHOUETTE_pred = Silhouette(X=X, labels=labels_pred)
        self.TIME = time

        self.__metrics = {
            "true": self.SILHOUETTE_true,
            "pred": self.SILHOUETTE_pred,
            "time": self.TIME}

    def get(self, metric: str) -> float:
        return self.__metrics.get(metric, None)


@dataclass
class Projection:
    id: int
    embeddings: Sequence[Sequence[float]]
    labels_true: List[int]
    labels_pred: List[int]
    metrics: Evaluation

    def __init__(self,
                 id: int,
                 X: Sequence[Sequence[float]],
                 labels_true: List[int],
                 labels_pred: List[int],
                 time: float):

        self.id = id
        self.embeddings = X
        self.labels_true = labels_true
        self.labels_pred = labels_pred
        self.metrics = Evaluation(X=X,
                                  labels_true=labels_true,
                                  labels_pred=labels_pred,
                                  time=time)

    def save(self, folder: str):
        with open(f"{folder}/{self.id:03}.bin", "wb") as pkl_file:
            pickle.dump(
                obj=self,
                file=pkl_file,
                protocol=pickle.DEFAULT_PROTOCOL,
                fix_imports=True)

    @classmethod
    def load(cls, folder: str, id: int):
        return pickle.load(open(f"{folder}/{id:03}.bin", "rb"))


class Report:
    name: str
    samples_path: str

    def __init__(self, dataset: str, doc_model: str, name: str):
        self.name = name

        self.samples_path = str(Path(
            f"./data/{dataset}/projection/{name}/{doc_model}"
        ).resolve())

        Path(self.samples_path).mkdir(parents=True, exist_ok=True)

    def append(self, projection: Projection):
        projection.save(folder=self.samples_path)
        del projection

    def get(self, metric: str) -> str:
        observations = [sample.metrics.get(metric=metric)
                        for sample in self]
        return "{mean:.4f} ± {std:.4f}".format(
            mean=mean(observations, axis=0),
            std=std(observations, axis=0))

    def __len__(self) -> int:
        files = Path(self.samples_path).glob("*.bin")
        return len([f for f in files if f.is_file()])

    def __getitem__(self, index: int) -> Projection:
        n = len(self)
        if isinstance(index, int):
            if index > n:
                raise IndexError
            return Projection.load(folder=self.samples_path, id=id)
        else:
            raise TypeError("Invalid index type")

    def __iter__(self):
        n = len(self)
        for index in range(n):
            yield Projection.load(folder=self.samples_path, id=index)

    def save(self, folder: str):
        with open(f"{folder}/{self.name}.pkl", "wb") as f_pkl:
            pickle.dump(obj=self,
                        file=f_pkl,
                        fix_imports=True)
