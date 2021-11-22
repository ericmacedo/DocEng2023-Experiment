from models.metrics import (
    Metric, Metrics,
    Silhouette, AMI, ARS, FMS, VMeasure, Time)
from dataclasses import dataclass, field
from typing import Sequence, List, Dict
from numpy import mean, std
from pathlib import Path
from utils import timing
import pickle


@dataclass
class Evaluation:
    SILHOUETTE: Silhouette
    AMI: AMI
    ARS: ARS
    FMS: FMS
    VMEASURE: VMeasure
    TIME: Time
    __metrics: Dict[Metrics, Metric] = field(default_factory=dict())

    def __init__(self, X: Sequence[Sequence[float]],
                 labels_true: List[int],
                 labels_pred: List[int],
                 time: float):
        params = dict(X=X, labels_true=labels_true, labels_pred=labels_pred)

        self.SILHOUETTE = Silhouette(**params)
        self.AMI = AMI(**params)
        self.ARS = ARS(**params)
        self.FMS = FMS(**params)
        self.VMEASURE = VMeasure(**params)
        self.TIME = Time(time=time)

        self.__metrics = {
            Metrics.SILHOUETTE: self.SILHOUETTE,
            Metrics.AMI: self.AMI,
            Metrics.ARS: self.ARS,
            Metrics.FMS: self.FMS,
            Metrics.V: self.VMEASURE,
            Metrics.TIME: self.TIME}

    def get(self, metric: Metrics) -> Metric:
        return self.__metrics.get(metric, None)


@dataclass
class Clustering:
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
        return pickle.load(open(f"{folder}/{id:03}.bin", "wb"))


class Report:
    name: str
    __samples_path: str

    def __init__(self, dataset: str, name: str):
        self.name = name
        self.dataset = dataset

        self.__samples_path = f"./data/{dataset}/clustering/{name}"

        Path(self.__samples_path).resolve().mkdir(parents=True, exist_ok=True)

    def append(self, clustering: Clustering):
        clustering.save(folder=self.__samples_path)
        del clustering

    def get(self, metric: Metrics) -> str:
        observations = [sample.metrics.get(metric=metric) for sample in self]
        return "{mean:.4f} ± {std:.4f}".format(
            mean=mean(observations, axis=0),
            std=std(observations, axis=0))

    def __len__(self) -> int:
        files = Path(self.__samples_path).glob("*.bin")
        return len([f for f in files if f.is_file()])

    def __getitem__(self, index: int) -> Clustering:
        n = len(self)
        if isinstance(index, int):
            if index > n:
                raise IndexError
            return Clustering.load(folder=self.__samples_path, id=id)
        else:
            raise TypeError("Invalid index type")

    def __iter__(self):
        n = len(self)
        for index in range(n):
            yield Clustering.load(folder=self.__samples_path, id=index)

    def save(self, folder: str):
        with open(f"{folder}/{self.name}.pkl", "wb") as f_pkl:
            pickle.dump(obj=self,
                        file=f_pkl,
                        fix_imports=True)


@timing
def clusterer(dataset: str, id: int,
              word_vectors,
              doc_vectors,
              embeddings: List[List[float]],
              k: int) -> List[int]:
    seed_paragraphs = []
    w_centers = word_vectors.cluster(id=id, dataset=dataset, k=k)

    # Builds seed paragraphs to cluster documents
    seed_paragraphs = [
        word_vectors.seed_paragraph(
            id=id,
            dataset=dataset,
            centroid=centroid
        ) for centroid in w_centers]

    # If word_vectors is Bag-of-Words:
    # seed_paragraph has "vector"
    # Else
    # Genereate "vector" for paragraphs
    if not [p["vector"] for p in seed_paragraphs if "vector" in p]:
        vectors = doc_vectors.get_vectors(
            id=id,
            dataset=dataset,
            data=[
                " ".join(p["paragraph"])
                for p in seed_paragraphs])

        for i in range(k):
            seed_paragraphs[i]["vector"] = vectors.pop(0)

    return doc_vectors.cluster(
        id=id,
        k=k,
        dataset=dataset,
        seed_paragraphs=seed_paragraphs,
        embeddings=embeddings)
