from models.metrics import (
    Metric, Metrics,
    Silhouette, AMI, ARS, FMS, VMeasure, Time)
from dataclasses import dataclass, field
from typing import Sequence, List, Dict
from numpy import mean, std
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
        self.AMI        = AMI(**params)
        self.ARS        = ARS(**params)
        self.FMS        = FMS(**params)
        self.VMEASURE   = VMeasure(**params)
        self.TIME       = Time(time=time)

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
    name: str
    embeddings: Sequence[Sequence[float]]
    labels_true: List[int]
    labels_pred: List[int]
    metrics: Evaluation

    def __init__(self, 
                 name: str,
                 X: Sequence[Sequence[float]],
                 labels_true: List[int],
                 labels_pred: List[int],
                 time: float):

        self.name = name
        self.embeddings = X
        self.labels_true = labels_true
        self.labels_pred = labels_pred
        self.metrics = Evaluation(X=X,
                                  labels_true=labels_true,
                                  labels_pred=labels_pred,
                                  time=time)


class Report:
    name: str
    samples: List[Clustering] = field(default_factory=list)

    def __init__(self, name: str):
        self.name = name
        self.samples = []

    def append(self, clustering: Clustering):
        self.samples.append(clustering)

    def get(self, metric: Metrics) -> str:
        observations = [
            sample.metrics.get(metric=metric).measure
            for sample in self.samples]
        return "{mean:.4f} ± {std:.4f}".format(
            mean=mean(observations, axis=0),
            std=std(observations, axis=0))

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
