from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score,
    fowlkes_mallows_score, silhouette_score,
    v_measure_score)

from typing import List, Sequence
from dataclasses import dataclass

from enum import Enum

class Metrics(Enum):
    SILHOUETTE  = "Silhouette Score"
    AMI         = "Adjusted Mutual Info Score"
    ARS         = "Adjusted rand Score"
    FMS         = "Fowlkes-Mallows Score"
    V           = "V-measure Score"
    TIME        = "Time"

@dataclass
class Metric:
    name: str
    short: str
    measure: float

class Silhouette(Metric):
    def __init__(self,
                 X: Sequence[Sequence[float]],
                 labels_true: List[int],
                 labels_pred: List[int]):
        self.name = Metrics.SILHOUETTE.value
        self.short = Metrics.SILHOUETTE.name
        self.measure = silhouette_score(X=X,
                                        labels=labels_pred,
                                        metric="euclidean")

class AMI(Metric):
    def __init__(self,
                 X: Sequence[Sequence[float]],
                 labels_true: List[int],
                 labels_pred: List[int]):
        self.name = Metrics.AMI.value
        self.short = Metrics.AMI.name
        self.measure = adjusted_mutual_info_score(labels_true=labels_true,
                                                  labels_pred=labels_pred)

class ARS(Metric):
    def __init__(self,
                 X: Sequence[Sequence[float]],
                 labels_true: List[int],
                 labels_pred: List[int]):
        self.name = Metrics.ARS.value
        self.short = Metrics.ARS.name
        self.measure = adjusted_rand_score(labels_true=labels_true,
                                           labels_pred=labels_pred)

class FMS(Metric):
    def __init__(self,
                 X: Sequence[Sequence[float]],
                 labels_true: List[int],
                 labels_pred: List[int]):
        self.name = Metrics.FMS.value
        self.short = Metrics.FMS.name
        self.measure = fowlkes_mallows_score(labels_true=labels_true,
                                             labels_pred=labels_pred)

class VMeasure(Metric):
    def __init__(self,
                 X: Sequence[Sequence[float]],
                 labels_true: List[int],
                 labels_pred: List[int]):
        self.name = Metrics.V.value
        self.short = Metrics.V.name
        self.measure = v_measure_score(labels_true=labels_true, 
                                       labels_pred=labels_pred)

class Time(Metric):
    def __init__(self, time: float):
        self.name = Metrics.TIME.value
        self.short = Metrics.TIME.name
        self.measure = time