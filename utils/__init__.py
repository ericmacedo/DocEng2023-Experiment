from joblib import Parallel, delayed
from importlib.util import (
    spec_from_file_location,
    module_from_spec)
from models import ModelType
from typing import Callable
from functools import wraps
from time import time
import numpy as np
import sys

class Logger(object):
    def __init__(self, path: str):
        self.terminal = sys.stdout
        self.file = open(path, "w")

        self.outputs = [self.terminal, self.file]
   
    def write(self, message: str):
        for output in self.outputs:
            output.write(message)
            output.flush()

    def flush(self):
        for output in self.outputs:
            output.flush()

    def __del__(self):
        self.file.close()

def batch_processing(fn: Callable, data: list, **kwargs) -> list:
    return Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(fn)(data=i, **kwargs) for i in data)

def sysout_it(s: str, level: int = 1):
    sys.stdout.write("{0}{1}".format("\t" * level, s))

def print_header(title: str, cols: int = 80):
    sys.stdout.write("{}\n".format("=" * cols))
    sys.stdout.write("#\t{}\n".format(title.capitalize()))
    sys.stdout.write("{}\n".format("=" * cols))


def load_model(dataset: str, model_type: ModelType, name: str):
    spec = spec_from_file_location(
        name,
        f"./models/{model_type.value}/{name}.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timing(fn: Callable):
    @wraps(fn)
    def wrap(*args, **kw) -> tuple:
        t0 = time()
        result = fn(*args, **kw)
        t = time()
        return (result, t - t0)
    return wrap

def l2_norm(data: list) -> np.array:
    data = np.array(data, dtype=float)
    dist = np.sqrt((data ** 2).sum(-1))[..., np.newaxis]
    return data / dist

def calculateSample(corpus_size: int) -> float:
    if corpus_size > 500:
        return 1e-5

    return 1 * (1.0 / (10 ** int(corpus_size/100)))