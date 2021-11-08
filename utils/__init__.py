from joblib import Parallel, delayed
from importlib.util import (
    spec_from_file_location,
    module_from_spec)
from models import ModelType
from typing import Callable
from functools import wraps
from time import time
import sys

def batch_processing(fn: Callable, data: list, **kwargs) -> list:
    return Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(fn)(data=i, **kwargs) for i in data)

def sysout_it(s: str, level: int = 1):
    sys.stdout.write(f"{'\t' * level}{s}")

def print_header(title: str, cols: int = 80):
    sys.stdout.write(f"{'=' * cols}")
    sys.stdout.write(f"#\t{title.capitalize()}")
    sys.stdout.write(f"{'=' * cols}")


def load_model(dataset: str, model_type: ModelType, name: str):
    spec = spec_from_file_location(
        name,
        f"./data/{dataset}/{model_type.value}/{name}.py")
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

class Logger(object):
    def __init__(self, path: str):
        self.terminal = sys.stdout
        self.log = open(path, "a")
   
    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    