from timeit import default_timer as timer
from types import GeneratorType
from typing import Mapping

import numpy as np


def eval_func(func, inp):
    if isinstance(inp, Mapping):

        def inner(inp):
            return func(**inp)

    elif isinstance(inp, GeneratorType):

        def inner(inp):
            return func(*inp)

    else:

        def inner(inp):
            return func(*inp)

    return inner


def timeit(func, inp, tag="", warmup=10, nit=100):
    timer = Timer(tag=tag)
    inner = eval_func(func, inp)

    for _ in range(warmup):
        inner(inp)
    for _ in range(nit):
        with timer:
            inner(inp)
    return timer


class Timer:
    def __init__(self, tag="", logger=None):
        self.tag = tag
        self.elapsed = []
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = timer()

    def __exit__(self, type, value, traceback):
        self.end = timer()
        self.elapsed.append(self.end - self.start)

    def mean(self):
        return np.mean(self.elapsed)

    def stdev(self):
        return np.std(self.elapsed)

    def min(self):
        return np.min(self.elapsed)

    def max(self):
        return np.max(self.elapsed)

    def samples(self):
        return self.elapsed

    def dumps(self):
        data = {
            "tag": self.tag,
            "mean": self.mean(),
            "stdev": self.stdev(),
            "min": self.min(),
            "max": self.max(),
            "samples": self.samples(),
        }
        return data

    def __repr__(self) -> str:
        timings = self.dumps()
        return f'{timings["tag"]} ' + " / ".join(
            [
                f"{k}={timings[k]*1000:.5f} [ms]"
                for k in ["mean", "stdev", "min", "max"]
            ],
        )
