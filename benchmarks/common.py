from enum import Enum
from gzip import GzipFile

from numpy import load


class IMAGE_TYPE_BENCHMARK(Enum):
    RAND = 1
    LUNGS = 2
    BRONCHI = 3

    def __repr__(self):
        return self.name


def discard_arg(idx: int):
    def inner(f):
        def wrapper(*args):
            nonlocal idx
            if idx < 0:
                idx = len(args) + idx
            return f(*args[:idx], *args[idx + 1 :])

        return wrapper

    return inner


def load_npy_gz(path):
    with GzipFile(path, 'rb') as f:
        return load(f)


IMAGE_TYPES_BENCHMARK = list(IMAGE_TYPE_BENCHMARK)
NUMS_THREADS_TO_BENCHMARK = list(range(1, 9))
