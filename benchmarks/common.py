from enum import Enum


NUMS_THREADS_TO_BENCHMARK = list(range(1, 9))


def discard_arg(idx: int):
    def inner(f):
        def wrapper(*args):
            nonlocal idx
            if idx < 0:
                idx = len(args) + idx
            return f(*args[:idx], *args[idx + 1 :])

        return wrapper

    return inner


class IMAGE_TYPE_BENCHMARK(Enum):
    RAND = 1
    LUNGS = 2
    BRONCH = 3


IMAGE_TYPES_BENCHMARK = list(IMAGE_TYPE_BENCHMARK)
