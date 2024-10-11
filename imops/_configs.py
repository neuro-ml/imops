from functools import partial
from itertools import product

from .backend import Cupy, Cython, Numba, Scipy


scipy_backends = [Scipy]
radon_backends = [partial(Cython, fast=fast) for fast in [False, True]]
numeric_backends = [
    Scipy,
    *[partial(Cython, fast=fast) for fast in [False, True]],
]
measure_backends = [
    Scipy,
    *[partial(Cython, fast=fast) for fast in [False, True]],
]
morphology_backends = [
    Scipy,
    *[partial(Cython, fast=fast) for fast in [False, True]],
]
zoom_backends = [
    Scipy,
    *[partial(Cython, fast=fast) for fast in [False, True]],
    *[partial(Numba, *flags) for flags in product([False, True], repeat=3)],
    Cupy,
]
interp1d_backends = [
    Scipy,
    *[partial(Cython, fast=fast) for fast in [False, True]],
    *[partial(Numba, *flags) for flags in product([False, True], repeat=3)],
]


def is_available(backend):
    try:
        backend()
        return True
    except ModuleNotFoundError:
        return False


def available_backends(backends):
    return [backend for backend in backends if is_available(backend)]


def to_repr(backend):
    if not isinstance(backend, partial):
        return backend

    backend_name = backend.func.__name__
    backend_args = ', '.join(map(str, backend.args))
    backend_kwargs = ', '.join(f'{key}={value}' for key, value in backend.keywords.items())

    if not backend_args:
        if not backend_kwargs:
            return backend_name
        else:
            return f'{backend_name}({backend_kwargs})'
    else:
        if not backend_kwargs:
            return f'{backend_name}({backend_args})'

    return f'{backend_name}({backend_args}, {backend_kwargs})'
