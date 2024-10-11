from itertools import product

from .backend import Cupy, Cython, Numba, Scipy


scipy_configs = [Scipy()]
radon_configs = [Cython(fast) for fast in [False, True]]
numeric_configs = [
    Scipy(),
    *[Cython(fast) for fast in [False, True]],
]
measure_configs = [
    Scipy(),
    *[Cython(fast) for fast in [False, True]],
]
morphology_configs = [
    Scipy(),
    *[Cython(fast) for fast in [False, True]],
]
zoom_configs = [
    Scipy(),
    *[Cython(fast) for fast in [False, True]],
    *[Numba(*flags) for flags in product([False, True], repeat=3)],
    Cupy(),
]
interp1d_configs = [
    Scipy(),
    *[Cython(fast) for fast in [False, True]],
    *[Numba(*flags) for flags in product([False, True], repeat=3)],
]
