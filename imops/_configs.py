from .backend import Cython, Scipy


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
]
interp1d_configs = [
    Scipy(),
    *[Cython(fast) for fast in [False, True]],
]
