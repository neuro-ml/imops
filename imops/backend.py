from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Type, Union
from warnings import warn


class Backend:
    def __init_subclass__(cls, **kwargs):
        name = cls.__name__
        if name in _AVAILABLE_BACKENDS:
            raise ValueError(f'The name "{name}" is already in use.')
        _AVAILABLE_BACKENDS[name] = cls
        if not hasattr(Backend, name):
            setattr(Backend, name, cls)

    @property
    def name(self):
        return type(self).__name__

    Cython: 'Cython'
    Numba: 'Numba'
    Scipy: 'Scipy'


BackendLike = Union[str, Backend, Type[Backend], None]
_AVAILABLE_BACKENDS: Dict[str, Type[Backend]] = {}


def resolve_backend(value: BackendLike, warn_stacklevel: int = 1) -> Backend:
    if value is None:
        return DEFAULT_BACKEND

    if isinstance(value, str):
        if value not in _AVAILABLE_BACKENDS:
            raise ValueError(f'"{value}" is not in the list of available backends: {tuple(_AVAILABLE_BACKENDS)}.')

        return _AVAILABLE_BACKENDS[value]()

    if isinstance(value, type):
        value = value()

    if not isinstance(value, Backend):
        raise ValueError(f'Expected a `Backend` instance, got {value}.')

    if isinstance(value, Cython) and value.fast:
        warn('`fast=True` has no effect for `Cython` backend for now.', stacklevel=warn_stacklevel)

    return value


def set_backend(backend: BackendLike) -> Backend:
    global DEFAULT_BACKEND
    current = DEFAULT_BACKEND
    DEFAULT_BACKEND = resolve_backend(backend)
    return current


@contextmanager
def imops_backend(backend: BackendLike):
    previous = set_backend(backend)
    try:
        yield
    finally:
        set_backend(previous)


# implementations
# TODO: Investigate whether it is safe to use -ffast-math in numba
@dataclass(frozen=True)
class Numba(Backend):
    parallel: bool = True
    nogil: bool = True
    cache: bool = True

    def __post_init__(self):
        try:
            import numba  # noqa: F401
        except ModuleNotFoundError:  # pragma: no cover
            raise ModuleNotFoundError('Install `numba` package (pip install numba) to use "numba" backend.')


@dataclass(frozen=True)
class Cython(Backend):
    fast: bool = False


@dataclass(frozen=True)
class Scipy(Backend):
    pass


DEFAULT_BACKEND = Cython()

BACKEND_NAME2ENV_NUM_THREADS_VAR_NAME = {Cython.__name__: 'OMP_NUM_THREADS', Numba.__name__: 'NUMBA_NUM_THREADS'}
SINGLE_THREADED_BACKENDS = (Scipy.__name__,)
