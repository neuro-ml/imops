import contextlib
from dataclasses import dataclass
from typing import Dict, Type, Union


class Backend:
    def __init_subclass__(cls, **kwargs):
        name = cls.__name__
        if name in AVAILABLE_BACKENDS:
            raise ValueError(f'The name "{name}" is already in use.')
        AVAILABLE_BACKENDS[name] = cls
        if not hasattr(Backend, name):
            setattr(Backend, name, cls)

    @property
    def name(self):
        return type(self).__name__

    Cython: 'Cython'
    Numba: 'Numba'
    Scipy: 'Scipy'


BackendLike = Union[str, Backend, Type[Backend], None]
AVAILABLE_BACKENDS: Dict[str, Type[Backend]] = {}


def resolve_backend(value: BackendLike) -> Backend:
    if value is None:
        return DEFAULT_BACKEND

    if isinstance(value, str):
        if value not in AVAILABLE_BACKENDS:
            raise ValueError(f'"{value}" is not in the list of available backends: {tuple(AVAILABLE_BACKENDS)}.')

        return AVAILABLE_BACKENDS[value]()

    if isinstance(value, type):
        value = value()

    if not isinstance(value, Backend):
        raise ValueError(f'Expected a `Backend` instance, got {value}.')

    return value


def set_backend(backend: BackendLike) -> Backend:
    global DEFAULT_BACKEND
    current = DEFAULT_BACKEND
    DEFAULT_BACKEND = resolve_backend(backend)
    return current


@contextlib.contextmanager
def imops_backend(backend: BackendLike):
    previous = set_backend(backend)
    try:
        yield
    finally:
        set_backend(previous)


# implementations
# TODO: Investigate whether it is safe to use -ffast-math in numba
@dataclass
class Numba(Backend):
    parallel: bool = True
    nogil: bool = True
    cache: bool = True

    def __post_init__(self):
        try:
            import numba  # noqa: F401
        except ModuleNotFoundError:  # pragma: no cover
            raise ModuleNotFoundError('Install `numba` package (pip install numba) to use "numba" backend.')


@dataclass
class Cython(Backend):
    fast: bool = False


@dataclass
class Scipy(Backend):
    pass


DEFAULT_BACKEND = Cython()

BACKEND2NUM_THREADS_VAR_NAME = {Cython.__name__: 'OMP_NUM_THREADS', Numba.__name__: 'NUMBA_NUM_THREADS'}
SINGLE_THREADED_BACKENDS = (Scipy.__name__,)
