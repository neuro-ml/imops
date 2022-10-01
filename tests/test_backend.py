import pytest

from imops.backend import resolve_backend, Cython, Numba, Scipy, set_backend, imops_backend


def test_resolve():
    assert resolve_backend(None) == Cython()

    for cls in [Numba, Cython, Scipy]:
        assert resolve_backend(cls) == cls()
        assert resolve_backend(cls.__name__) == cls()

    with pytest.raises(ValueError):
        resolve_backend(1)


def test_backend_change():
    assert resolve_backend(None) == Cython()
    set_backend('Numba')
    assert resolve_backend(None) == Numba()
    set_backend(Cython(fast=True))
    assert resolve_backend(None) == Cython(fast=True)
    set_backend('Cython')
    assert resolve_backend(None) == Cython()

    with imops_backend('Scipy'):
        assert resolve_backend(None) == Scipy()

    assert resolve_backend(None) == Cython()
