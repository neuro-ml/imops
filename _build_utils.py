import platform
import shutil
from pathlib import Path

from setuptools import Extension
from setuptools.command.build_py import build_py


class NumpyImport(dict):
    """Hacky way to return Numpy's `include` path with lazy import."""

    # Must be json-serializable due to
    # https://github.com/cython/cython/blob/6ad6ca0e9e7d030354b7fe7d7b56c3f6e6a4bc23/Cython/Compiler/ModuleNode.py#L773
    def __init__(self):
        return super().__init__(self, description=self.__doc__)

    # Must be hashable due to
    # https://github.com/cython/cython/blob/6ad6ca0e9e7d030354b7fe7d7b56c3f6e6a4bc23/Cython/Compiler/Main.py#L307
    def __hash__(self):
        return id(self)

    def __repr__(self):
        import numpy as np

        return np.get_include()

    __fspath__ = __repr__


class NumpyLibImport(str):
    """Hacky way to return Numpy's `lib` path with lazy import."""

    # Exploit of https://github.com/pypa/setuptools/blob/1ef36f2d336e239bd8f83507cb9447e060b6ed60/setuptools/_distutils/
    # unixccompiler.py#L276-L277
    def __radd__(self, left):
        import numpy as np

        return left + str(Path(np.get_include()).parent / 'lib')

    def __hash__(self):
        return id(self)


class PyprojectBuild(build_py):
    def run(self):
        self.run_command('build_ext')
        return super().run()

    def initialize_options(self):
        super().initialize_options()

        self.distribution.ext_modules = get_ext_modules()


def get_ext_modules():
    name = 'imops'
    on_windows = platform.system() == 'Windows'
    args = ['/openmp' if on_windows else '-fopenmp']

    # Cython extension and .pyx source file names must be the same to compile
    # https://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu
    modules = ['backprojection', 'measure', 'morphology', 'numeric', 'radon', 'zoom']
    modules_to_link_against_numpy_core_math_lib = ['numeric']

    for module in modules:
        src_dir = Path(__file__).parent / name / 'src'
        shutil.copyfile(src_dir / f'_{module}.pyx', src_dir / f'_fast_{module}.pyx')

    return [
        Extension(
            f'{name}.src._{prefix}{module}',
            [f'{name}/src/_{prefix}{module}.pyx'],
            include_dirs=[NumpyImport()],
            library_dirs=[NumpyLibImport()] if module in modules_to_link_against_numpy_core_math_lib else [],
            libraries=['npymath'] + ['m'] * (not on_windows)
            if module in modules_to_link_against_numpy_core_math_lib
            else [],
            extra_compile_args=args + additional_args,
            extra_link_args=args + additional_args,
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        )
        for module in modules
        # FIXME: import of `ffast-math` compiled modules changes global FPU state, so now `fast=True` will just
        # fallback to standard `-O2` compiled versions until https://github.com/neuro-ml/imops/issues/37 is resolved
        # for prefix, additional_args in zip(['', 'fast_'], [[], ['-ffast-math']])
        for prefix, additional_args in zip(['', 'fast_'], [[], []])
    ]
