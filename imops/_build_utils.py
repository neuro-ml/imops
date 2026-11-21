import os
import platform
import shutil
from pathlib import Path

from setuptools import Extension
from setuptools.command.build_py import build_py


class LazyImport(dict):
    """Hacky way to return any module's `include` path with lazy import."""

    # Must be json-serializable due to
    # https://github.com/cython/cython/blob/6ad6ca0e9e7d030354b7fe7d7b56c3f6e6a4bc23/Cython/Compiler/ModuleNode.py#L773
    def __init__(self, module_name):
        self.module_name = module_name
        super().__init__(self, description=self.__doc__)

    # Must be hashable due to
    # https://github.com/cython/cython/blob/6ad6ca0e9e7d030354b7fe7d7b56c3f6e6a4bc23/Cython/Compiler/Main.py#L307
    def __hash__(self):
        return id(self)

    def __repr__(self):
        scope = {}
        exec(f'import {self.module_name} as module', scope)

        return scope['module'].get_include()

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


def cpu_levels(on_windows):
    """Module name suffix -> compiler flags. The `''` level runs on every CPU of the build architecture."""
    if os.environ.get('IMOPS_NATIVE') == '1':
        return {'': [] if on_windows else ['-march=native']}

    machine = platform.machine().lower()
    if machine in ('x86_64', 'amd64'):
        if on_windows:
            return {'': [], '_v3': ['/arch:AVX2']}
        return {'': ['-march=x86-64-v2'], '_v3': ['-march=x86-64-v3']}
    if platform.system() == 'Darwin' and machine == 'arm64':
        return {'': ['-mcpu=apple-m1']}
    return {'': []}


def get_ext_modules():
    name = 'imops'
    on_windows = platform.system() == 'Windows'
    args = ['/openmp' if on_windows else '-fopenmp', '-O3']
    cpp_args = [
        '/std:c++20' if on_windows else '-std=c++17',
        '/O3' if on_windows else '-O3',
    ]  # FIXME: account for higher gcc versions
    levels = cpu_levels(on_windows)
    baseline_args = args + levels['']

    modules = [
        'argmax',
        'convex_hull',
        'backprojection',
        'measure',
        'morphology',
        'numeric',
        'radon',
        'rotate',
        'zoom',
    ]
    modules_to_link_against_numpy_core_math_lib = ['numeric']

    src_dir = Path(__file__).parent / 'src'
    extensions = [
        Extension(
            f'{name}.cpp.cpp_modules',
            [f'{name}/cpp/main.cpp'],
            include_dirs=[LazyImport('pybind11')],
            extra_compile_args=baseline_args + cpp_args,
            extra_link_args=baseline_args + cpp_args,
            language='c++',
        ),
        Extension(
            f'{name}.src._utils',
            [f'{name}/src/_utils.pyx'],
            language='c++',
            include_dirs=[LazyImport('numpy')],
            extra_compile_args=baseline_args + cpp_args,
            extra_link_args=baseline_args + cpp_args,
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        ),
    ]
    for module in modules:
        libraries = []
        library_dirs = []
        include_dirs = [LazyImport('numpy')]

        if module in modules_to_link_against_numpy_core_math_lib:
            library_dirs.append(NumpyLibImport())
            libraries.append('npymath')
            if not on_windows:
                libraries.append('m')

        for suffix, level_args in levels.items():
            if suffix:
                # Cython extension and .pyx source file names must be the same to compile
                # https://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu
                shutil.copyfile(src_dir / f'_{module}.pyx', src_dir / f'_{module}{suffix}.pyx')
            extensions.append(
                Extension(
                    f'{name}.src._{module}{suffix}',
                    [f'{name}/src/_{module}{suffix}.pyx'],
                    language='c',
                    include_dirs=include_dirs,
                    library_dirs=library_dirs,
                    libraries=libraries,
                    extra_compile_args=args + level_args,
                    extra_link_args=args + level_args,
                    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                )
            )

    return extensions
