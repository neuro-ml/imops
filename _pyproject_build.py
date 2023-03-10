import shutil
from pathlib import Path

from setuptools import Extension
from setuptools.command.build_py import build_py


class NumpyImport:
    def __repr__(self):
        import numpy as np

        return np.get_include()

    __fspath__ = __repr__


class PyprojectBuild(build_py):
    def run(self):
        self.run_command('build_ext')
        return super().run()

    def initialize_options(self):
        super().initialize_options()

        name = 'imops'
        args = ['-fopenmp']

        if self.distribution.ext_modules is None:
            self.distribution.ext_modules = []

        # Cython extension and .pyx source file names must be the same to compile
        # https://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu
        modules = ['backprojection', 'radon', 'zoom', 'morphology', 'numeric']
        for module in modules:
            src_dir = Path(__file__).parent / 'imops' / 'src'
            shutil.copyfile(src_dir / f'_{module}.pyx', src_dir / f'_fast_{module}.pyx')

        for module in modules:
            for prefix, additional_args in zip(['', 'fast_'], [[], ['-ffast-math']]):
                self.distribution.ext_modules.append(
                    Extension(
                        f'{name}.src._{prefix}{module}',
                        [f'{name}/src/_{prefix}{module}.pyx'],
                        include_dirs=[NumpyImport()],
                        extra_compile_args=args + additional_args,
                        extra_link_args=args + additional_args,
                        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
                    )
                )
