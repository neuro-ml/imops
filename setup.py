import runpy
import shutil
from pathlib import Path

from setuptools import Extension, find_packages, setup


name = 'imops'
root = Path(__file__).parent


class NumpyImport:
    def __repr__(self):
        import numpy as np

        return np.get_include()

    __fspath__ = __repr__


with open(root / 'requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()
with open(root / 'README.md', encoding='utf-8') as file:
    long_description = file.read()
version = runpy.run_path(root / name / '__version__.py')['__version__']

# Cython extension and .pyx source file names must be the same to compile
# https://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu
modules = ['backprojection', 'radon', 'zoom']
for module in modules:
    src_dir = Path(__file__).parent / 'imops' / 'src'
    shutil.copyfile(src_dir / f'_{module}.pyx', src_dir / f'_fast_{module}.pyx')

args = ['-Xpreprocessor', '-fopenmp']
ext_modules = [
    Extension(
        f'{name}.src._{prefix}{module}',
        [f'{name}/src/_{prefix}{module}.pyx'],
        include_dirs=[NumpyImport()],
        extra_compile_args=args + additional_args,
        extra_link_args=args + additional_args,
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )
    for module in modules
    for prefix, additional_args in zip(['', 'fast_'], [[], ['-ffast-math']])
]

setup(
    name=name,
    packages=find_packages(include=(name,)),
    include_package_data=True,
    version=version,
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    extras_require={'numba': ['numba'], 'all': ['numba']},
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'Cython',
    ],
    ext_modules=ext_modules,
    python_requires='>=3.6',
)
