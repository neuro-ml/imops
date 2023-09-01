import runpy
import shutil
from pathlib import Path

from setuptools import Extension, find_packages, setup


name = 'imops'
root = Path(__file__).parent
classifiers = [
    'Development Status :: 5 - Production/Stable',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
]


class NumpyImport(dict):
    """Hacky way to return Numpy's include path with lazy import."""

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


with open(root / 'requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()
with open(root / 'README.md', encoding='utf-8') as file:
    long_description = file.read()
version = runpy.run_path(root / name / '__version__.py')['__version__']

# Cython extension and .pyx source file names must be the same to compile
# https://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu
# FIXME: code for cythonizing is duplicated in `_pyproject_build.py`
modules = ['backprojection', 'measure', 'morphology', 'numeric', 'radon', 'zoom']
for module in modules:
    src_dir = Path(__file__).parent / name / 'src'
    shutil.copyfile(src_dir / f'_{module}.pyx', src_dir / f'_fast_{module}.pyx')

args = ['-fopenmp']
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
    description='Efficient parallelizable algorithms for multidimensional arrays to speed up your data pipelines',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='maxme1, vovaf709, talgat',
    author_email='maxs987@gmail.com, vovaf709@yandex.ru, saparov2130@gmail.com',
    license='MIT',
    url='https://github.com/neuro-ml/imops',
    download_url='https://github.com/neuro-ml/imops/archive/v%s.tar.gz' % version,
    keywords=[
        'image processing',
        'fast',
        'ndarray',
        'data pipelines',
    ],
    classifiers=classifiers,
    install_requires=requirements,
    extras_require={'numba': ['numba'], 'all': ['numba']},
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'Cython<3.0.0',
    ],
    ext_modules=ext_modules,
    python_requires='>=3.6',
)
