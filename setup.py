import runpy
from pathlib import Path

from setuptools import Extension, find_packages, setup


class NumpyImport:
    def __repr__(self):
        import numpy as np

        return np.get_include()

    __fspath__ = __repr__


name = 'imops'
root = Path(__file__).parent
with open(root / 'requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()
with open(root / 'README.md', encoding='utf-8') as file:
    long_description = file.read()
version = runpy.run_path(root / name / '__version__.py')['__version__']

args = ['-fopenmp']
ext_modules = [
    Extension(
        f'{name}.src._{module}',
        [f'{name}/src/_{module}.pyx'],
        include_dirs=[NumpyImport()],
        extra_compile_args=args,
        extra_link_args=args,
    )
    for module in ['backprojection', 'fast_radon', 'fast_zoom']
]

setup(
    name=name,
    packages=find_packages(include=(name,)),
    include_package_data=True,
    version=version,
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'Cython',
    ],
    ext_modules=ext_modules,
    define_macros=[('NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION')],
    python_requires='>=3.6',
)
