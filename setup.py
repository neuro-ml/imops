import runpy
from pathlib import Path

from setuptools import find_packages, setup


name = 'imops'
root = Path(__file__).parent
classifiers = [
    'Development Status :: 5 - Production/Stable',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13',
]

with open(root / 'requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()
with open(root / 'README.md', encoding='utf-8') as file:
    long_description = file.read()
version = runpy.run_path(root / name / '__version__.py')['__version__']

build_utils = root / name / '_build_utils.py'
scope = {'__file__': str(build_utils)}
exec(build_utils.read_text(), scope)
ext_modules = scope['get_ext_modules']()

setup(
    name=name,
    packages=find_packages(include=(name,), exclude=('tests', 'tests.*')),
    include_package_data=True,
    version=version,
    description='Efficient parallelizable algorithms for multidimensional arrays to speed up your data pipelines',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='maxme1, vovaf709, talgat, alexeybelkov',
    author_email='max@aumi.ai, vovaf709@yandex.ru, saparov2130@gmail.com, fpmbelkov@gmail.com',
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
        'setuptools<69.0.0',
        'numpy<3.0.0',
        'Cython>=3.0.0,<4.0.0',
        'pybind11',
    ],
    ext_modules=ext_modules,
    python_requires='>=3.7',
)
