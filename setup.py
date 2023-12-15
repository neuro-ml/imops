import runpy
from pathlib import Path
import os
from setuptools import find_packages, setup

shared_objects = {'cpp_modules': False}

# a dumb suboptimal way to check if cmake builds are okay
for item in os.listdir('imops/cpp/build/'):
    for name in shared_objects:
        if name in item and '.so' in item:
            shared_objects[name] = True


not_builded = []
for name, builded in shared_objects.items():
    if not builded:
        not_builded.append(name)
if not_builded:    
    raise Exception(f"Failed to build following C++ shared objects: {not_builded}")


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

with open(root / 'requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()
with open(root / 'README.md', encoding='utf-8') as file:
    long_description = file.read()
version = runpy.run_path(root / name / '__version__.py')['__version__']

scope = {'__file__': __file__}
exec((root / '_build_utils.py').read_text(), scope)
ext_modules = scope['get_ext_modules']()

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
    py_modules=['imops/cpp/build/cpp_modules'],
    python_requires='>=3.6',
)
