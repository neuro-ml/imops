import os

import numpy as np


# TODO: Remove this crutch as soon as _configs.py appears in the master
try:
    from imops._configs import morphology_configs
except ModuleNotFoundError:
    from imops.backend import Cython, Scipy

    morphology_configs = [
        Scipy(),
        *[Cython(fast) for fast in [False, True]],
    ]

from imops.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    boxed_binary_dilation,
    boxed_binary_erosion,
)

from .common import IMAGE_TYPE_BENCHMARK, IMAGE_TYPES_BENCHMARK, NUMS_THREADS_TO_BENCHMARK, discard_arg


class MorphologySuite:
    params = [morphology_configs, ('bool', 'int64'), NUMS_THREADS_TO_BENCHMARK, IMAGE_TYPES_BENCHMARK, (False, True)]
    param_names = ['backend', 'dtype', 'num_threads', 'image_type', 'box']

    @discard_arg(1)
    @discard_arg(3)
    @discard_arg(4)
    @discard_arg(5)
    def setup(self, dtype):
        rand_image = np.random.randint(0, 5 if dtype == 'int64' else 2, (512, 512, 512)).astype(dtype)
        if 'REAL_IMAGES_PATH' in os.environ:
            real_images_path = os.environ['REAL_IMAGES_PATH']
            lungs_image = np.load(os.path.join(real_images_path, 'lungs.npy'))
            bronch_image = np.load(os.path.join(real_images_path, 'bronch.npy'))
        else:
            lungs_image = np.random.randint(0, 5 if dtype == 'int64' else 2, (512, 512, 512)).astype(dtype)
            bronch_image = np.random.randint(0, 5 if dtype == 'int64' else 2, (512, 512, 512)).astype(dtype)
        self.images = {
            IMAGE_TYPE_BENCHMARK.RAND: rand_image,
            IMAGE_TYPE_BENCHMARK.LUNGS: lungs_image,
            IMAGE_TYPE_BENCHMARK.BRONCH: bronch_image,
        }

    # FIXME: generalize this code somehow
    @discard_arg(2)
    def time_erosion(self, backend, num_threads, image_type, box):
        im = self.images[image_type]
        func = boxed_binary_erosion if box else binary_erosion
        func(im, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def time_dilation(self, backend, num_threads, image_type, box):
        im = self.images[image_type]
        func = boxed_binary_dilation if box else binary_dilation
        func(im, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    @discard_arg(5)
    def time_opening(self, backend, num_threads, image_type):
        im = self.images[image_type]
        binary_opening(im, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    @discard_arg(5)
    def time_closing(self, backend, num_threads, image_type):
        im = self.images[image_type]
        binary_closing(im, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def peakmem_erosion(self, backend, num_threads, image_type, box):
        im = self.images[image_type]
        func = boxed_binary_erosion if box else binary_erosion
        func(im, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def peakmem_dilation(self, backend, num_threads, image_type, box):
        im = self.images[image_type]
        func = boxed_binary_dilation if box else binary_dilation
        func(im, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    @discard_arg(5)
    def peakmem_opening(self, backend, num_threads, image_type):
        im = self.images[image_type]
        binary_opening(im, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    @discard_arg(5)
    def peakmem_closing(self, backend, num_threads, image_type):
        im = self.images[image_type]
        binary_closing(im, num_threads=num_threads, backend=backend)
