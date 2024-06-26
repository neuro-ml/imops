from pathlib import Path

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

from imops.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening, distance_transform_edt

from .common import IMAGE_TYPE_BENCHMARK, IMAGE_TYPES_BENCHMARK, NUMS_THREADS_TO_BENCHMARK, discard_arg, load_npy_gz


class MorphologySuite:
    params = [morphology_configs, ('bool', 'int64'), NUMS_THREADS_TO_BENCHMARK, IMAGE_TYPES_BENCHMARK, (False, True)]
    param_names = ['backend', 'dtype', 'num_threads', 'image_type', 'boxed']

    @discard_arg(1)
    @discard_arg(2)
    @discard_arg(-1)
    @discard_arg(-1)
    def setup(self, dtype):
        real_images_path = Path(__file__).parent / 'data'

        lungs_image = load_npy_gz(real_images_path / 'lungs.npy.gz').astype(dtype, copy=False)
        bronchi_image = load_npy_gz(real_images_path / 'bronchi.npy.gz').astype(dtype, copy=False)
        rand_image = np.random.randint(0, 5 if dtype == 'int64' else 2, (512, 512, 512)).astype(dtype, copy=False)

        self.images = {
            IMAGE_TYPE_BENCHMARK.RAND: rand_image,
            IMAGE_TYPE_BENCHMARK.LUNGS: lungs_image,
            IMAGE_TYPE_BENCHMARK.BRONCHI: bronchi_image,
        }

    # FIXME: generalize this code somehow
    @discard_arg(2)
    def time_erosion(self, backend, num_threads, image_type, boxed):
        im = self.images[image_type]
        try:
            binary_erosion(im, num_threads=num_threads, backend=backend, boxed=boxed)
        except TypeError:
            binary_erosion(im, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def time_dilation(self, backend, num_threads, image_type, boxed):
        im = self.images[image_type]
        try:
            binary_dilation(im, num_threads=num_threads, backend=backend, boxed=boxed)
        except TypeError:
            binary_dilation(im, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def time_opening(self, backend, num_threads, image_type, boxed):
        im = self.images[image_type]
        try:
            binary_opening(im, num_threads=num_threads, backend=backend, boxed=boxed)
        except TypeError:
            binary_opening(im, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def time_closing(self, backend, num_threads, image_type, boxed):
        im = self.images[image_type]
        try:
            binary_closing(im, num_threads=num_threads, backend=backend, boxed=boxed)
        except TypeError:
            binary_closing(im, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def peakmem_erosion(self, backend, num_threads, image_type, boxed):
        im = self.images[image_type]
        try:
            binary_erosion(im, num_threads=num_threads, backend=backend, boxed=boxed)
        except TypeError:
            binary_erosion(im, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def peakmem_dilation(self, backend, num_threads, image_type, boxed):
        im = self.images[image_type]
        try:
            binary_dilation(im, num_threads=num_threads, backend=backend, boxed=boxed)
        except TypeError:
            binary_dilation(im, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def peakmem_opening(self, backend, num_threads, image_type, boxed):
        im = self.images[image_type]
        try:
            binary_opening(im, num_threads=num_threads, backend=backend, boxed=boxed)
        except TypeError:
            binary_opening(im, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def peakmem_closing(self, backend, num_threads, image_type, boxed):
        im = self.images[image_type]
        try:
            binary_closing(im, num_threads=num_threads, backend=backend, boxed=boxed)
        except TypeError:
            binary_closing(im, num_threads=num_threads, backend=backend)


class EDTSuite:
    params = [morphology_configs, ('bool', 'int32', 'int64'), NUMS_THREADS_TO_BENCHMARK, [False, True], [False, True]]
    param_names = ['backend', 'dtype', 'num_threads', 'return_distances', 'return_indices']

    @discard_arg(1)
    @discard_arg(-1)
    @discard_arg(-1)
    @discard_arg(-1)
    def setup(self, dtype):
        self.image = np.random.randint(0, 5 if 'int' in dtype else 2, (512, 512, 512)).astype(dtype, copy=False)

    @discard_arg(2)
    def time_edt(self, backend, num_threads, return_distances, return_indices):
        if not return_distances and not return_indices:
            return

        distance_transform_edt(
            self.image,
            return_distances=return_distances,
            return_indices=return_indices,
            backend=backend,
            num_threads=num_threads,
        )

    @discard_arg(2)
    def peakmem_edt(self, backend, num_threads, return_distances, return_indices):
        if not return_distances and not return_indices:
            return

        distance_transform_edt(
            self.image,
            return_distances=return_distances,
            return_indices=return_indices,
            backend=backend,
            num_threads=num_threads,
        )
