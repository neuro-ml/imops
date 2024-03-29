import numpy as np
from skimage.measure import label as sk_label


try:
    from imops._configs import measure_configs
except ImportError:
    from imops._configs import numeric_configs as measure_configs

from imops.measure import center_of_mass, label

from .common import NUMS_THREADS_TO_BENCHMARK, discard_arg


class LabelSuite:
    def setup(self):
        self.image = np.random.randint(0, 5, size=(256, 256, 256))

    def time_label(self):
        label(self.image)

    def time_sk_label(self):
        sk_label(self.image)

    def peakmem_label(self):
        label(self.image)

    def peakmem_sk_label(self):
        sk_label(self.image)


class CenterOfMassSuite:
    params = [measure_configs, ('bool', 'float64'), NUMS_THREADS_TO_BENCHMARK]
    param_names = ['backend', 'dtype', 'num_threads']

    @discard_arg(1)
    @discard_arg(-1)
    def setup(self, dtype):
        self.image = np.random.randn(512, 512, 512).astype(dtype)

    @discard_arg(2)
    def time_center_of_mass(self, backend, num_threads):
        center_of_mass(self.image, num_threads=num_threads, backend=backend)

    @discard_arg(2)
    def peakmem_center_of_mass(self, backend, num_threads):
        center_of_mass(self.image, num_threads=num_threads, backend=backend)


class LabeledCenterOfMassSuite:
    params = [measure_configs, ('bool', 'float64'), [1, 4, 16]]
    param_names = ['backend', 'dtype', 'n_labels']

    @discard_arg(1)
    def setup(self, dtype, n_labels):
        self.image = np.random.randn(512, 512, 512).astype(dtype)
        self.labels = np.random.randint(0, n_labels, size=self.image.shape)
        self.index = np.arange(n_labels)

    @discard_arg(-1)
    @discard_arg(-1)
    def time_labeled_center_of_mass(self, backend):
        center_of_mass(self.image, labels=self.labels, index=self.index, backend=backend)

    @discard_arg(-1)
    @discard_arg(-1)
    def peakmem_labeled_center_of_mass(self, backend):
        center_of_mass(self.image, labels=self.labels, index=self.index, backend=backend)
