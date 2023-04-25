import numpy as np
from skimage.measure import label as sk_label

from imops._configs import measure_configs
from imops.measure import center_of_mass, label

from .common import discard_arg


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
    params = [measure_configs, ('bool', 'float64')]
    param_names = ['backend', 'dtype']

    @discard_arg(1)
    def setup(self, dtype):
        self.image = np.random.randn(512, 512, 512).astype(dtype)

    @discard_arg(2)
    def time_center_of_mass(self, backend):
        center_of_mass(self.image, backend=backend)

    @discard_arg(2)
    def peakmem_center_of_mass(self, backend):
        center_of_mass(self.image, backend=backend)


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
