import numpy as np
from skimage.measure import label as sk_label

from imops.measure import label


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
