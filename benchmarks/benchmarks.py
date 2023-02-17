import numpy as np

from imops.zoom import zoom


class TimeSuiteZoom:
    def setup(self):
        self.image = np.random.randn(256, 256, 256)

    def time_zoom(self):
        zoom(self.image, 2)
