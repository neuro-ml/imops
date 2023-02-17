import numpy as np

from imops._configs import zoom_configs
from imops.zoom import zoom


class TimeSuiteZoom:
    params = zoom_configs

    def setup(self):
        self.image = np.random.randn(256, 256, 256)

    def time_zoom(self, backend):
        zoom(self.image, 2, backend=backend)
