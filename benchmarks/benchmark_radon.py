# TODO: Remove this crutch as soon as _configs.py appears in the master
try:
    from imops._configs import radon_configs
except ModuleNotFoundError:
    from imops.backend import Cython

    radon_configs = [Cython(fast) for fast in [False, True]]

import numpy as np

from imops.radon import inverse_radon, radon


# TODO: Remove this crutch as soon as updated radon.py appears in the master
try:
    from imops.testing import sample_ct, sk_iradon, sk_radon
except ImportError:
    import warnings

    from skimage.transform import iradon as iradon_, radon as radon_

    def sk_iradon(xs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='numpy')
            warnings.simplefilter('ignore', DeprecationWarning)
            warnings.simplefilter('ignore', np.VisibleDeprecationWarning)
            return np.stack([iradon_(x) for x in xs])

    def sk_radon(xs, strict=True):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='numpy')
            warnings.simplefilter('ignore', DeprecationWarning)
            warnings.simplefilter('ignore', np.VisibleDeprecationWarning)
            if strict:
                warnings.filterwarnings('error', '.*image must be zero.*', module='skimage')
            return np.stack([radon_(x) for x in xs])

    def sample_ct(n_slices, size, fill=0):
        shape = (n_slices, size, size)

        water = np.random.randn(*shape) * 100
        air = np.random.randn(*shape) * 100 - 1000
        choice = np.random.binomial(1, 0.5, shape).astype(bool)

        image = choice * water + ~choice * air
        return fill_outside(image, fill)

    def fill_outside(x, fill):
        x = x.copy()
        size = x.shape[-1]
        radius = size // 2
        xpr, ypr = np.mgrid[:size, :size] - radius
        x[:, (xpr**2 + ypr**2) > radius**2] = fill
        return x


from .common import discard_arg


class RadonSuite:
    # TODO: Add `Scipy` backend to radon-s
    scipy_backend = 'Scipy()'
    params = [radon_configs + [scipy_backend], ('float32', 'float64')]
    param_names = ['backend', 'dtype']
    timeout = 300

    @discard_arg(1)
    def setup(self, dtype):
        self.image = sample_ct(256, 256).astype(dtype)

    @discard_arg(2)
    def time_radon(self, backend):
        if backend == self.scipy_backend:
            sk_radon(self.image)
        else:
            radon(self.image, axes=(1, 2), backend=backend)

    @discard_arg(2)
    def time_inverse_radon(self, backend):
        if backend == self.scipy_backend:
            sk_iradon(self.image)
        else:
            inverse_radon(self.image, axes=(1, 2), theta=256, backend=backend)

    @discard_arg(2)
    def peakmem_radon(self, backend):
        if backend == self.scipy_backend:
            sk_radon(self.image)
        else:
            radon(self.image, axes=(1, 2), backend=backend)

    @discard_arg(2)
    def peakmem_inverse_radon(self, backend):
        if backend == self.scipy_backend:
            sk_iradon(self.image)
        else:
            inverse_radon(self.image, axes=(1, 2), theta=256, backend=backend)
