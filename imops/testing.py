import warnings

import numpy as np
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


def seeded_by(seed):
    def wrapper(func):
        def inner(*args, **kwargs):
            old_state = np.random.get_state()
            np.random.seed(seed)

            try:
                return func(*args, **kwargs)
            finally:
                np.random.set_state(old_state)

        return inner

    return wrapper
