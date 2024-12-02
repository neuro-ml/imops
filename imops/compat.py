try:
    from numpy.lib.array_utils import normalize_axis_tuple
except ModuleNotFoundError:
    from numpy.core.numeric import normalize_axis_tuple

try:
    from numpy.exceptions import VisibleDeprecationWarning
except ModuleNotFoundError:
    from numpy import VisibleDeprecationWarning

try:
    from scipy.ndimage._morphology import _ni_support
except ImportError:
    from scipy.ndimage.morphology import _ni_support

try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError

from scipy.ndimage._nd_image import euclidean_feature_transform  # noqa


normalize_sequence = _ni_support._normalize_sequence  # noqa
