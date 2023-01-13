from typing import Tuple, Union
from warnings import warn

import numpy as np
from cc3d import connected_components
from fastremap import remap
from skimage.measure import label as skimage_label


# (ndim, skimage_connectivity) -> cc3d_connectivity
skimage2cc3d = {
    (1, 1): 4,
    (2, 1): 4,
    (2, 2): 8,
    (3, 1): 6,
    (3, 2): 18,
    (3, 3): 26,
}


def label(
    label_image: np.ndarray, background: int = None, return_num: bool = False, connectivity: int = None
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    ndim = label_image.ndim
    connectivity = connectivity or ndim

    if not 1 <= connectivity <= ndim:
        raise ValueError(f'Connectivity for {ndim}D image should be in [1, ..., {ndim}]. Got {connectivity}.')

    if ndim > 3:
        warn("Fast label is only supported for ndim<=3, Falling back to scikit-image's implementation.")
        return skimage_label(label_image, background=background, return_num=return_num, connectivity=connectivity)

    if ndim == 1:
        label_image = label_image[None]

    if background:
        label_image = remap(label_image, {background: 0, 0: background}, preserve_missing_labels=True, in_place=False)

    labeled, num_components = connected_components(
        label_image,
        connectivity=skimage2cc3d[(ndim, connectivity)],
        return_N=True,
    )

    if ndim == 1:
        labeled = labeled[0]

    if return_num:
        return labeled, num_components

    return labeled
