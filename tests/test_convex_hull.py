import numpy as np
from skimage import data
from skimage.morphology import convex_hull_image
from skimage.morphology.convex_hull import _offsets_diamond
from skimage.util import invert, unique_rows

from imops.morphology import convex_hull_image as convex_hull_image_fast
from imops.src._convex_hull import _left_right_bounds, _offset_unique


def test_bounds():
    image = np.zeros((100, 100), dtype=bool)
    image[20:70, 20:90] = np.random.randn(50, 70) > 0.5

    im_any = np.any(image, axis=1)
    x_indices = np.arange(0, image.shape[0])[im_any]
    y_indices_left = np.argmax(image[im_any], axis=1)
    y_indices_right = image.shape[1] - 1 - np.argmax(image[im_any][:, ::-1], axis=1)
    left = np.stack((x_indices, y_indices_left), axis=-1)
    right = np.stack((x_indices, y_indices_right), axis=-1)
    coords_ref = np.vstack((left, right))

    coords = _left_right_bounds(image)

    # _left_right_bounds has another order
    assert len(coords_ref) == len(unique_rows(np.concatenate((coords, coords_ref), 0)))


def test_offset():
    image = np.zeros((100, 100), dtype=bool)
    image[20:70, 20:90] = np.random.randn(50, 70) > 0.5

    coords = _left_right_bounds(image)

    offsets = _offsets_diamond(2)
    coords_ref = unique_rows((coords[:, None, :] + offsets).reshape(-1, 2))

    coords = _offset_unique(coords)

    # _left_right_bounds has another order
    assert len(coords_ref) == len(unique_rows(np.concatenate((coords, coords_ref), 0)))


def test_convex_hull_image():
    image = invert(data.horse())

    chull_ref = convex_hull_image(image, offset_coordinates=True, include_borders=True)

    chull = convex_hull_image_fast(image, offset_coordinates=True)

    assert not (chull < image).any()
    assert not (chull < chull_ref).any()
