import numpy as np
import pytest
from skimage import data
from skimage.morphology import convex_hull_image
from skimage.morphology.convex_hull import _offsets_diamond
from skimage.util import invert, unique_rows

from imops.morphology import convex_hull_image as convex_hull_image_fast
from imops.src._convex_hull import _left_right_bounds, _offset_unique


np.random.seed(1337)
N_STRESS = 1000


@pytest.fixture(params=[False, True])
def offset_coordinates(request):
    return request.param


def test_bounds():
    for _ in range(N_STRESS):
        image = np.zeros((100, 100), dtype=bool)
        image[20:70, 20:90] = np.random.randn(50, 70) > 1.5

        im_any = np.any(image, axis=1)
        x_indices = np.arange(0, image.shape[0])[im_any]
        y_indices_left = np.argmax(image[im_any], axis=1)
        y_indices_right = image.shape[1] - 1 - np.argmax(image[im_any][:, ::-1], axis=1)
        left = np.stack((x_indices, y_indices_left), axis=-1)
        right = np.stack((x_indices, y_indices_right), axis=-1)
        coords_ref = np.vstack((left, right))

        coords = _left_right_bounds(image)

        # _left_right_bounds has another order
        assert len(unique_rows(coords_ref)) == len(unique_rows(np.concatenate((coords, coords_ref), 0)))


def test_offset():
    for _ in range(N_STRESS):
        image = np.zeros((100, 100), dtype=bool)
        image[20:70, 20:90] = np.random.randn(50, 70) > 1.5

        coords = _left_right_bounds(image)

        offsets = _offsets_diamond(2)
        coords_ref = unique_rows((coords[:, None, :] + offsets).reshape(-1, 2))

        coords = _offset_unique(coords)

        # _left_right_bounds has another order
        assert len(coords_ref) == len(unique_rows(np.concatenate((coords, coords_ref), 0)))


def test_convex_hull_image(offset_coordinates):
    image = invert(data.horse())

    try:
        chull_ref = convex_hull_image(image, offset_coordinates=offset_coordinates, include_borders=True)
    except TypeError:
        chull_ref = convex_hull_image(image, offset_coordinates=offset_coordinates)

    chull = convex_hull_image_fast(image, offset_coordinates=offset_coordinates)

    assert (chull >= image).all()
    assert (chull >= chull_ref).all()

    assert ((chull > chull_ref).sum() / chull_ref.sum()) < 5e-3


def test_convex_hull_image_random(offset_coordinates):
    for _ in range(N_STRESS):
        image = np.zeros((200, 200), dtype=bool)

        image[15:-15, 5:-25] = np.random.randn(170, 170) > 3

        try:
            chull_ref = convex_hull_image(image, offset_coordinates=offset_coordinates, include_borders=True)
        except TypeError:
            chull_ref = convex_hull_image(image, offset_coordinates=offset_coordinates)

        chull = convex_hull_image_fast(image, offset_coordinates=offset_coordinates)

        assert (chull >= image).all()
        assert (chull >= chull_ref).all()

        assert ((chull > chull_ref).sum() / chull_ref.sum()) < 5e-3


def test_convex_hull_image_non2d(offset_coordinates):
    image = np.zeros((3, 3, 3), dtype=bool)

    with pytest.raises(ValueError):
        _ = convex_hull_image_fast(image, offset_coordinates=offset_coordinates)


def test_convex_hull_image_empty(offset_coordinates):
    image = np.zeros((10, 10), dtype=bool)
    chull = convex_hull_image_fast(image, offset_coordinates=offset_coordinates)

    assert (chull == np.zeros_like(chull)).all()


def test_convex_hull_image_qhullsrc_issues():
    image = np.zeros((10, 10), dtype=bool)
    image[1, 1] = True
    image[-2, -2] = True

    with pytest.warns(UserWarning):
        chull = convex_hull_image_fast(image, offset_coordinates=False)

    assert (chull == np.zeros_like(chull)).all()
