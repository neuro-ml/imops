import struct
import sys
import threading
from multiprocessing import get_context


def _bits(x):
    return struct.unpack('<Q', struct.pack('<d', x))[0]


def _fp_state():
    """Raw results that change with flush-to-zero, denormals-are-zero or the rounding mode."""
    one = float(len(sys.argv) >= 0)  # known only at runtime, so CPython can not fold the probes
    tiny = float.fromhex('0x1p-1022')
    subnormal = float.fromhex('0x0.0000000000001p-1022')
    return (
        _bits(tiny * one * 0.5),
        subnormal > 0.0,
        _bits(one + 2.0**-53),
        _bits(one + 3 * 2.0**-53),
        _bits(-one - 2.0**-53),
    )


def _fp_states_around_imops():
    import numpy as np

    before = _fp_state()

    from imops import rotate, zoom
    from imops.interp2d import Linear2DInterpolator

    image = np.random.rand(16, 16, 16)
    zoom(image, 2, num_threads=2)
    rotate(image, 30, num_threads=2)
    points = np.unique(np.random.randint(0, 64, (200, 2)), axis=0)
    Linear2DInterpolator(points[::2], np.random.rand(len(points[::2])))(points[1::2])

    in_new_thread = []
    thread = threading.Thread(target=lambda: in_new_thread.append(_fp_state()))
    thread.start()
    thread.join()

    return before, _fp_state(), in_new_thread[0]


def test_imops_keeps_fp_state():
    # A fresh interpreter: imops loads its extensions once, and other tests have loaded them already.
    with get_context('spawn').Pool(1) as pool:
        before, after, in_new_thread = pool.apply(_fp_states_around_imops)

    assert after == before
    assert in_new_thread == before
