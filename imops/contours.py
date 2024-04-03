import numpy as np
from typing import List, Tuple
from .cpp.cpp_modules import get_contour_segments_stacked_cpp


def get_contour_segments_stacked(array_input: np.ndarray, level: float) -> List[List[Tuple]]:
    return get_contour_segments_stacked_cpp(array_input, level)
