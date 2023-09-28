# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

import numpy as np
cimport numpy as np

from cython.parallel import prange


def get_circular_mask(w: int, r2: float):
    """
    Generate a circular mask as a NumPy array.

    This Cython function generates a circular mask as a 2D NumPy array, where pixels inside the circular region are set to 1.0 and pixels outside are set to 0.0.

    Parameters:
        - w (int): The width and height of the square output mask.
        - r2 (float): The radius of the circular mask relative to the width 'w'. Should be a value between 0 and 1.

    Returns:
        - mask (numpy.ndarray): A 2D NumPy array representing the circular mask. The shape of the array is (w, w), and it contains float values (0.0 or 1.0).

    Example:
        # Generate a circular mask with a radius of 0.3 relative to a 100x100 square.
        circular_mask = get_circular_mask(100, 0.3)

    Note:
        - The 'w' parameter determines the dimensions of the square mask.
        - The 'r2' parameter specifies the radius of the circular region relative to the width 'w', where 0.0 means no circle (all 0s), and 1.0 means a full circle (all 1s).
        - The generated circular mask is a NumPy array with float values.
        - This function uses Cython for performance and is designed for speed.
    """
    return np.array(_get_circular_mask(w, r2))

cdef float[:, :] _get_circular_mask(int w, float r2) nogil:

    cdef int y_i, x_i 
    cdef double radius = r2 * w * w / 4
    cdef double dist
    cdef float[:, :] mask

    with gil:
        mask = np.zeros((w, w), dtype=np.float32)

    for y_i in prange(w):
        for x_i in range(w):
            dist = (y_i - w/2)**2 + (x_i - w/2)**2
            if dist < radius:
                mask[y_i, x_i] = 1

    return mask