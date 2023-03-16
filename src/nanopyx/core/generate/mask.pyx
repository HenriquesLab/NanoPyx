# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

import numpy as np
cimport numpy as np

from cython.parallel import prange


def get_circular_mask(w: int, r2: float):

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