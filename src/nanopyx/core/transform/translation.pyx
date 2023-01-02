# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True

import numpy as np
cimport numpy as np

from .interpolation.catmull_rom cimport _shift


def translate_array(float[:, :, :] img_arr, float[:, :] drift_t):
    """
    Translate an array of images using the drift data.
    """

    cdef int n_slices = img_arr.shape[0]
    cdef float drift_x, drift_y
    cdef float[:,:] tmp

    for i in range(n_slices):
        drift_x = drift_t[i][1]
        drift_y = drift_t[i][2]
        tmp = _shift(img_arr[i], drift_x, drift_y)
        img_arr[i] = tmp

    return img_arr
