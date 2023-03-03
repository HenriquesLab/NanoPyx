# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

import numpy as np
cimport numpy as np

from cython.parallel import prange

from .interpolation_catmull_rom cimport Interpolator


def translate_array(float[:, :, :] img_arr, float[:, :] drift_t):
    """
    Translate an array of images using the drift data.
    """

    cdef int n_slices = img_arr.shape[0]
    cdef float drift_x, drift_y

    for i in range(n_slices):
        drift_x = drift_t[i][1]
        drift_y = drift_t[i][2]
        img_arr[i,  :, :] = Interpolator(img_arr[i])._shift(drift_y, drift_x)

    return img_arr
