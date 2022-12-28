import cython
import numpy as np

from .interpolation.catmull_rom import shift

@cython.boundscheck(False)
@cython.wraparound(False)

def translate_array(cython.float[:, :, :] img_arr, cython.float[:, :] drift_t):

    cdef cython.int n_slices = img_arr.shape[0]
    cdef cython.int y = img_arr.shape[1]
    cdef cython.int x = img_arr.shape[2]
    cdef cython.float drift_x, drift_y
    cdef cython.float[:,:] tmp

    for i in range(n_slices):
        drift_x = drift_t[i][1]
        drift_y = drift_t[i][2]
        tmp = shift(img_arr[i], drift_x, drift_y)
        img_arr[i] = tmp

    return img_arr
