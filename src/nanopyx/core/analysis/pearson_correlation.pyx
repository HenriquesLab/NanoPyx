# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True

import numpy as np
cimport numpy as np
from cython.parallel import prange


def pearson_correlation(np.ndarray im1, np.ndarray im2):
    return _pearson_correlation(im1.astype(np.float32), im2.astype(np.float32))


cdef float _pearson_correlation(float[:,:] im1, float[:,:] im2) nogil:

    cdef int w = im1.shape[0]
    cdef int h = im2.shape[0]
    cdef int wh = w*h

    cdef float mean_im1 = 0.0
    cdef float mean_im2 = 0.0
    cdef float sum_im12 = 0.0
    cdef float sum_im11 = 0.0
    cdef float sum_im22 = 0.0
    
    cdef int i, j
    cdef float d_im1, d_im2

    for j in prange(h):
        for i in range(w):
            mean_im1 += im1[i,j]
            mean_im2 += im2[i,j]
    mean_im1 /= wh
    mean_im2 /= wh
    for j in prange(h):
        for i in range(w):
            d_im1 = im1[i,j] - mean_im1
            d_im2 = im2[i,j] - mean_im2
            sum_im12 += d_im1 * d_im2
            sum_im11 += d_im1 * d_im1
            sum_im22 += d_im2 * d_im2
    return sum_im12 / (sum_im11 * sum_im22)**0.5
