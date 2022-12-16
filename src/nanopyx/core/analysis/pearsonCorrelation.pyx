# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from cython.parallel import prange

def pearsonCorrelation(float[:,:] im1, float[:,:] im2):

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

    with nogil:
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
