# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

from libc.math cimport fabs

import numpy as np
cimport numpy as np
from cython.parallel import prange


def calculate_ppmcc(np.ndarray im1, np.ndarray im2, int shift_x, int shift_y):
    """
    Calculates the Pearson's correlation between two images after applying a shift.
    :param im1: numpy array with shape (y, x)
    :param im2: numpy array with shape (y, x)
    :param shift_x: int; value to shift images in x dimension
    :param shift_y: int; value to shift images in y dimension
    :return: float; value of Pearson's Correlation function after shifting the two images.
    """
    return _calculate_ppmcc(im1.astype(np.float32), im2.astype(np.float32), shift_x, shift_y)


cdef float _calculate_ppmcc(float[:, :] im1, float[:, :] im2, int shift_x, int shift_y) nogil:
    cdef int w = im1.shape[1]
    cdef int h = im1.shape[0]
    cdef int new_w = int(w - fabs(shift_x))
    cdef int new_h = int(h - fabs(shift_y))

    cdef int x0 = max(0, -shift_x)
    cdef int y0 = max(0, -shift_y)
    cdef int x1 = x0 + shift_x
    cdef int y1 = y0 + shift_y

    return _pearson_correlation(im1[y0:y0+new_h, x0:x0+new_w], im2[y1:y1+new_h, x1:x1+new_w])


def pearson_correlation(np.ndarray im1, np.ndarray im2):
    """
    Calculates the Pearson's correlation between two images.
    :param im1: numpy array with shape (y, x)
    :param im2: numpy array with shape (y, x)
    :return: float; value of Pearson's correlation between two images
    """
    return _pearson_correlation(im1.astype(np.float32), im2.astype(np.float32))


cdef float _pearson_correlation(float[:,:] im1, float[:,:] im2) nogil:

    cdef int w = im1.shape[1]
    cdef int h = im1.shape[0]
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
            mean_im1 += im1[j, i]
            mean_im2 += im2[j, i]
    mean_im1 /= wh
    mean_im2 /= wh
    for j in prange(h):
        for i in range(w):
            d_im1 = im1[j, i] - mean_im1
            d_im2 = im2[j, i] - mean_im2
            sum_im12 += d_im1 * d_im2
            sum_im11 += d_im1 * d_im1
            sum_im22 += d_im2 * d_im2
    if sum_im11 == 0 or sum_im22 == 0:
        return 0
    else:
        return sum_im12 / (sum_im11 * sum_im22)**0.5
