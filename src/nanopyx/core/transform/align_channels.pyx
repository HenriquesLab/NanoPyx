import numpy as np
cimport numpy as np

from cython.parallel import prange

cdef extern from "_c_interpolation_bicubic.h":
    float _c_interpolate(float* image, float r, float c, int rows, int cols) nogil

def align_channels(img, translation_masks):
    return _align_channels(img, translation_masks)

cdef float[:, :, :] _align_channels(float[:, :, :] img, float[:, :] translation_mask):

    cdef int n_rows = img.shape[1]
    cdef int n_cols = img.shape[2]

    cdef float[:, :, :] out = np.empty_like(img)

    cdef int row, col
    cdef float d_col, d_row = 0
    with nogil:
        for row in prange(n_rows):
            for col in range(n_cols):
                d_col = translation_mask[row, col]
                d_row = translation_mask[row, col + n_cols]
                out[0, row, col] = _c_interpolate(&img[0, 0, 0], row-d_row, col-d_col, n_rows, n_cols)

    return np.asarray(out, dtype=np.float32)