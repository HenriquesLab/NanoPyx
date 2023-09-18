# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

import numpy as np
cimport numpy as np

import cython


cdef extern from "_c_interpolation_catmull_rom.h":
    float _c_cr_interpolate "_c_interpolate" (float *image, float row, float col, int rows, int cols) 

cdef float _cr_interpolate(float[:, :] img_stack, float row, float col):
    cdef int rows = img_stack.shape[1]
    cdef int cols = img_stack.shape[2]
    return _c_cr_interpolate(&img_stack[0,0], row, col, rows, cols)

def cr_interpolate(img_stack, row, col):
    img_stack = np.array(img_stack, dtype=np.float32)
    return _cr_interpolate(img_stack, row, col)

