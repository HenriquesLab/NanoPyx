# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

import numpy as np
cimport numpy as np

import cython

from ._le_interpolation_catmull_rom import ShiftAndMagnify as ShiftMagnify_CR


cdef extern from "_c_interpolation_catmull_rom.h":
    float _c_cr_interpolate "_c_interpolate" (float *image, float row, float col, int rows, int cols) 

cdef float _cr_interpolate(float[:, :] img_stack, float row, float col):
    cdef int rows = img_stack.shape[0]
    cdef int cols = img_stack.shape[1]
    return _c_cr_interpolate(&img_stack[0,0], row, col, rows, cols)

def cr_interpolate(img_stack, row, col):
    img_stack = np.array(img_stack, dtype=np.float32)
    return _cr_interpolate(img_stack, row, col)

def interpolate_3d(image, magnification_xy: int = 5, magnification_z: int = 5):
    interpolator = ShiftMagnify_CR()

    xy_interpolated = interpolator.run(image, 0, 0, magnification_xy, magnification_xy)

    xyz_interpolated = interpolator.run(np.transpose(xy_interpolated, axes=[1, 0, 2]).copy(), 0, 0, magnification_z, 1)

    return np.transpose(xyz_interpolated, axes=[1, 0, 2]).copy()