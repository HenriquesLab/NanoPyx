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

    yz_interpolated = np.transpose(interpolator.run(np.transpose(xy_interpolated, axes=[1, 0, 2]).copy(), 0, 0, magnification_z, 1), axes=[1, 0, 2]).copy()
    xz_interpolated = np.transpose(interpolator.run(np.transpose(xy_interpolated, axes=[2, 1, 0]).copy(), 0, 0, 1, magnification_z), axes=[2,1,0]).copy()

    return np.mean(np.array([yz_interpolated, xz_interpolated]), axis=0)

def linear_interpolation_1D_z(image, magnification):
    # linear interpolator that takes a 3D image and do linear interpolation in a 1D column in the chosen axis

    image_interpolated = np.zeros((image.shape[0] * magnification, image.shape[1], image.shape[2]), dtype=np.float32)

    # for s in range(image.shape[0]):
    #     for r in range(image.shape[1]):
    #         for c in range(image.shape[2]):
    #             image_interpolated[s*magnification:(s+1)*magnification, r, c] = np.linspace(image[s, r, c], image[s+1, r, c], magnification)

    # run over all the planes
    for s in range(image.shape[0]-1):
        image_interpolated[s*magnification:(s+1)*magnification, :,:] = np.linspace(image[s, :,:], image[s+1, :,:], magnification)

    return image_interpolated


def interpolate_3d_zlinear(image, magnification_xy: int = 5, magnification_z: int = 5):
    interpolator_xy = ShiftMagnify_CR()

    xy_interpolated = interpolator_xy.run(image, 0, 0, magnification_xy, magnification_xy)
    z_interpolated = linear_interpolation_1D_z(xy_interpolated, magnification_z)

    return z_interpolated
