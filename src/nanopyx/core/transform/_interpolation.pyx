# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

import numpy as np
cimport numpy as np

import cython
from math import floor

from ._le_interpolation_catmull_rom import ShiftAndMagnify as ShiftMagnify_CR

from cython.parallel import prange


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

    z_coords = np.linspace(0, image.shape[0], image.shape[0])
    new_z_coords = np.linspace(0, image.shape[0], image.shape[0] * magnification)

    for r in range(image.shape[1]):
        for c in range(image.shape[2]):
            slc = image[:, r, c]
            image_interpolated[:, r, c] = np.interp(new_z_coords, z_coords, slc)

    
    return image_interpolated

cdef _linear_interpolation_1D_z(float[:, :, :] image, int magnification_z):

    cdef int slices = image.shape[0]
    cdef int slicesM = int(image.shape[0] * magnification_z)
    cdef int rows = image.shape[1]
    cdef int cols = image.shape[2]

    cdef int sM, r, c
    cdef int slice0, slice1, slc
    cdef float weight0, weight1

    cdef float[:, :, :] image_out = np.zeros((slicesM, rows, cols), dtype=np.float32)


    for sM in range(slicesM):
        slc = sM / magnification_z
        slice0 = int(floor(slc))
        slice1 = slice0 + 1
        weight1 = slc - slice0
        weight0 = 1.0 - weight1
        with nogil:
            for r in prange(rows):
                for c in prange(cols):
                    

                    if slice0 >= 0 and slice1 < slices:
                        image_out[sM, r, c] = weight0 * image[slice0, r, c] + weight1 * image[slice1, r, c];
                    elif slice0 >= 0:
                        image_out[sM, r, c] = image[slice0, r, c];
                    elif slice1 < slices:
                        image_out[sM, r, c] = image[slice1, r, c];
                    else:
                        image_out[sM, r, c] = 0.0;

    return image_out


def interpolate_3d_zlinear(image, magnification_xy: int = 5, magnification_z: int = 5):
    interpolator_xy = ShiftMagnify_CR(verbose=False)

    if magnification_xy > 1:
        xy_interpolated = np.asarray(interpolator_xy.run(np.ascontiguousarray(image), 0, 0, magnification_xy, magnification_xy))
    else:
        xy_interpolated = np.asarray(image)
    if magnification_z > 1:
        z_interpolated = _linear_interpolation_1D_z(xy_interpolated, magnification_z)
    else:
        z_interpolated = np.asarray(xy_interpolated)

    return z_interpolated
