# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=False

import numpy as np
cimport numpy as np

import cython
from math import floor
import numpy.fft as fft
from libc.math cimport ceil, log2, pow

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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:, :] _mirror_padding_even_square_c(float[:, :] img):
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]

    # Get power-of-2 dimension >= each side, then pick square
    cdef int bitW = <int>pow(2, ceil(log2(w)))
    cdef int bitH = <int>pow(2, ceil(log2(h)))
    cdef int outputDim = max(bitH, bitW)

    cdef int scaleW = <int>ceil(outputDim / float(w))
    if scaleW % 2 == 0:
        scaleW += 1
    cdef int scaleH = <int>ceil(outputDim / float(h))
    if scaleH % 2 == 0:
        scaleH += 1

    cdef int i, j, sH, sW, p, q
    cdef float[:, :] padded = np.zeros((scaleH * h, scaleW * w), dtype=np.float32)

    # use integer arithmetic for the tile-flip decision
    cdef int midW = (scaleW - 1) // 2
    cdef int midH = (scaleH - 1) // 2

    for j in range(h):
        for i in range(w):
            for sH in range(scaleH):
                for sW in range(scaleW):
                    if ((sW - midW) % 2) == 0:
                        p = i
                    else:
                        p = w - 1 - i
                    if ((sH - midH) % 2) == 0:
                        q = j
                    else:
                        q = h - 1 - j
                    padded[j + sH * h, i + sW * w] = img[q, p]

    # Crop to square outputDim
    cdef int xROI = (scaleW * w - outputDim) // 2
    cdef int yROI = (scaleH * h - outputDim) // 2
    return padded[yROI:yROI + outputDim, xROI:xROI + outputDim]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float[:, :] _fht_space_interpolation_c(float[:, :] img, int intFactor, bint doMirrorPadding):
    cdef float[:, :] working_img
    cdef float orig_min = np.min(img)
    cdef float orig_max = np.max(img)
    cdef float scale_factor

    if doMirrorPadding:
        working_img = _mirror_padding_even_square_c(img)
    else:
        working_img = img

    # Forward FFT (use complex128 for accuracy)
    cdef complex[:, :] F = fft.fft2(np.asarray(working_img))
    cdef complex[:, :] Fshift = fft.fftshift(F)

    cdef int h = Fshift.shape[0]
    cdef int w = Fshift.shape[1]
    cdef int hInt = h * intFactor
    cdef int wInt = w * intFactor

    # Zero-padded enlarged spectrum and place centered block
    cdef complex[:, :] FInt = np.zeros((hInt, wInt), dtype=np.complex128)

    # center positions (integer)
    cdef int xROI = (wInt - w) // 2
    cdef int yROI = (hInt - h) // 2

    # place the whole centered block at once (faster and correct)
    FInt[yROI:yROI + h, xROI:xROI + w] = Fshift

    # Inverse transform
    cdef complex[:, :] result = fft.ifft2(fft.ifftshift(FInt))
    cdef float[:, :] output = np.real(result).astype(np.float32)

    # Scale to match input range (robust to constant images)
    cdef float out_min = np.min(output)
    cdef float out_max = np.max(output)
    if out_max != out_min:
        scale_factor = (orig_max - orig_min) / (out_max - out_min)
        output = (np.asarray(output) - out_min) * scale_factor + orig_min

    if doMirrorPadding:
        xROI = (wInt - intFactor * img.shape[1]) // 2
        yROI = (hInt - intFactor * img.shape[0]) // 2
        return output[yROI:yROI + intFactor * img.shape[0],
                      xROI:xROI + intFactor * img.shape[1]]
    return output

def fht_space_interpolation(image, int magnification=2, bint doMirrorPadding=True):
    """
    Fourier interpolation of 2D images or 3D stacks
    
    Parameters:
        image: 2D or 3D numpy array
        magnification: Integer interpolation factor
        doMirrorPadding: If True, applies mirror padding before FFT
    """
    cdef float[:, :] img2d
    if image.ndim == 2:
        img2d = np.ascontiguousarray(image, dtype=np.float32)
        return np.ascontiguousarray(np.asarray(_fht_space_interpolation_c(img2d, magnification, doMirrorPadding)))
    elif image.ndim == 3:
        return np.ascontiguousarray(np.stack([np.asarray(_fht_space_interpolation_c(
            np.ascontiguousarray(frame, dtype=np.float32), 
            magnification, doMirrorPadding)) for frame in image]))
    else:
        raise ValueError("Input must be 2D or 3D array")
