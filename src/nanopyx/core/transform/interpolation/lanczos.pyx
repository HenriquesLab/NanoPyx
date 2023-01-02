# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True

from libc.math cimport pi, fabs, sin, floor, ceil, M_PI

import numpy as np
cimport numpy as np

from cython.parallel import prange

cdef double _interpolate(float[:,:] image, double x, double y, int taps):
    """
    Interpolate the value of a 2D image at the given coordinates using the Lanczos interpolation method.
    
    Parameters:
        image (np.ndarray): The 2D image to interpolate.
        x (float): The x coordinate of the point to interpolate.
        y (float): The y coordinate of the point to interpolate.
        taps (int): The number of taps (interpolation points) to use in the Lanczos kernel.
        
    Returns:
        The interpolated value at the given coordinates.
    """
    
    cdef int w = image.shape[0]
    cdef int h = image.shape[1]
    cdef double x_factor, y_factor
    cdef int i, j
    
    # Determine the low and high indices for the x and y dimensions
    cdef int x_low = int(floor(x) - taps)
    cdef int x_high = int(ceil(x) + taps)
    cdef int y_low = int(floor(y) - taps)
    cdef int y_high = int(ceil(y) + taps)
    
    # Initialize the interpolation value to 0
    cdef double interpolation = 0
    cdef double weight
    cdef double weight_sum = 0
    
    # Loop over the taps in the x and y dimensions
    for i in range(x_low, x_high+1):
        x_factor = _lanczos_kernel(x - i, taps)
        for j in range(y_low, y_high+1):
            y_factor = _lanczos_kernel(y - j, taps)                        
            # Check if the indices are in bounds
            i = max(0, min(i, w-1))
            j = max(0, min(j, h-1))

            # Add the contribution from this tap to the interpolation
            weight = x_factor * y_factor
            interpolation += image[i, j] * weight
            weight_sum += weight
    
    return float(interpolation / weight_sum)


# Lanczos kernel function
cdef double _lanczos_kernel(double x, int taps):
    """
    Calculate the Lanczos kernel (windowed sinc function) value for a given value.
    REF: https://en.wikipedia.org/wiki/Lanczos_resampling
    
    Parameters:
        x (float): The value for which to calculate the kernel.
        taps (int): The number of taps (interpolation points) in the kernel.
        
    Returns:
        The kernel value for the given value.
    """
    if x == 0:
        return 1.0
    elif fabs(x) < taps:
        return taps * sin(pi * x) * sin(M_PI * x / taps) / (M_PI * M_PI * x * x)
    else:
        return 0.0


def magnify(np.ndarray im, int magnification, int taps):
    """
    Magnify a 2D image using the Lanczos interpolation method.
    
    Parameters:
        im (np.ndarray): The 2D image to magnify.
        magnification (int): The magnification factor.
        taps (int): The number of taps (interpolation points) to use in the Lanczos kernel.
        
    Returns:
        The magnified image.
    """
    assert im.ndim == 2
    imMagnified = _magnify(im.view(np.float32), magnification, taps)
    return imMagnified.astype(im.dtype)


cdef float[:,:] _magnify(float[:,:] im, int magnification, int taps):
    """
    Magnify a 2D image using the Lanczos interpolation method.
    
    Parameters:
        im (np.ndarray): The 2D image to magnify.
        magnification (int): The magnification factor.
        taps (int): The number of taps (interpolation points) to use in the Lanczos kernel.
    """
    cdef int w = im.shape[0]
    cdef int h = im.shape[1]
    cdef int wM = w * magnification
    cdef int hM = h * magnification
    cdef int i, j
    cdef float _x, _y

    cdef float[:,:] imMagnified = np.zeros((wM, hM), dtype=np.float32)

    for j in range(hM):
        _y = j / magnification
        for i in range(wM):
            _x = i / magnification
            imMagnified[i,j] = _interpolate(im, _x, _y, taps)


def shift(np.ndarray im, double dx, double dy, int taps):
    """
    Shift a 2D image using the Lanczos interpolation method.

    Parameters:
        im (np.ndarray): The 2D image to shift.
        dx (float): The amount to shift the image in the x direction.
        dy (float): The amount to shift the image in the y direction.
        taps (int): The number of taps (interpolation points) to use in the Lanczos kernel.
    """
    assert im.ndim == 2    
    imShifted = _shift(im.view(np.float32), dx, dy, taps)
    return imShifted.astype(im.dtype)


cdef float[:,:] _shift(float[:,:] im, double dx, double dy, int taps):
    """
    Shift a 2D image using the Lanczos interpolation method.

    Parameters:
        im (np.ndarray): The 2D image to shift.
        dx (float): The amount to shift the image in the x direction.
        dy (float): The amount to shift the image in the y direction.
        taps (int): The number of taps (interpolation points) to use in the Lanczos kernel.
    """
    cdef int w = im.shape[0]
    cdef int h = im.shape[1]
    cdef int i, j

    cdef float[:,:] imShifted = np.zeros((w, h), dtype=np.float32)

    for j in range(h):
        for i in range(w):
            imShifted[i,j] = _interpolate(im, i + dx, j + dy, taps)
