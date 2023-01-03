# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True

from libc.math cimport pi, fabs, sin, floor, ceil, M_PI

import numpy as np
cimport numpy as np

from cython.parallel import prange

cdef double _interpolate(float[:,:] image, double x, double y, int taps) nogil:
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
    cdef int x0 = int(x)
    cdef int y0 = int(y)
    if x == x0 and y == y0:
        return image[y0, x0]

    cdef int w = image.shape[1]
    cdef int h = image.shape[0]
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
            interpolation += image[j, i] * weight
            weight_sum += weight
    
    return float(interpolation / weight_sum)


# Lanczos kernel function
cdef double _lanczos_kernel(double x, int taps) nogil:
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
    imMagnified = _magnify(im.astype(np.float32), magnification, taps)
    return np.asarray(imMagnified).astype(im.dtype)


cdef float[:,:] _magnify(float[:,:] image, int magnification, int taps):
    """
    Magnify a 2D image using the Lanczos interpolation method.
    
    Parameters:
        image (np.ndarray): The 2D image to magnify.
        magnification (int): The magnification factor.
        taps (int): The number of taps (interpolation points) to use in the Lanczos kernel.
    """
    cdef int i, j
    cdef int w = image.shape[1]
    cdef int h = image.shape[0]
    cdef int w2 = int(w * magnification)
    cdef int h2 = int(h * magnification)
    cdef float x, y

    cdef float[:,:] imMagnified = np.empty((h2, w2), dtype=np.float32)

    with nogil:
        for i in prange(w2):
            x = i / magnification
            for j in range(h2):
                y = j / magnification
                imMagnified[j, i] = _interpolate(image, x, y, taps)
    
    return imMagnified


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
    imShifted = _shift(im.astype(np.float32), dx, dy, taps)
    return np.asarray(imShifted).astype(im.dtype)


cdef float[:,:] _shift(float[:,:] im, double dx, double dy, int taps):
    """
    Shift a 2D image using the Lanczos interpolation method.

    Parameters:
        im (np.ndarray): The 2D image to shift.
        dx (float): The amount to shift the image in the x direction.
        dy (float): The amount to shift the image in the y direction.
        taps (int): The number of taps (interpolation points) to use in the Lanczos kernel.
    """
    cdef int w = im.shape[1]
    cdef int h = im.shape[0]
    cdef int i, j
    cdef int _dx = int(dx)
    cdef int _dy = int(dy)

    cdef float[:,:] imShifted = np.zeros((h, w), dtype=np.float32)

    cdef int x_start = max(0, _dx)
    cdef int y_start = max(0, _dy)
    cdef int x_end = min(w, w + _dx)
    cdef int y_end = min(h, h + _dy)
    
    for i in range(x_start, x_end):
        for j in range(y_start, y_end):
            imShifted[j,i] = _interpolate(im, i - dx, j - dy, taps)

    return imShifted 
