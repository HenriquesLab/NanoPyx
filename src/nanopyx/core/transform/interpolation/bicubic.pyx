# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True

import numpy as np
cimport numpy as np

from cython.parallel import prange


# bicubic spline interpolation of a 2D array
cdef double _interpolate(float[:,:] image, double x, double y) nogil:
    """
    Bicubic spline interpolation of a 2D array.

    Parameters:
        image (np.ndarray): image to interpolate.
        x (double): x-coordinate to interpolate at.
        y (double): y-coordinate to interpolate at.

    Returns:
        Interpolated pixel value (float).
    """
    cdef int x0 = int(x)
    cdef int y0 = int(y)
    if x == x0 and y == y0:
        return image[y0, x0]

    cdef int w = image.shape[1]
    cdef int h = image.shape[0]
    
    if x<-1 or x>w or y<-1 or y>h:
        return 0

    cdef int i, j, _x, _y
    cdef double dx = x - x0
    cdef double dy = y - y0
    cdef double dx2 = dx * dx
    cdef double dy2 = dy * dy
    cdef double dx3 = dx2 * dx
    cdef double dy3 = dy2 * dy
    cdef double[4] a
    cdef double[4] b
    a[0] = -0.5 * dx3 + dx2 - 0.5 * dx
    a[1] = 1.5 * dx3 - 2.5 * dx2 + 1
    a[2] = -1.5 * dx3 + 2 * dx2 + 0.5 * dx
    a[3] = 0.5 * dx3 - 0.5 * dx2
    b[0] = -0.5 * dy3 + dy2 - 0.5 * dy
    b[1] = 1.5 * dy3 - 2.5 * dy2 + 1
    b[2] = -1.5 * dy3 + 2 * dy2 + 0.5 * dy
    b[3] = 0.5 * dy3 - 0.5 * dy2
    cdef double result = 0

    for i in range(4):
        for j in range(4):
            _x = max(0, min((x0-1+i)%w, w-1))
            _y = max(0, min((y0-1+j)%h, h-1))
            result += a[i] * b[j] * image[_y, _x]

    return result


def magnify(np.ndarray im, int magnification):
    """
    Magnify an image by a factor of magnification using bicubic spline interpolation.

    Parameters:
        im (np.ndarray): image to magnify.
        magnification (int): magnification factor.

    Returns:
        Magnified image (np.ndarray).
    """
    assert im.ndim == 2
    imMagnified = _magnify(im.astype(np.float32), magnification)
    return np.asarray(imMagnified).astype(im.dtype)


# bicubid magnification of a 2D array
cdef float[:,:] _magnify(float[:,:] image, float magnification):
    """
    Bicubic magnification of a 2D array.

    image (np.ndarray): image to magnify.
    magnification (float): magnification factor.

    Returns:
    Magnified image (np.ndarray).
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
                imMagnified[j, i] = _interpolate(image, x, y)
    
    return imMagnified


def shift(np.ndarray im, double dx, double dy):
    """
    Shift an image by (dx, dy) using bicubic spline interpolation.

    im (np.ndarray): image to shift.
    dx (float): shift along x-axis.
    dy (float): shift along y-axis.

    Returns:
    Shifted image (np.ndarray).
    """
    assert im.ndim == 2
    imShifted = _shift(im.astype(np.float32), dx, dy)
    return np.asarray(imShifted).astype(im.dtype)


cdef float[:,:] _shift(float[:,:] im, float dx, float dy):
    """
    Shift an image by (dx, dy) using bicubic spline interpolation.

    im (np.ndarray): image to shift.
    dx (float): shift along x-axis.
    dy (float): shift along y-axis.

    Returns:
    Shifted image (np.ndarray).
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
            imShifted[j,i] = _interpolate(im, i - dx, j - dy)

    return imShifted 
