# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True

from libc.math cimport floor, isnan, isinf, fmax, fmin

import numpy as np
cimport numpy as np

from cython.parallel import prange

def interpolate(np.ndarray im, double x, double y) -> float:
    """
    Interpolate image using Catmull-Rom interpolation.

    im (np.ndarray): image to interpolate.
    x (float): x-coordinate to interpolate at.
    y (float): y-coordinate to interpolate at.

    Returns:
    Interpolated pixel value (float).
    """
    return _interpolate(im.view(np.float32), x, y).astype(im.dtype)


cdef double _interpolate(float[:,:] image, double x, double y) nogil:
    """
    Interpolate image using Catmull-Rom interpolation.

    image (np.ndarray): image to interpolate.
    x (float): x-coordinate to interpolate at.
    y (float): y-coordinate to interpolate at.

    Returns:
    Interpolated pixel value (float).
    """
    cdef int x0 = int(x)
    cdef int y0 = int(y)
    if x == x0 and y == y0:
        return image[y0, x0]

    cdef int w = image.shape[1]
    cdef int h = image.shape[0]

    if x<0 or x>=w or y<0 or y>=h:
        return 0

    cdef int u0 = int(x - 0.5)
    cdef int v0 = int(y - 0.5)
    cdef double q = 0
    cdef double p
    cdef int v, u, i, j, _u, _v

    for j in range(4):
        v = v0 - 1 + j
        p = 0
        for i in range(4):
            u = u0 - 1 + i
            _u = max(0, min(u, w-1))
            _v = max(0, min(v, h-1))
            p = p + image[_v, _u] * _cubic(x - (u + 0.5))
        q = q + p * _cubic(y - (v + 0.5))

    #if isnan(q) or isinf(q):
    #    return 0.

    return float(q)


def magnify(np.ndarray im, int magnification):
    """
    Magnify image.

    im (np.ndarray): image to magnify.
    magnification (int): magnification factor.

    Returns:
    Magnified image (np.ndarray).
    """
    assert im.ndim == 2
    imMagnified = _magnify(im.astype(np.float32), magnification)
    return np.asarray(imMagnified).astype(im.dtype)


cdef float[:,:] _magnify(float[:,:] image, int magnification):
    """
    Magnify image using Catmull-Rom interpolation.

    image (np.ndarray): image to magnify.
    magnification (int): magnification factor.

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
    Shift image.

    im (np.ndarray): image to shift.
    dx (float): shift along x-axis.
    dy (float): shift along y-axis.

    Returns:
    Shifted image (np.ndarray).
    """
    assert im.ndim == 2
    imShifted = _shift(im.astype(np.float32), dx, dy)
    return np.asarray(imShifted).astype(im.dtype)


cdef float[:,:] _shift(float[:,:] im, double dx, double dy):
    """
    Shift image using Catmull-Rom interpolation.

    im (np.ndarray): image to shift.
    imShifted (np.ndarray): shifted image.
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

cdef double _cubic(double x) nogil:
    """
    Cubic function used in Catmull-Rom interpolation.

    x (float): input value.

    Returns:
    Output value of cubic function (float).
    """
    cdef float a = 0.5  # Catmull-Rom interpolation
    cdef float z = 0
    if x < 0:
        x = -x
    if x < 1:
        z = x * x * (x * (-a + 2.) + (a - 3.)) + 1.
    elif x < 2:
        z = -a * x * x * x + 5. * a * x * x - 8. * a * x + 4.0 * a
    return z
