# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True

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


cdef double _interpolate(float[:,:] im, double x, double y) nogil:
    """
    Interpolate image using Catmull-Rom interpolation.

    im (np.ndarray): image to interpolate.
    x (float): x-coordinate to interpolate at.
    y (float): y-coordinate to interpolate at.

    Returns:
    Interpolated pixel value (float).
    """
    cdef int w = im.shape[0]
    cdef int h = im.shape[1]
    cdef int _x = int(x)
    cdef int _y = int(y)

    if x<0.5 or x>w-1.5 or y<0.5 or y>h-1.5:
        return im[_x, _y]
    
    cdef int u0 = int(x - 0.5)
    cdef int v0 = int(y - 0.5)
    cdef double q = 0
    cdef double p
    cdef int v, u, i, j

    for j in range(4):
        v = v0 - 1 + j
        p = 0
        for i in range(4):
            u = u0 - 1 + i
            p = p + im[u, v] * _cubic(x - (u + 0.5))
        q = q + p * _cubic(y - (v + 0.5))

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
    return imMagnified.astype(im.dtype)


cdef float[:,:] _magnify(float[:,:] im, int magnification):
    """
    Magnify image using Catmull-Rom interpolation.

    im (np.ndarray): image to magnify.
    imM (np.ndarray): magnified image.
    magnification (int): magnification factor.

    Returns:
    Magnified image (np.ndarray).
    """

    cdef int wM = im.shape[0] * magnification
    cdef int hM = im.shape[1] * magnification
    cdef int i, j
    cdef float _x, _y

    cdef float[:,:] imMagnified = np.zeros((im.shape[0] * magnification, im.shape[1] * magnification), dtype=np.float32)

    for j in range(hM):
        _y = j / magnification
        for i in range(wM):
            _x = i / magnification
            imMagnified[i,j] = _interpolate(im, _x, _y)


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
    imShifted = _shift(im.view(np.float32), dx, dy)
    return imShifted.astype(im.dtype)


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
    cdef int w = im.shape[0]
    cdef int h = im.shape[1]
    cdef int i, j

    cdef float[:,:] imShifted = np.zeros((w, h), dtype=np.float32)

    for j in range(h):
        for i in range(w):
            imShifted[i,j] = _interpolate(im, i + dx, j + dy)


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
