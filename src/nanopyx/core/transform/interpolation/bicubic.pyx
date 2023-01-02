# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True

import numpy as np
cimport numpy as np

from cython.parallel import prange


# bicubic spline interpolation of a 2D array
cdef float _interpolate(float[:,:] image, float x, float y):
    """
    Bicubic spline interpolation of a 2D array.

    image (np.ndarray): image to interpolate.
    x (float): x-coordinate to interpolate at.
    y (float): y-coordinate to interpolate at.

    Returns:
    Interpolated pixel value (float).
    """
    cdef int i, j
    cdef int m = image.shape[0]
    cdef int n = image.shape[1]
    cdef int x0 = int(x)
    cdef int y0 = int(y)
    cdef float dx = x - x0
    cdef float dy = y - y0
    cdef float dx2 = dx * dx
    cdef float dy2 = dy * dy
    cdef float dx3 = dx2 * dx
    cdef float dy3 = dy2 * dy
    cdef float[4] a
    cdef float[4] b
    a[0] = -0.5 * dx3 + dx2 - 0.5 * dx
    a[1] = 1.5 * dx3 - 2.5 * dx2 + 1
    a[2] = -1.5 * dx3 + 2 * dx2 + 0.5 * dx
    a[3] = 0.5 * dx3 - 0.5 * dx2
    b[0] = -0.5 * dy3 + dy2 - 0.5 * dy
    b[1] = 1.5 * dy3 - 2.5 * dy2 + 1
    b[2] = -1.5 * dy3 + 2 * dy2 + 0.5 * dy
    b[3] = 0.5 * dy3 - 0.5 * dy2
    cdef float result = 0

    for i in range(4):
        for j in range(4):
            result += a[i] * b[j] * image[(x0-1+i)%m][(y0-1+j)%n]

    return result


def magnify(np.ndarray im, int magnification):
    """
    Magnify an image by a factor of magnification using bicubic spline interpolation.

    im (np.ndarray): image to magnify.
    magnification (int): magnification factor.

    Returns:
    Magnified image (np.ndarray).
    """
    assert im.ndim == 2
    imMagnified = _magnify(im.view(np.float32), magnification)
    return imMagnified.astype(im.dtype)


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
    cdef int m = image.shape[0]
    cdef int n = image.shape[1]
    cdef int m2 = int(m * magnification)
    cdef int n2 = int(n * magnification)
    cdef float x, y

    cdef float[:,:] imMagnified = np.zeros((image.shape[0] * magnification, image.shape[1] * magnification), dtype=np.float32)

    for i in range(m2):
        x = i / magnification
        for j in range(n2):
            y = j / magnification
            imMagnified[i, j] = _interpolate(image, x, y)
    
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
    imShifted = _shift(im.view(np.float32), dx, dy)
    return imShifted.astype(im.dtype)


cdef float[:,:] _shift(float[:,:] im, float dx, float dy):
    """
    Shift an image by (dx, dy) using bicubic spline interpolation.

    im (np.ndarray): image to shift.
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

    return imShifted
