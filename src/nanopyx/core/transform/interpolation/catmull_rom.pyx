# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True

import numpy as np
cimport numpy as np

from cython.parallel import prange

def interpolate(np.ndarray im, float x, float y) -> float:
    return _interpolate(im.view(np.float32), x, y).astype(im.dtype)


cdef float _interpolate(float[:,:] im, float x, float y) nogil:
    """
    Carryout Catmull-Rom interpolation, 
    """
    cdef int w = im.shape[0]
    cdef int h = im.shape[1]
    cdef int _x = int(x)
    cdef int _y = int(y)

    if x<0.5 or x>w-1.5 or y<0.5 or y>h-1.5:
        return im[_x, _y]
    
    cdef int u0 = int(x - 0.5)
    cdef int v0 = int(y - 0.5)
    cdef float q = 0
    cdef float p
    cdef int v, u, i, j

    for j in range(4):
        v = v0 - 1 + j
        p = 0
        for i in range(4):
            u = u0 - 1 + i
            p = p + im[u, v] * _cubic(x - (u + 0.5))
        q = q + p * _cubic(y - (v + 0.5))

    return q


def magnify(np.ndarray im, int magnification):
    assert im.ndim == 2
    cdef np.ndarray im_new = np.asarray(_magnify(im.astype(np.float32), magnification))
    return im_new.astype(im.dtype)


cdef float[:,:] _magnify(float[:,:] im, int magnification):
    cdef int w = im.shape[0]
    cdef int h = im.shape[1]
    cdef int wM = w * magnification
    cdef int hM = h * magnification

    cdef float[:,:] imMagnified = np.empty((wM, hM), dtype=np.float32)

    cdef int i, j
    with nogil:
        for j in prange(hM):
            for i in range(wM):
                imMagnified[i,j] = _interpolate(im, i / magnification, j / magnification)

    return imMagnified


def shift(np.ndarray im, float dx, float dy):
    assert im.ndim == 2
    cdef np.ndarray im_new = np.asarray(_shift(im.astype(np.float32), dx, dy))
    return im_new.astype(im.dtype)


cdef float[:,:] _shift(float[:,:] im, float dx, float dy):
    cdef int w = im.shape[0]
    cdef int h = im.shape[1]

    cdef float[:,:] imShifted = np.empy((w, h), dtype=np.float32)

    cdef int i, j
    with nogil:
        for j in prange(h):
            for i in range(w):
                imShifted[i,j] = _interpolate(im, i + dx, j + dy)

    return imShifted


cdef float _cubic(float x) nogil:
    cdef float a = 0.5  # Catmull-Rom interpolation
    cdef float z = 0
    if x < 0: 
        x = -x
    if x < 1:
        z = x * x * (x * (-a + 2.) + (a - 3.)) + 1.
    elif x < 2:
        z = -a * x * x * x + 5. * a * x * x - 8. * a * x + 4.0 * a
    return z
