# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

from libc.math cimport fabs, pi, cos, fmin

import numpy as np
cimport numpy as np

from cython.parallel import prange

def apodize_edges(img: np.ndarray):
    return _apodize_edges(img)

cdef float[:, :] _apodize_edges(float[:, :] img) nogil:

    cdef float[:] pin
    cdef float d, c
    cdef int y_i, x_i
    cdef int height = img.shape[0]
    cdef int width = img.shape[1]
    cdef int offset = 20 # number of pixels used for smooth apodization
    cdef float radius = width / 2 - offset
    cdef int count = 0
    cdef float edge_mean = 0
    cdef float dist = 0
    cdef float x0 = 0
    cdef float y0 = 0

    with gil:
        pin = np.ravel(np.copy(img))
        
    for x_i in prange(width):
        for y_i in range(height):
            dist = (y_i - height/2)**2 + (x_i - width/2)**2
            if dist > radius**2:
                edge_mean += pin[y_i*width + x_i]
                count += 1

    with gil:
        edge_mean /= count
        
    for x_i in prange(width):
        for y_i in range(height):
            x0 = fabs(x_i - width/2)
            y0 = fabs(y_i - height/2)
            if fabs(x0 - width/2) <= offset or fabs(y0 - height/2) <= offset:
                d = fmin(fabs(x0 - width/2), fabs(y0 - height/2))
                c = (cos(d * pi / offset - pi) + 1) / 2
                pin[y_i*width + x_i] = <float> (c*(pin[y_i*width + x_i]-edge_mean)) + edge_mean
            elif fabs(x_i - width/2) > width/2 and fabs(y_i - height/2) > height/2:
                pin[y_i*width + x_i] = <float> edge_mean
    with gil:
        return np.reshape(pin, (height, width))