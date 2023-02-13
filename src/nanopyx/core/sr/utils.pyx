# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

from libc.math cimport sqrt, fabs, pi, cos, fmin

from ..transform.interpolation.catmull_rom cimport _interpolate, Interpolator
from ..transform.image_magnify import cv2_zoom as zoom
# from ..transform.image_magnify import fourier_zoom as zoom
from ..utils.time.timeit import timeit2


import numpy as np
cimport numpy as np

from cython.parallel import prange



def normalizeFFT(fft_real: np.ndarray, fft_imag: np.ndarray) -> np.ndarray:
    return _normalizeFFT(fft_real, fft_imag)

cdef float[:, :, :] _normalizeFFT(float[:, :] fft_real, float[:, :] fft_imag):

    cdef float[:, :, :] output = np.empty((2, fft_real.shape[0], fft_real.shape[1]), dtype=np.float32)
    cdef float mag
    cdef int x_i, y_i

    with nogil:
        for x_i in prange(fft_real.shape[1]):
            for y_i in range(fft_imag.shape[0]):
                mag = sqrt(fft_real[y_i, x_i]**2 + fft_imag[y_i, x_i])

                if mag != 0:
                    output[0] = fft_real[y_i, x_i] / mag
                    output[1] = fft_imag[y_i, x_i] / mag
                else:
                    output[0] = 0
                    output[1] = 0

    return output

def apodize_edges(img: np.ndarray) -> np.ndarray:
    return _apodize_edges(img)

cdef float[:, :] _apodize_edges(float[:, :] img):

    cdef float[:, :] output = img.copy()
    cdef float dist, edge_mean, x0, y0, d, c
    cdef int y_i, x_i
    cdef int height = img.shape[1]
    cdef int width = img.shape[0]
    cdef int offset = 20 # number of pixels used for smooth apodization
    cdef float radius = width / 2 - offset
    cdef int count = 0

    with nogil:
        for x_i in prange(width):
            for y_i in range(height):
                dist = (y_i - height//2)**2 + (x_i - width//2)**2
                if dist > radius**2:
                    edge_mean += img[y_i, x_i]
                    count += 1

    edge_mean /= count

    with nogil:
        for x_i in prange(width):
            for y_i in range(height):
                x0 = fabs(x_i - width/2)
                y0 = fabs(y_i - height/2)
                if fabs(x0 - width/2) <= offset or fabs(y0 - height/2) <= offset:
                    d = fmin(fabs(x_i - width/2), fabs(y_i - height/2))
                    c = (cos(d * pi / offset - pi) + 1) / 2
                    output[y_i, x_i] = <float> (c*(output[y_i, x_i]-edge_mean)) + edge_mean
                elif fabs(x_i - width/2) > width/2 and fabs(y_i - height/2) > height/2:
                    output[y_i, x_i] = <float> edge_mean

    return output
