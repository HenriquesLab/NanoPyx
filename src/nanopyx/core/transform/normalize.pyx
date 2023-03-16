# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

from libc.math cimport sqrt
import numpy as np
cimport numpy as np

from cython.parallel import prange


def normalizeFFT(fft_real: np.ndarray, fft_imag: np.ndarray):
    return _normalizeFFT(fft_real, fft_imag)

cdef float[:, :, :] _normalizeFFT(float[:, :] fft_real, float[:, :] fft_imag) nogil:

    cdef double mag
    cdef int x_i, y_i
    cdef float[:, :, :] output
    with gil:
        output = np.zeros((2, fft_real.shape[0], fft_real.shape[1]), dtype=np.float32)

    for x_i in prange(fft_real.shape[1]):
        for y_i in range(fft_real.shape[0]):
            mag = sqrt(fft_real[y_i, x_i]**2 + fft_imag[y_i, x_i]**2)

            if mag != 0:
                output[0][y_i, x_i] = fft_real[y_i, x_i] / mag
                output[1][y_i, x_i] = fft_imag[y_i, x_i] / mag

    return output