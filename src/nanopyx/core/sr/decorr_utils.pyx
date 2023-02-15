# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

from libc.math cimport sqrt, fabs, pi, cos, fmin

from ..transform.interpolation.catmull_rom cimport _interpolate, Interpolator
from ..transform.image_magnify import cv2_zoom as zoom
# from ..transform.image_magnify import fourier_zoom as zoom
from ..utils.time.timeit import timeit2


import numpy as np
cimport numpy as np

from cython.parallel import prange



def normalizeFFT(fft_real: np.ndarray, fft_imag: np.ndarray):
    return np.array(_normalizeFFT(fft_real, fft_imag), dtype=np.float32)

cdef float[:, :, :] _normalizeFFT(float[:, :] fft_real, float[:, :] fft_imag):

    cdef float[:, :, :] output = np.zeros((2, fft_real.shape[0], fft_real.shape[1]), dtype=np.float32)
    cdef float mag
    cdef int x_i, y_i
    
    with nogil:
        for x_i in prange(fft_real.shape[1]):
            for y_i in range(fft_real.shape[0]):
                mag = sqrt(fft_real[y_i, x_i]**2 + fft_imag[y_i, x_i]**2)

                if mag != 0:
                    output[0][y_i, x_i] = fft_real[y_i, x_i] / mag
                    output[1][y_i, x_i] = fft_imag[y_i, x_i] / mag

    return output

def apodize_edges(img: np.ndarray):
    return np.array(_apodize_edges(img), dtype=np.float32)

cdef float[:, :] _apodize_edges(float[:, :] img):

    cdef float[:, :] output = np.copy(img)
    cdef float dist, edge_mean, x0, y0, d, c
    cdef int y_i, x_i
    cdef int height = img.shape[0]
    cdef int width = img.shape[1]
    cdef int offset = 20 # number of pixels used for smooth apodization
    cdef float radius = width / 2 - offset
    cdef int count = 0

    with nogil:
        for x_i in prange(width):
            for y_i in range(height):
                dist = (y_i - height//2)**2 + (x_i - width//2)**2
                if dist > radius**2:
                    edge_mean += output[y_i, x_i]
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

def linmap(val: float, valmin: float, valmax: float, mapmin:float, mapmax:float):
    return float(_linmap(val, valmin, valmax, mapmin, mapmax))


cdef float _linmap(float val, float valmin, float valmax, float mapmin, float mapmax):

    cdef float out = 0
    out = (val - valmin)/(valmax - valmin)
    out = out * (mapmax-mapmin) + mapmin
    return out

def get_mask(w: int, r2: float):

    return np.array(_get_mask(w, r2), dtype=np.float32)

cdef float[:, :] _get_mask(int w, float r2):

    cdef int y_i, x_i
    cdef float radius = r2 * w * w / 4
    cdef float dist

    cdef float[:, :] mask = np.empty((w, w), dtype=np.float32)

    with nogil:
        for y_i in prange(w):
            for x_i in range(w):
                dist = (y_i - w/2)**2 + (x_i - w/2)**2
                if dist < radius:
                    mask[y_i, x_i] = 1

    return mask

def get_corr_coef_norm(fft_real: np.ndarray, fft_imag: np.ndarray, mask: np.ndarray):

    return float(_get_corr_coef_norm(fft_real, fft_imag, mask))

cdef float _get_corr_coef_norm(float[:, :] fft_real, float[:, :] fft_imag, float[:, :] mask):

    cdef int y_i, x_i
    cdef float c = 0
    
    with nogil:
        for y_i in prange(fft_real.shape[0]):
            for x_i in range(fft_real.shape[1]):
                if mask[y_i, x_i] == 1:
                    c += fft_real[y_i, x_i]**2 + fft_imag[y_i, x_i]**2

    return sqrt(c)

def get_max(arr: np.ndarray, x1: int, x2: int):
    return np.array(_get_max(arr, x1, x2), dtype=np.float32)

cdef float[:] _get_max(float[:] arr, int x1, int x2):
    cdef int k
    cdef float[:] out = np.empty((2), dtype=np.float32)
    out[0] = x1
    out[1] = arr[x1]

    for k in range(x1, x2):
        if arr[k] > out[1]:
            out[0] = k
            out[1] = arr[k]

    return out

def get_min(arr: np.ndarray, x1: int, x2: int):
    return np.array(_get_min(arr, x1, x2), dtype=np.float32)

cdef float[:] _get_min(float[:] arr, int x1, int x2):
    cdef int k
    cdef float[:] out = np.empty((2), dtype=np.float32)
    out[0] = x1
    out[1] = arr[x1]

    for k in range(x1, x2):
        if arr[k] < out[1]:
            out[0] = k
            out[1] = arr[k]

    return out

def get_best_score(kc: np.ndarray, a: np.ndarray):
    return np.array(_get_best_score(kc, a), dtype=np.float32)

cdef _get_best_score(float[:] kc, float[:] a):
    cdef int k
    cdef float gm_max
    cdef float[:] gm = np.empty(kc.shape[0], dtype=np.float32)
    cdef float[:] out = np.empty((3), dtype=np.float32)
    gm_max = 0.0
    
    for k in range(kc.shape[0]):
        gm[k] = kc[k]*a[k]
        
        if gm[k] > gm_max:
            gm_max = gm[k]
            out[0] = kc[k]
            out[1] = a[k]
            out[2] = k
            
    return out

def get_max_score(kc: np.ndarray, a: np.ndarray):
    return np.array(_get_max_score(kc, a), dtype=np.float32)

cdef _get_max_score(float[:] kc, float[:] a):
    cdef int k
    cdef float kc_max
    cdef float[:] out = np.empty((3), dtype=np.float32)
    kc_max = 0.0

    for k in range(kc.shape[0]):
        if kc[k] > kc_max:
            kc_max = kc[k]
            out[0] = kc[k]
            out[1] = a[k]
            out[2] = k
            
    return out