# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

from libc.math cimport sqrt, fabs, pi, cos, fmin

import numpy as np
cimport numpy as np

from scipy.signal.windows import tukey
from cython.parallel import prange

def interpolate_y(x1: int, y1: int, x2: int, y2: int, x: int):
    return _interpolate_y(x1, y1, x2, y2, x)

cdef float _interpolate_y(int x1, int y1, int x2, int y2, int x):

    cdef float m = (y2 -y1) / (x2 - x1)
    cdef float c = y1 - m * x1

    return m * x + c

def compute(images: np.ndarray, data_a1: np.ndarray, data_b1: np.ndarray, data_a2: np.ndarray, data_b2: np.ndarray):
    _compute(images, data_a1, data_b1, data_a2, data_b2)

cdef _compute(float[:, :] images, float[:] data_a1, float[:] data_b1, float[:] data_a2, float[:] data_b2):

    cdef int i
    cdef float a1, a2, b1, b2

    cdef int size = data_a1.shape[0]

    with nogil:
        for i in prange(size):
            a1 = data_a1[i]
            a2 = data_a2[i]
            b1 = data_b1[i]
            b2 = data_b2[i]
            
            images[0, i] = a1 * a2 + b1 * b2
            images[1, i] = a1 * a1 + b1 * b1
            images[2, i] = a2 * a2 + b2 * b2

def get_interpolated_values(y: float, x: float, images: np.ndarray, maxx: int):
    return _get_interpolated_values(y, x, images, maxx)

cdef float[:] _get_interpolated_values(float y, float x, float[:, :] images, int maxx):
    cdef int x_base = int(x)
    cdef int y_base = int(y)
    cdef float x_fraction = x - x_base
    cdef float y_fraction = y - y_base
    if x_fraction < 0:
        x_fraction = 0
    if y_fraction < 0:
        y_fraction = 0

    cdef int lower_left_index = y_base * maxx + x_base
    cdef int lower_right_index = lower_left_index + 1
    cdef int upper_left_index = lower_left_index + maxx
    cdef int upper_right_index = upper_left_index + 1

    cdef float[:] values = np.empty((images.shape[0]), dtype=np.float32)
    cdef float lower_left, lower_right, upper_left, upper_right, upper_average, lower_average

    cdef int i

    with nogil:
        for i in prange(images.shape[0]):
            lower_left = images[i][lower_left_index]
            lower_right = images[i][lower_right_index]
            upper_left = images[i][upper_left_index]
            upper_right = images[i][upper_right_index]

            upper_average = upper_left + x_fraction * (upper_right - upper_left)
            lower_average = lower_left + x_fraction * (lower_right - lower_left)
            values[i] = lower_average + y_fraction * (upper_average - lower_average)

    return values

def get_sine(angle: float, cos_a: float):
    return _get_sine(angle, cos_a)

cdef float _get_sine(float angle, float cos_a) nogil:
    if angle > pi:
        return sqrt(1 - (cos_a * cos_a)) * -1
    else:
        return sqrt(1 - (cos_a * cos_a))