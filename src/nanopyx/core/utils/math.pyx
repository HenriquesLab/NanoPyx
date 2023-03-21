# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

from libc.math cimport sqrt, pi

def get_sine(angle: float, cos_a: float):
    return _get_sine(angle, cos_a)

cdef float _get_sine(float angle, float cos_a) nogil:
    if angle > pi:
        return sqrt(1 - (cos_a * cos_a)) * -1
    else:
        return sqrt(1 - (cos_a * cos_a))