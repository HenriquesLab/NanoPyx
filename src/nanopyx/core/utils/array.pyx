# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

import numpy as np
cimport numpy as np


def get_max(arr: np.ndarray, x1: int, x2: int):
    return np.array(_get_max(arr, x1, x2), dtype=np.float32)

cdef float[:] _get_max(float[:] arr, int x1, int x2) nogil:
    cdef int k
    cdef float[:] out
    with gil:
        out = np.empty((2), dtype=np.float32)
    out[0] = x1
    out[1] = arr[x1]

    for k in range(x1, x2):
        if arr[k] > out[1]:
            out[0] = k
            out[1] = arr[k]

    return out

def get_min(arr: np.ndarray, x1: int, x2: int):
    return np.array(_get_min(arr, x1, x2), dtype=np.float32)

cdef float[:] _get_min(float[:] arr, int x1, int x2) nogil:
    cdef int k
    cdef float[:] out
    with gil:
        out = np.empty((2), dtype=np.float32)
    out[0] = x1
    out[1] = arr[x1]

    for k in range(x1, x2):
        if arr[k] < out[1]:
            out[0] = k
            out[1] = arr[k]

    return out