# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

from libc.stdlib cimport rand, RAND_MAX

def random() -> float:
    """
    Returns a random value between 0 and 1.
    """
    return _random()

cdef double _random() nogil:
    # not thread safe since it depends on a time seed
    return float(rand()) / float(RAND_MAX)
