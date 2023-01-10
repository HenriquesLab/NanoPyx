import numpy as np
cimport numpy as np

def check_even_square(np.ndarray image_arr):
    return _check_even_square(image_arr)

cdef bint _check_even_square(float[:, :, :] image_arr):
    cdef int w = image_arr.shape[2]
    cdef int h = image_arr.shape[1]

    if w != h:
        return False
    if w % 2 != 0:
        return False

    return True

def get_closest_even_square_size(np.ndarray image_arr):
    return _get_closest_even_square_size(image_arr)

cdef int _get_closest_even_square_size(float[:, :, :] image_arr):
    cdef int w = image_arr.shape[2]
    cdef int h = image_arr.shape[1]
    cdef int min_size = min(w, h)

    if min_size % 2 != 0:
        min_size -= 1

    return min_size


def make_even_square(np.ndarray image_arr):
    return _make_even_square(image_arr)

cdef float[:, :, :] _make_even_square(float[:, :, :] image_arr):
    if _check_even_square(image_arr):
        return image_arr

    cdef int w = image_arr.shape[2]
    cdef int h = image_arr.shape[1]
    cdef int min_size = _get_closest_even_square_size(image_arr)

    cdef int h_start, h_finish, w_start, w_finish

    h_start = int((h-min_size)/2)
    if (h-min_size) % 2 != 0:
        h_finish = h - int((h-min_size)/2) - 1
    else:
        h_finish = h - int((h-min_size)/2)

    w_start = int((w-min_size)/2)
    if (w - min_size) % 2 != 0:
        w_finish = w - int((w-min_size)/2) - 1
    else:
        w_finish = w - int((w-min_size)/2)

    return image_arr[:, h_start:h_finish, w_start:w_finish]