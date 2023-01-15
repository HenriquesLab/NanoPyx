# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True
# nanopyx: autogen_pxd=True

import numpy as np
cimport numpy as np


def check_even_square(np.ndarray image_arr):
    """
    Function used to check if an image array is an even square.
    :param image_arr: numpy array with shape (t, y, x)
    :return: bool, True if an image array is an even square
    """
    return _check_even_square(image_arr)


cdef bint _check_even_square(float[:, :, :] image_arr) nogil:
    cdef int w = image_arr.shape[2]
    cdef int h = image_arr.shape[1]

    if w != h:
        return False
    if w % 2 != 0:
        return False

    return True


def get_closest_even_square_size(np.ndarray image_arr):
    """
    Function used to calculate the closest even square.
    :param image_arr: numpy array with shape (t, y, x); image to be cropped
    :return: int; value of dimensions to be used for cropping
    """
    return _get_closest_even_square_size(image_arr)


cdef int _get_closest_even_square_size(float[:, :, :] image_arr) nogil:
    cdef int w = image_arr.shape[2]
    cdef int h = image_arr.shape[1]
    cdef int min_size = min(w, h)

    if min_size % 2 != 0:
        min_size -= 1

    return min_size


def make_even_square(np.ndarray image_arr):
    """
    Function used to crop an image array into an even square.
    :param image_arr: numpy array with shape (t, y, x); image to be cropped
    :return: numpy array with shape (t, y, x)
    """
    return _make_even_square(image_arr)


cdef float[:, :, :] _make_even_square(float[:, :, :] image_arr) nogil:
    if _check_even_square(image_arr):
        return image_arr

    cdef int w = image_arr.shape[2]
    cdef int h = image_arr.shape[1]
    cdef int min_size = _get_closest_even_square_size(image_arr)
    cdef int h_start, h_finish, w_start, w_finish

    h_start = (h-min_size)//2
    if (h-min_size) % 2 != 0:
        h_finish = h - (h-min_size) // 2 - 1
    else:
        h_finish = h - (h-min_size) // 2

    w_start = int((w-min_size)/2)
    if (w - min_size) % 2 != 0:
        w_finish = w - (w-min_size) // 2 - 1
    else:
        w_finish = w - (w-min_size) // 2 

    return image_arr[:, h_start:h_finish, w_start:w_finish]