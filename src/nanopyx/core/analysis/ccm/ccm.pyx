# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

import numpy as np
cimport numpy as np

import cython
from libc.math cimport pi

from .helper_functions cimport _check_even_square, _make_even_square
from ..pearson_correlation cimport _calculate_ppmcc

from ...transform.interpolation.catmull_rom cimport Interpolator

def calculate_ccm(np.ndarray img_stack, int ref):
    """
    Function used to generate a cross correlation matrix of an image stack.
    Cross correlation is calculated using either the first image of the stack or the previous image.
    Cross correlation values are normalized by the minimum and maximum Pearson's correlation between the two
    images.
    :param img_stack: numpy array with shape (t, y, x)
    :param ref: either 0 or 1, 0 is used to calculate the ccm based on the first frame, 1 used to calculate based on the previous frame
    :return: numpy array with shape (t, y, x), corresponding to the cross correlation matrix
    """
    return _calculate_ccm(img_stack, ref)


cdef float[:, :, :] _calculate_ccm(float[:, :, :] img_stack, int ref):

    if not _check_even_square(img_stack):
        img_stack = _make_even_square(img_stack)

    cdef int stack_w = img_stack.shape[2]
    cdef int stack_h = img_stack.shape[1]
    cdef int stack_n = img_stack.shape[0]
    cdef float[:, :, :] ccm = np.empty((stack_n, stack_h, stack_w), dtype=np.float32)

    cdef float[:, :] img_ref
    cdef int i, _n 

    for i in range(stack_n):
        if ref == 0:
            img_ref = img_stack[0]
        else:
            _n = max(0, i-1)
            img_ref = img_stack[_n]
        ccm[i] = _calculate_slice_ccm(img_ref, img_stack[i])

    return ccm


def calculate_ccm_from_ref(np.ndarray img_stack, np.ndarray img_ref):
    """
    Function used to generate a cross correlation matrix of an image stack.
    Cross correlation is calculated using a static image frame.
    Cross correlation values are normalized by the minimum and maximum Pearson's correlation between the two
    images.
    :param img_stack: numpy array with shape (t, y, x)
    :param img_ref: numpy array with shape (y, x)
    :return: numpy array with shape (t, y, x), corresponding to the cross correlation matrix
    """
    return _calculate_ccm_from_ref(img_stack, img_ref)



cdef float[:, :, :] _calculate_ccm_from_ref(float[:, :, :] img_stack, float[:, :] img_ref):

    if not _check_even_square(img_stack):
        img_stack = _make_even_square(img_stack)

    cdef float[:,:,:] tmp = np.array([img_ref])
    if not _check_even_square(tmp):
        tmp = _make_even_square(tmp)
        img_ref = tmp[0]

    cdef int stack_w = img_stack.shape[2]
    cdef int stack_h = img_stack.shape[1]
    cdef int stack_n = img_stack.shape[0]
    cdef float[:, :, :] ccm = np.empty((stack_n, stack_h, stack_w), dtype=np.float32)

    for i in range(stack_n):
        ccm[i] = _calculate_slice_ccm(img_ref, img_stack[i])

    return ccm


cdef float[:, :] _calculate_slice_ccm(float[:, :] img_ref, float[:, :] img_slice):
    cdef float[:, :] ccm_slice = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(img_ref) * np.fft.fft2(img_slice).conj())).real.astype(np.float32)
    ccm_slice = ccm_slice[::-1, ::-1]
    _normalize_ccm(img_ref, img_slice, ccm_slice)

    return ccm_slice[0:ccm_slice.shape[0], 0:ccm_slice.shape[1]]


cdef void _normalize_ccm(float[:, :] img_ref, float[:, :] img_slice, float[:, :] ccm_slice) nogil:
    """
    Function used to normalize the cross correlation matrix.

    The code above does the following:
    1. Find the maximum and minimum values of the cross-correlation matrix
    2. Calculate the maximum and minimum PPMCC
    3. Normalize the matrix values to the PPMCC values
    """
    
    cdef int w = ccm_slice.shape[1]
    cdef int h = ccm_slice.shape[0]

    cdef float min_value = ccm_slice[0, 0]
    cdef float max_value = ccm_slice[0, 0]
    cdef int x_max = 0
    cdef int y_max = 0
    cdef int x_min = 0
    cdef int y_min = 0
    cdef float v

    for j in range(h):
        for i in range(w):
            v = ccm_slice[j, i]
            if v < min_value:
                min_value = v
                x_min = i
                y_min = j
            if v > max_value:
                max_value = v
                x_max = i
                y_max = j

    cdef int shift_x_max = x_max - w // 2
    cdef int shift_y_max = y_max - h // 2
    cdef int shift_x_min = x_min - w // 2
    cdef int shift_y_min = y_min - h // 2

    cdef float max_ppmcc = _calculate_ppmcc(img_ref, img_slice, shift_x_max, shift_y_max)
    cdef float min_ppmcc = _calculate_ppmcc(img_ref, img_slice, shift_x_min, shift_y_min)

    cdef float delta_v = max_value - min_value
    cdef float value
    cdef float delta_ppmcc = max_ppmcc - min_ppmcc

    for j in range(h):
        for i in range(w):
            value = (ccm_slice[j, i] - min_value) / delta_v
            value = value * delta_ppmcc + min_ppmcc
            ccm_slice[j, i] = value


def calculate_rccm(np.ndarray img_slice, np.ndarray img_ref):
    """
    Function used to generate a rotational cross correlation matrix of an image against a reference image
    Cross correlation values are normalized by the minimum and maximum Pearson's correlation between the two
    images.
    :param img_slice: numpy array with shape (y, x)
    :param img_ref: numpy array with shape (y, x)
    :return: numpy array with shape (360, y, x), corresponding to the rotational cross correlation matrix
    """
    return np.array(_calculate_rccm(img_slice, img_ref))

cdef float[:,:,:] _calculate_rccm(float[:, :] img_slice, float[:, :] img_ref):
    
    cdef float[:,:,:] tmp

    tmp = np.array([img_slice])
    if not _check_even_square(tmp):
        tmp = _make_even_square(tmp)
        img_slice = tmp[0]

    tmp = np.array([img_ref])
    if not _check_even_square(tmp):
        tmp = _make_even_square(tmp)
        img_ref = tmp[0]

    cdef int height = img_slice.shape[0]
    cdef int width = img_slice.shape[1]

    cdef float[:,:,:] rccm = np.empty((360,height,width), dtype=np.float32)

    cdef float[:,:] rotated_img_slice

    cdef int degree
    cdef float radian
    for degree in range(360):
        radian = degree * pi/180
        rotated_img_slice = Interpolator(img_slice).rotate(radian)
        rccm[degree] = _calculate_slice_ccm(img_ref, rotated_img_slice)

    return rccm

def calculate_ccm_polar(np.ndarray img_slice, np.ndarray img_ref):
    """
    Function used to generate a cross correlation matrix of an image against a reference image where both are to be expressed in polar coordinates (theta,r)
    Cross correlation values are normalized by the minimum and maximum Pearson's correlation between the two polar images.
    Rotation in cartesian space will show as translation in the theta dimension. 
    :param img_slice: numpy array with shape (y, x)
    :param img_ref: numpy array with shape (y, x)
    :return: numpy array with shape (360,r), corresponding to the cross correlation matrix in polar coordinates
    """
    return np.array(_calculate_ccm_polar(img_slice, img_ref))

cdef float[:,:] _calculate_ccm_polar(float[:, :] img_slice, float[:, :] img_ref):

    cdef float[:,:] polar_slice = Interpolator(img_slice).polar()
    cdef float[:,:] polar_ref = Interpolator(img_ref).polar()

    return _calculate_slice_ccm(polar_ref, polar_slice)


def calculate_ccm_logpolar(np.ndarray img_slice, np.ndarray img_ref):
    """
    Function used to generate a cross correlation matrix of an image against a reference image where both are to be expressed in logpolar coordinates (theta, log(r))
    Cross correlation values are normalized by the minimum and maximum Pearson's correlation between the two logpolar images.
    Rotation and scaling in cartesian space will show as translation in the theta and log(r) dimensions respectively.
    :param img_slice: numpy array with shape (y, x)
    :param img_ref: numpy array with shape (y, x)
    :return: numpy array with shape (360,log(r)), corresponding to the cross correlation matrix in polar coordinates
    """
    return np.array(_calculate_ccm_logpolar(img_slice, img_ref))

cdef float[:,:] _calculate_ccm_logpolar(float[:, :] img_slice, float[:, :] img_ref):

    cdef float[:,:] polar_slice = Interpolator(img_slice).polar(scale='log')
    cdef float[:,:] polar_ref = Interpolator(img_ref).polar(scale='log')

