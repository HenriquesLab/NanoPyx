import numpy as np
cimport numpy as np

from cython.parallel import prange
from .helper_functions import check_even_square, make_even_square
from ..analysis.pearson_correlation import calculate_ppmcc

def calculate_ccm(np.ndarray img_stack, int ref):
    return _calculate_ccm(img_stack, ref)

cdef float[:, :, :] _calculate_ccm(float[:, :, :] img_stack, int ref):

    if not check_even_square(np.array(img_stack).astype(np.float32)):
        img_stack = make_even_square(np.array(img_stack).astype(np.float32))

    cdef int stack_w = img_stack.shape[2]
    cdef int stack_h = img_stack.shape[1]
    cdef int stack_n = img_stack.shape[0]
    cdef float[:, :, :] ccm = np.zeros((stack_n, stack_h, stack_w)).astype(np.float32)

    cdef float[:, :] img_ref

    for i in range(stack_n):
        if ref == 0:
            img_ref = img_stack[0]
        else:
            img_ref = img_stack[max(0, i-1)]
        ccm[i] = _calculate_slice_ccm(img_ref, img_stack[i])

    return ccm

def calculate_ccm_from_ref(np.ndarray img_stack, np.ndarray img_ref):
    return _calculate_ccm_from_ref(img_stack, img_ref)

cdef float[:, :, :] _calculate_ccm_from_ref(float[:, :, :] img_stack, float[:, :] img_ref):

    if not check_even_square(np.array(img_stack).astype(np.float32)):
        img_stack = make_even_square(np.array(img_stack).astype(np.float32))

    tmp = np.array([img_ref]).astype(np.float32)
    if not check_even_square(np.array(tmp).astype(np.float32)):
        tmp = make_even_square(np.array(tmp).astype(np.float32))
        img_ref = tmp[0]

    cdef int stack_w = img_stack.shape[2]
    cdef int stack_h = img_stack.shape[1]
    cdef int stack_n = img_stack.shape[0]
    cdef float[:, :, :] ccm = np.zeros((stack_n, stack_h, stack_w)).astype(np.float32)

    for i in range(stack_n):
        ccm[i] = _calculate_slice_ccm(img_ref, img_stack[i])

    return ccm

cdef float[:, :] _calculate_slice_ccm(float[:, :] img_ref, float[:, :] img_slice):
    cdef float[:, :] ccm_slice = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(img_ref) * np.fft.fft2(img_slice).conj())).real.astype(np.float32)
    ccm_slice = ccm_slice[::-1, ::-1]
    ccm_slice = _normalize_ccm(img_ref, img_slice, ccm_slice)

    return ccm_slice[0:ccm_slice.shape[0], 0:ccm_slice.shape[1]]

cdef _normalize_ccm(float[:, :] img_ref, float[:, :] img_slice, float[:, :] ccm_slice):
    cdef int w = ccm_slice.shape[1]
    cdef int h = ccm_slice.shape[0]

    cdef float[:] ccm_pixels = np.array(ccm_slice).ravel()
    cdef int min_idx = np.argmin(ccm_pixels)
    cdef float min_value = ccm_pixels[min_idx]
    cdef int max_idx = np.argmax(ccm_pixels)
    cdef float max_value = ccm_pixels[max_idx]

    cdef int shift_x_max = int((max_idx % w) - w / 2)
    cdef int shift_y_max = int((max_idx / h) - h / 2)
    cdef int shift_x_min = int((min_idx % w) - w / 2)
    cdef int shift_y_min = int((min_idx / h) - h / 2)

    cdef float max_ppmcc = calculate_ppmcc(np.array(img_ref).astype(np.float32), np.array(img_slice).astype(np.float32), shift_x_max, shift_y_max)
    cdef float min_ppmcc = calculate_ppmcc(np.array(img_ref).astype(np.float32), np.array(img_slice).astype(np.float32), shift_x_min, shift_y_min)

    cdef float delta_v = max_value - min_value
    cdef float value

    for i in range(ccm_pixels.shape[0]):
        value = (ccm_pixels[i] - min_value) / delta_v
        value = (value * (max_ppmcc - min_ppmcc)) + min_ppmcc
        ccm_pixels[i] = value.real

    return np.array(ccm_pixels).reshape((h, w))
