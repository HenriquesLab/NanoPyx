# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

import numpy as np
cimport numpy as np


def padding(np.ndarray image_arr, int padrow, int padcol)->np.ndarray:
    """
    Function used to ZERO PAD an image array. It tries to crop maintaining the center of the image
    :param image_arr: numpy array with the shape (...,row,col) to be padded
    :param padrow: number of rows to pad
    :param padcol: number of cols to pad
    :return: numpy array with shape (...,row+padrow, col+padcol)
    """
    return np.array(_padding(image_arr, padrow, padcol))

cdef np.ndarray _padding(np.ndarray image_arr, int padrow, int padcol):
    
    cdef int ndim = image_arr.ndim 
    cdef int ncol = image_arr.shape[ndim-1]
    cdef int nrow = image_arr.shape[ndim-2]
    
    cdef int nt 
    cdef float[:,:,:] buffer

    if ndim == 2:
        nt = 1
        buffer = image_arr[np.newaxis,:,:]
    else: 
        nt = image_arr.shape[0]
        buffer = image_arr
        
    padded = np.zeros((nt,nrow+padrow,ncol+padcol), dtype=np.float32)
    
    cdef int row_start, row_finish, col_start, col_finish
    row_start = padrow//2
    if padrow % 2 != 0:
        row_finish = (nrow+padrow) - padrow // 2 - 1
    else:
        row_finish = (nrow+padrow) - padrow // 2

    col_start = padcol//2
    if padcol % 2 != 0:
        col_finish = (ncol+padcol) - padcol // 2 - 1
    else:
        col_finish = (ncol+padcol) - padcol // 2 

    padded[:, row_start:row_finish, col_start:col_finish] = buffer

    if ndim == 2:
        return padded[0,:,:]
    else:
        return padded


def crop(np.ndarray image_arr, int croprow, int cropcol)->np.ndarray:
    """
    Function used to CROP an image array. It tries to crop maintaining the center of the image
    :param image_arr: numpy array with shape (...,row,col) to be cropped
    :param croprow: number of rows to be cropped
    :param cropcol: number of cols to be cropped
    :return: numpy array with shape (...,row-croprow, col-cropcol)
    """
    return np.array(_crop(image_arr, croprow, cropcol))

cdef np.ndarray _crop(np.ndarray image_arr, int croprow, int cropcol):

    cdef int ndim = image_arr.ndim 
    cdef int ncol = image_arr.shape[ndim-1]
    cdef int nrow = image_arr.shape[ndim-2]

    cdef int row_start, row_finish, col_start, col_finish

    row_start = croprow//2
    if croprow % 2 != 0:
        row_finish = nrow - croprow // 2 - 1
    else:
        row_finish = nrow - croprow // 2

    col_start = cropcol//2
    if cropcol % 2 != 0:
        col_finish = ncol - cropcol // 2 - 1
    else:
        col_finish = ncol - cropcol // 2 

    if ndim == 3:
        return image_arr[:, row_start:row_finish, col_start:col_finish]
    else: 
        return image_arr[row_start:row_finish, col_start:col_finish]

def make_even_square(np.ndarray image_arr)->np.ndarray:
    """
    Function used to CROP an image array into an even square.
    :param image_arr: numpy array with shape (..., row, col); image to be cropped
    :return: numpy array with shape (..., row, col)
    """
    return np.array(_make_even_square(image_arr))


cdef np.ndarray _make_even_square(np.ndarray image_arr):

    cdef int ndim = image_arr.ndim
    cdef int ncol = image_arr.shape[ndim-1]
    cdef int nrow = image_arr.shape[ndim-2]

    if ncol%2==0 and ncol==nrow:
        return image_arr

    cdef int min_size = min(ncol, nrow)
    if min_size%2 != 0:
        min_size -= 1

    cdef int row_crop_amount = nrow - min_size
    cdef int col_crop_amount = ncol - min_size
    
    cdef np.ndarray cropped = _crop(image_arr, row_crop_amount, col_crop_amount)

    return cropped