cimport numpy as np

cdef np.ndarray _padding(np.ndarray image_arr, int padrow, int padcol) 
cdef np.ndarray _crop(np.ndarray image_arr, int croprow, int cropcol) 
cdef np.ndarray _make_even_square(np.ndarray image_arr) 