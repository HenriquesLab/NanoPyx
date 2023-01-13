cdef bint _check_even_square(float[:, :, :] image_arr) nogil
cdef int _get_closest_even_square_size(float[:, :, :] image_arr) nogil
cdef float[:, :, :] _make_even_square(float[:, :, :] image_arr) nogil