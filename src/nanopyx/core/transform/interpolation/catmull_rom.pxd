
cdef double _interpolate(float[:,:] im, double x, double y) nogil
cdef float[:,:] _magnify(float[:,:] im, float[:,:] imM, int magnification) nogil
cdef float[:,:] _shift(float[:,:] im, double dx, double dy)
