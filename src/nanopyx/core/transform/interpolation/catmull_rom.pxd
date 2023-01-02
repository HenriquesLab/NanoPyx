
cdef double _interpolate(float[:,:] im, double x, double y) nogil
cdef float[:,:] _magnify(float[:,:] im, int magnification)
cdef float[:,:] _shift(float[:,:] im, double dx, double dy)
