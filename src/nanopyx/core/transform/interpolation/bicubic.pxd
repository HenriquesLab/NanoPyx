
cdef double _interpolate(float[:,:] image, double x, double y) nogil
cdef float[:,:] _magnify(float[:,:] image, float magnification)
cdef float[:,:] _shift(float[:,:] im, float dx, float dy)
