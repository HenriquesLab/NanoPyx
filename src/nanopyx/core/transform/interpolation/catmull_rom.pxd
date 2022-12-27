
cdef float _interpolate(float[:,:] im, float x, float y) nogil
cdef float[:,:] _magnify(float[:,:] im, int magnification)
cdef float[:,:] _shift(float[:,:] im, float dx, float dy)
