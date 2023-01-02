cdef double _interpolate(float[:,:] image, double x, double y, int taps)
cdef float[:,:] _magnify(float[:,:] im, int magnification, int taps)
cdef float[:,:] _shift(float[:,:] im, double dx, double dy, int taps)