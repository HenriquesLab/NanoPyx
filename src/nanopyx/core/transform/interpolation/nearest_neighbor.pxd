
cdef double _interpolate(float[:,:] image, double x, double y) nogil

cdef class Interpolator:

    cdef: 
        float[:,:] image
        int w, h

    cdef float _interpolate(self, float x, float y) nogil
    cdef float[:,:] _magnify(self, float magnification)
    cdef float[:,:] _shift(self, float dx, float dy)
