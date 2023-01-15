from .nearest_neighbor cimport Interpolator as InterpolatorNearestNeighbor

cdef double _interpolate(float[:,:] image, double x, double y) nogil

cdef class Interpolator(InterpolatorNearestNeighbor):
    cdef float _interpolate(self, float x, float y) nogil
