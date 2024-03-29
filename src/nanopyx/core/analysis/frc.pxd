# Code below is autogenerated by pyx2pxd - https://github.com/HenriquesLab/pyx2pxd


cdef class FIRECalculator:
    cdef float pixel_size, threshold
    cdef str units
    cdef float[:] threshold_curve
    cdef float[:, :] frc_curve, intersections
    cdef int field_of_view
    cdef public float fire_number
    cdef float[:, :] _get_squared_tapered_image(self, float[:, :] img)
    cdef float _interpolate_y(self, float x1, float y1, float x2, float y2, float x)
    cdef _compute(self, float[:, :] images, float[:] data_a1, float[:] data_b1, float[:] data_a2, float[:] data_b2)
    cdef float[:] _get_interpolated_values(self, float y, float x, float[:, :] images, int maxx) nogil
    cdef _get_smoothed_curve(self)
    cdef _calculate_threshold_curve(self)
    cdef _calculate_frc_value(self, int centre, int size, float[:, :] images, float pixel_size)
    cdef _calculate_frc_curve(self, float[:, :] img1, float[:, :] img2)
    cdef _get_intersections(self)
    cdef _calculate_fire_number(self, float[:, :] img_1, float[:, :] img_2)
