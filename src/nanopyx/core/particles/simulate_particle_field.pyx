# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

from ..utils.random cimport _random
from ..transform.interpolation.catmull_rom cimport _interpolate

from libc.math cimport sqrt, fabs

import numpy as np
cimport numpy as np


def simulate_particle_field_based_on_2D_PDF(image_pdf, 
                                            min_particles: int = 10, max_particles: int = 1000, 
                                            min_distance: float = 0.01, mean_distance_threshold: float = 0):
    """
    Simulate a particle field based on a 2D probability density function (PDF)
    :param image_pdf: 2D array of floats, the PDF
    :param min_particles: int, the minimum number of particles to simulate
    :param max_particles: int, the maximum number of particles to simulate
    :param min_distance: float, the minimum distance between particles
    :param mean_distance_threshold: float, the mean distance between particles, if the mean distance is below this threshold, the simulation will stop
    :return: 2D array of floats, the simulated particle field

    The code does the following:
    1. It samples the image PDF and places a particle at a point with a probability that is proportional to the PDF at that point.
    2. It places the particles such that no two particles are closer than `min_distance` pixels.
    3. It stops placing particles once the mean distance between all particles is less than `mean_distance_threshold` pixels.

    Example:
    >>> import numpy as np
    >>> image_pdf = np.random.random((100, 200)).astype(np.float32)
    >>> particles = simulate_particle_field_based_on_2D_PDF(image_pdf, min_particles=100, mean_distance_threshold=0.1)
    >>> particles.shape
    (100, 2)
    """
    
    assert image_pdf.dtype == np.float32 and image_pdf.ndim == 2 and np.max(image_pdf) <= 1.0 and np.min(image_pdf) >= 0.0

    cdef int _width = image_pdf.shape[1] - 1
    cdef int _height = image_pdf.shape[0] - 1
    cdef float[:,:] _image_pdf = image_pdf

    cdef double distance_sum = 0
    cdef int n_particles = 0
    cdef int _max_particles = max_particles
    cdef int _min_particles = min_particles
    cdef bint passes_min_distance
    
    cdef float[:] xp = np.zeros(_max_particles, dtype=np.float32)
    cdef float[:] yp = np.zeros(_max_particles, dtype=np.float32)

    cdef float r, x, y, _x, _y, d, pdf 
    
    with nogil:
        while n_particles < _max_particles:
            x = _random() * _width
            y = _random() * _height
            pdf = _interpolate(_image_pdf, x, y)
            if pdf == 0:
                continue

            r = _random()
            if r < pdf:
                passes_min_distance = 1
                for i in range(n_particles):
                    _x = xp[i]
                    _y = yp[i]
                    if fabs(_x - x) > min_distance or fabs(_y - y) > min_distance:
                        continue
                    d = sqrt((_x - x) ** 2 + (_y - y) ** 2)
                    if d < min_distance:
                        passes_min_distance = 0
                        break
                if passes_min_distance == 0:
                    continue

                xp[n_particles] = x
                yp[n_particles] = y
                n_particles += 1
            
                if n_particles > _min_particles and mean_distance_threshold > 0:
                    distance_sum = 0
                    for i in range(n_particles):
                        for j in range(i):
                            distance_sum += sqrt((xp[i] - xp[j]) ** 2 + (yp[i] - yp[j]) ** 2)
                    if distance_sum / n_particles ** 2 < mean_distance_threshold:
                        break

    return np.array([xp[:n_particles], yp[:n_particles]]).T
