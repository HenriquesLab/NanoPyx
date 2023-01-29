# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

from ..utils.random cimport _random
from ..transform.interpolation.catmull_rom cimport _interpolate
from ..image.add_gaussians cimport _render_erf_gaussian

from libc.math cimport sqrt, fabs

import numpy as np
cimport numpy as np

from cython.parallel import prange
from tqdm import tqdm

def simulate_particle_field_based_on_2D_PDF(image_pdf, 
                                            min_particles: int = 10, max_particles: int = 1000, 
                                            min_distance: float = 0.1, mean_distance_threshold: float = 0, int max_tries = 3):
    """
    Simulate a particle field based on a 2D probability density function (PDF)
    :param image_pdf: 2D array of floats, the PDF
    :param min_particles: int, the minimum number of particles to simulate
    :param max_particles: int, the maximum number of particles to simulate
    :param min_distance: float, ensure that paricle distances are above minimum distance given
    :param mean_distance_threshold: float, the mean distance between closest particles, if the mean distance is below this threshold, the simulation will stop
    :param max_tries: int, the maximum number of tries to place particles before giving up
    :return: (2D array of floats, mean closest distance), for the first tupple element the shape is (n_particles, 2) where the last dimension is the x and y coordinates of the simulated particle

    The code does the following:
    1. It samples the image PDF and places a particle at a point with a probability that is proportional to the PDF at that point.
    2. It places the particles such that no two particles are closer than `min_distance` pixels.
    3. It stops placing particles once the mean distance between all particles is less than `mean_distance_threshold` pixels.

    Example:
    >>> import numpy as np
    >>> image_pdf = np.random.random((100, 200)).astype(np.float32)
    >>> particles = simulate_particle_field_based_on_2D_PDF(image_pdf, min_particles=100, mean_distance_threshold=0.1)
    """
    
    assert image_pdf.dtype == np.float32 and image_pdf.ndim == 2 and np.max(image_pdf) <= 1.0 and np.min(image_pdf) >= 0.0

    cdef float[:,:] _image_pdf = image_pdf
    
    cdef int _max_particles = max_particles
    cdef int _min_particles = min_particles
    cdef float _min_distance = min_distance
    cdef int _max_tries = max_tries
    cdef float _mean_distance_threshold = mean_distance_threshold
    
    cdef float[:] xp = np.full(_max_particles, -999999, dtype=np.float32)
    cdef float[:] yp = np.full(_max_particles, -999999, dtype=np.float32)
    
    cdef int n_particles = 0
    cdef int previous_n_particles = 0
    cdef int p
    cdef int tries = 0
    cdef float closest_distance, closest_distance_sum, mean_closest_distance

    # start by creating the minumal pool of particles    
    with tqdm(total=_max_particles, desc="Generating particles", unit="particles") as progress_bar:
        while 1:
            with nogil:
                n_particles = 0   
                closest_distance_sum = 0 
                for p in prange(_max_particles):
                    if xp[p] < 0: # particle not yet placed
                        _get_particle_candidate(_image_pdf, p, xp, yp, _min_distance)
                    else: # particle already placed, check that it doesnt break distance rule
                        closest_distance = _get_closest_distance(xp[p], yp[p], xp, yp)
                        if closest_distance < _min_distance:
                            xp[p] = -999999
                            yp[p] = -999999
                        else:
                            closest_distance_sum += closest_distance
                            n_particles += 1

                if n_particles > 0:
                    mean_closest_distance = closest_distance_sum / n_particles

                if n_particles == previous_n_particles:
                    tries += 1
                else:
                    tries = 0
                
                if n_particles == _max_particles or tries == _max_tries:
                    break

                if _mean_distance_threshold > 0 and n_particles > _min_particles and mean_closest_distance < _mean_distance_threshold:
                    # final cleaning up
                    n_particles = 0   
                    closest_distance_sum = 0
                    for p in prange(_max_particles):
                        if xp[p] >= 0:
                            closest_distance = _get_closest_distance(xp[p], yp[p], xp, yp)
                            if closest_distance < _min_distance:
                                xp[p] = -999999
                                yp[p] = -999999
                            else:
                                closest_distance_sum += closest_distance
                                n_particles += 1
                    mean_closest_distance = closest_distance_sum / n_particles
                    break

            progress_bar.update(n_particles-previous_n_particles)
            previous_n_particles = n_particles

    cdef float[:] _xp = np.zeros(n_particles, dtype=np.float32)
    cdef float[:] _yp = np.zeros(n_particles, dtype=np.float32)

    n_particles = 0
    for p in range(_max_particles):
        if xp[p] < 0:
            continue
        _xp[n_particles] = xp[p]
        _yp[n_particles] = yp[p]
        n_particles += 1

    return np.array([_xp, _yp]).T, mean_closest_distance


cdef bint _get_particle_candidate(float[:, :] _image_pdf, int particle_index, float[:] xp, float[:] yp, float min_distance) nogil:
    """
    Get a particle candidate by sampling the image PDF and placing a particle at a point with a probability that is proportional to the PDF at that point.
    :param _image_pdf: 2D array of floats, the PDF
    :param particle_index: int, the index of the particle to place
    :param xp: 1D array of floats, the x coordinates of the particles
    :param yp: 1D array of floats, the y coordinates of the particles
    :param min_distance: float, ensure that paricle distances are above minimum distance given
    :return: 1 if a particle was placed, 0 if not
    """
    
    cdef float x = _random() * (_image_pdf.shape[1] - 1)
    cdef float y = _random() * (_image_pdf.shape[0] - 1)
    cdef float pdf = _interpolate(_image_pdf, x, y)
    cdef float r = _random()
    
    if r < pdf and _get_closest_distance(x, y, xp, yp) > min_distance:
        xp[particle_index] = x
        yp[particle_index] = y
        return 1

    return 0


cdef double _get_closest_distance(float x, float y, float[:] xp, float[:] yp) nogil:
    """
    Get the closest distance between a point and a set of particles. 
    Ignores particles with exact same location as x, y.
    :param x: float, the x coordinate of the point
    :param y: float, the y coordinate of the point
    :param xp: 1D array of floats, the x coordinates of the particles
    :param yp: 1D array of floats, the y coordinates of the particles
    :return: double, the closest distance between the point and the particles
    """
    
    cdef float _x, _y
    cdef float _min_distance = 999999999999

    for i in range(xp.shape[0]):
        _x = xp[i]
        _y = yp[i]

        if _x < 0 or (x == _x and y == _y) or fabs(_x - x) > _min_distance or fabs(_y - y) > _min_distance:
            continue
        
        _min_distance = min(_min_distance, sqrt((_x - x) ** 2 + (_y - y) ** 2))

    return _min_distance


def get_closest_distance(float[:,:] particle_field):
    """
    Get the closest distance between all particles
    :param particle_field: 2D array of floats, the particle field
    :return: double, the closest distance between all particles
    """
    
    cdef float[:] xp = particle_field[:, 0]
    cdef float[:] yp = particle_field[:, 1]

    cdef float closest_distance = 999999999999
    cdef int p

    with nogil:
        for p in range(xp.shape[0]):
            if xp[p] < 0:
                continue
            closest_distance = min(closest_distance, _get_closest_distance(xp[p], yp[p], xp, yp))

    return closest_distance
    

def render_particle_histogram(float[:,:] particle_field, int w, int h):
    """
    Render a particle field as an image
    :param particle_field: 2D array of floats, the particle field with shape (n_particles, 2) where the last dimension is the x and y coordinates of the particle
    :param w: int, the width of the image
    :param h: int, the height of the image
    :return: 2D array of floats, the rendered particle field
    """

    image_particle_field = np.zeros((h, w), dtype=np.float32)
    cdef float[:,:] _image_particle_field = image_particle_field

    cdef int n_particles = particle_field.shape[0]
    cdef float[:] xp = particle_field[:, 0]
    cdef float[:] yp = particle_field[:, 1]

    cdef int x, y, i

    with nogil:
        for i in range(n_particles):
            x = int(xp[i])
            y = int(yp[i])
            if 0 <= x < w or 0 <= y < h:
                _image_particle_field[y, x] += 1

    return image_particle_field


def render_particle_histogram_with_tracks(float[:,:] particle_field, int[:,:] states, int w, int h):
    """
    Render a particle field as an image stack
    :param particle_field: 2D array of floats, the particle field
    :param states: 2D array of ints, the states of the particles
    :param w: int, the width of the stack (in pixels)
    :param h: int, the height of the stack (in pixels)
    :return: 3D array of floats, the rendered particle field
    """

    assert particle_field.shape[0] == states.shape[0]

    cdef int n_frames = states.shape[1]
    
    image_particle_field = np.zeros((n_frames, h, w), dtype=np.float32)
    cdef float[:,:,:] _image_particle_field = image_particle_field

    cdef int n_particles = particle_field.shape[0]    
    cdef float[:] xp = particle_field[:, 0]
    cdef float[:] yp = particle_field[:, 1]

    cdef int x, y, i, f

    with nogil:
        for i in prange(n_particles):
            x = int(xp[i])
            y = int(yp[i])
            if 0 <= x < w or 0 <= y < h:
                for f in range(n_frames):
                    if states[i, f] == 1:
                        _image_particle_field[f, y, x] += 1

    return image_particle_field


def render_particle_gaussians_with_tracks(float[:,:] particle_field, int[:,:] states, int w, int h, double amplitude, double sigma_x, double sigma_y):

    assert particle_field.shape[0] == states.shape[0]

    cdef int n_frames = states.shape[1]
    
    image_particle_field = np.zeros((n_frames, h, w), dtype=np.float32)
    cdef float[:,:,:] _image_particle_field = image_particle_field

    cdef int n_particles = particle_field.shape[0]    
    cdef float[:] xp = particle_field[:, 0]
    cdef float[:] yp = particle_field[:, 1]

    cdef int x, y, i, f, b, b_stop

    # break into 100 particles at a time
    with tqdm(total=n_particles, desc="Rendering particles", unit="particles") as progress_bar:
        for b in range(0, n_particles, 100):    
            with nogil:
                b_stop = min(b + 100, n_particles)
                for i in prange(b, b_stop):
                    x = int(xp[i])
                    y = int(yp[i])
                    if 0 <= x < w or 0 <= y < h:
                        for f in range(n_frames):
                            if states[i, f] == 1:
                                _render_erf_gaussian(_image_particle_field[f], x, y, amplitude, sigma_x, sigma_y)
            progress_bar.update(100)
                            
    return image_particle_field

