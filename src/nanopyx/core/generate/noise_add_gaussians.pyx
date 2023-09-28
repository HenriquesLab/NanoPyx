# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

from libc.math cimport erf, sqrt

from cython.parallel import prange

import numpy as np
cimport numpy as np

cdef double ROOT2 = sqrt(2)
cdef double SIGMA_CUTTOFF = 4


def render_random_gaussians(width: int, height: int, nFrames: int, particles_per_slice: int,
                            amplitude: float = 1000, sigma_x: float = 1.5, sigma_y: float = 1.5,
                            particles_per_slice_randomness: float = 0.25,
                            amplitude_randomness: float = 0.25,
                            sigma_x_randomness: float = 0.1,
                            sigma_y_randomness: float = 0.1):
    """
    Generate a stack of images with random particles
    :param width: the width of the image
    :param height: the height of the image
    :param nFrames: the number of frames in the stack
    :param particles_per_slice: the number of particles per frame
    :param amplitude: the amplitude of the particles
    :param sigma_x: the sigma of the particles in the x direction
    :param sigma_y: the sigma of the particles in the y direction
    :param particles_per_slice_randomness: the randomness of the number of particles per frame
    :param amplitude_randomness: the randomness of the amplitude of the particles
    :param sigma_x_randomness: the randomness of the sigma of the particles in the x direction
    :param sigma_y_randomness: the randomness of the sigma of the particles in the y direction
    :return: a stack of images with random particles
    """

    cdef double[:] x_positions, y_positions, amplitudes, sigmas_x, sigmas_y
    cdef float[:,:,:] ims = np.zeros((nFrames, height, width), dtype=np.float32)

    cdef int n_particles
    cdef int nf

    for nf in range(nFrames):
        n_particles = int(particles_per_slice * (1 + np.random.uniform(-1, 1) * particles_per_slice_randomness))
        x_positions = np.random.uniform(0, width-1, n_particles)
        y_positions = np.random.uniform(0, height-1, n_particles)
        sigmas_x = sigma_x * (1 + sigma_x_randomness * np.random.uniform(-1, 1, n_particles))
        sigmas_y = sigma_y * (1 + sigma_y_randomness * np.random.uniform(-1, 1, n_particles))
        amplitudes = amplitude * (1 + amplitude_randomness * np.random.uniform(-1, 1, n_particles))
        render_gaussians(ims[nf], x_positions, y_positions, amplitudes, sigmas_x, sigmas_y)

    return ims


def render_gaussians(float[:,:] image, double[:] x, double[:] y, double[:] amplitude, double[:] sigma_x, double[:] sigma_y):
    """
    Render a set of gaussian particles on an image using the error function (erf) to calculate the integral of the gaussian
    :param image: the image to render the particles on
    :param xp: the x position of the center of the gaussian
    :param yp: the y position of the center of the gaussian
    :param amplitude: the amplitude of the gaussian
    :param sigma_x: the sigma of the gaussian in the x direction
    :param sigma_y: the sigma of the gaussian in the y direction
    :return: the image with the particles rendered on it

    Original implementation: https://github.com/HenriquesLab/NanoJ-Core/blob/80020c9cf5ecac70019daa5731d0c296cb306ac4/Core/src/nanoj/core/java/image/rendering/SubPixelGaussianRendering.java#L12
    """

    cdef int p
    cdef int nParticles = amplitude.shape[0]

    with nogil:
        for p in prange(nParticles):
            _render_erf_gaussian(image, x[p], y[p], amplitude[p], sigma_x[p], sigma_y[p])


cdef float[:,:] _render_erf_gaussian(float[:,:] image, double xp, double yp, double amplitude, double sigma_x, double sigma_y) nogil:

    cdef int w = image.shape[1]
    cdef int h = image.shape[0]

    cdef double sx2 = ROOT2 * sigma_x # 2 * pow(sigmaX, 2)
    cdef double sy2 = ROOT2 * sigma_y # 2 * pow(sigmaY, 2)

    cdef int x_start = max(<int>(xp - SIGMA_CUTTOFF * sigma_x), 0)
    cdef int x_end = min(<int>(xp + SIGMA_CUTTOFF * sigma_x + 2), w) # plus 2 because <int> rounds down and range end in max_value - 1
    cdef int y_start = max(<int>(yp - SIGMA_CUTTOFF * sigma_y), 0)
    cdef int y_end = min(<int>(yp + SIGMA_CUTTOFF * sigma_y + 2), h) # plus 2 because <int> rounds down and range end in max_value - 1

    cdef int i, j
    cdef double Ex, Ey

    # based on Smith et al, Nath Meth, 2010: Fast, single-molecule localization that achieves theoretically
    # minimum uncertainty (see Sup Mat page 10)
    # note, the paper has an error on their formula 4a and 4b, 2sigma^2 should be sqrt(2)*sigma
    # see REF: https://en.wikipedia.org/wiki/Normal_distribution formula 'Cumulative distribution function'

    for j in range(y_start, y_end):
        Ey = 0.5 * (erf((j + 0.5 - yp) / sy2) - erf((j - 0.5 - yp) / sy2))
        for i in range(x_start, x_end):
            Ex = 0.5 * (erf((i + 0.5 - xp) / sx2) - erf((i - 0.5 - xp) / sx2))
            image[j,i] += amplitude * Ex * Ey
            
    return image

def render_erf_gaussian(image, xp, yp, amplitude, sigma_x, sigma_y):
    """
    Render a gaussian particle on an image using the error function (erf) to calculate the integral of the gaussian
    :param image: the image to render the particles on
    :param xp: the x position of the center of the gaussian
    :param yp: the y position of the center of the gaussian
    :param amplitude: the intensity of each particle
    :param sigma_x: the sigma of the gaussian in the x direction
    :param sigma_y: the sigma of the gaussian in the y direction
    """
    image = np.asarray(image, dtype=np.float32)
    return _render_erf_gaussian(image, xp, yp, amplitude, sigma_x, sigma_y)
