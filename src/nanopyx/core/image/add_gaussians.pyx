# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

from libc.math cimport erf, sqrt, fabs

from cython.parallel import prange

import numpy as np
cimport numpy as np

cdef float ROOT2 = sqrt(2)


def generate_stack_with_random_particles(int width, int height, int nFrames,
                                        int particles_per_slice, double particles_per_slice_randomness,
                                        int signal, double signal_randomness,
                                        double sigma_x, double sigma_x_randomness,
                                        double sigma_y, double sigma_y_randomness):
    """
    Generate a stack of images with random particles
    :param width: the width of the image
    :param height: the height of the image
    :param nFrames: the number of frames in the stack
    :param particles_per_slice: the number of particles per slice
    :param particles_per_slice_randomness: the randomness of the number of particles per slice
    :param signal: the signal of each particle
    :param signal_randomness: the randomness of the signal of each particle
    :param sigma_x: the sigma of the gaussian in the x direction
    :param sigma_x_randomness: the randomness of the sigma of the gaussian in the x direction
    :param sigma_y: the sigma of the gaussian in the y direction
    :param sigma_y_randomness: the randomness of the sigma of the gaussian in the y direction
    :param gauss_noise_sigma: the sigma of the gaussian noise
    :param gauss_noise_mean: the mean of the gaussian noise
    :return: a stack of images with random particles
    """

    cdef float[:] x_positions, y_positions, signals, sigmas_x, sigmas_y
    cdef float[:,:,:] ims = np.zeros((nFrames, height, width), dtype=np.float32)

    cdef int n_part
    cdef int t

    for t in range(nFrames):
        n_part = int(particles_per_slice * (1 + np.random.uniform(-1, 1) * particles_per_slice_randomness))
        x_positions = np.ramdom.uniform(0, width-1, n_part)
        y_positions = np.ramdom.uniform(0, height-1, n_part)
        sigmas_x = sigma_x * (1 + sigma_x_randomness * np.random.uniform(-1, 1, n_part))
        sigmas_y = sigma_y * (1 + sigma_y_randomness * np.random.uniform(-1, 1, n_part))
        signals = signal * (1 + signal_randomness * np.random.uniform(-1, 1, n_part))
        _renderERFGaussians(ims[t], signals, sigmas_x, sigmas_y, x_positions, y_positions)
        
    return ims


cdef _renderERFGaussians(float[:,:] image, float[:] singal, float[:] sigma_x, float[:] sigma_y, float[:] x, float[:] y):
    """
    Render a set of gaussian particles on an image using the error function (erf) to calculate the integral of the gaussian
    :param image: the image to render the particles on
    :param intensity: the intensity of each particle
    :param sigma_x: the sigma of the gaussian in the x direction
    :param sigma_y: the sigma of the gaussian in the y direction
    :param x: the x position of the center of the gaussian
    :param y: the y position of the center of the gaussian
    :return: None

    Original implementation: https://github.com/HenriquesLab/NanoJ-Core/blob/80020c9cf5ecac70019daa5731d0c296cb306ac4/Core/src/nanoj/core/java/image/rendering/SubPixelGaussianRendering.java#L12
    """
    
    cdef int nParticles = singal.shape[0]
    cdef int w = image.shape[1]
    cdef int h = image.shape[0]
    cdef int p, i, j
    cdef float amp, sx, sx2, sy, sy2, xp, yp, Ex, Ey, dx, dy, v

    # we go to each pixel separately and render the information of each particle, that way we avoid concurrency issues
    with nogil:
        for j in prange(h):
            for i in range(w):
                for p in range(nParticles):
                    amp = singal[p]
                    sx = sigma_x[p]
                    sy = sigma_y[p]
                    sx2 = ROOT2 * sx # 2 * pow(sigmaX, 2)
                    sy2 = ROOT2 * sy # 2 * pow(sigmaY, 2)
                    xp = x[p]
                    yp = y[p]

                    dx = i - xp
                    dy = j - yp

                    # based on Smith et al, Nath Meth, 2010: Fast, single-molecule localization that achieves theoretically
                    # minimum uncertainty (see Sup Mat page 10)
                    # note, the paper has an error on their formula 4a and 4b, 2sigma^2 should be sqrt(2)*sigma
                    # see REF: https://en.wikipedia.org/wiki/Normal_distribution formula 'Cumulative distribution function'

                    if (fabs(dx)<sx*10 and fabs(dy)<sy*10):
                        Ex = 0.5 * (erf((i + 0.5 - xp) / sx2) - erf((i - 0.5 - xp) / sx2))
                        Ey = 0.5 * (erf((j + 0.5 - yp) / sy2) - erf((j - 0.5 - yp) / sy2))
                        image[j,i] += amp * Ex * Ey

