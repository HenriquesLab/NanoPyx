# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

# Import necessary libraries
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

from cython.parallel import prange


def add_simplex_noise(im, double frequency = 0.1, int octaves = 4, double persistence = 0.1, double amplitude = 1, int seed = 1234):
    """
    Add simplex noise to a 2D numpy array.
    :param im: The 2D numpy array to add noise to.
    :param frequency: The frequency of the noise.
    :param octaves: The number of octaves to use.
    :param persistence: The persistence of the noise.
    :param seed: The seed to use for the noise generator.
    """

    assert im.ndim == 2, "The image must be 2D."
    assert frequency > 0, "The frequency must be greater than 0."
    assert octaves > 0, "The number of octaves must be greater than 0."
    assert persistence > 0, "The persistence must be greater than 0."

    # Convert the imay to a 2D numpy array
    im = im.view(np.float32)

    # Call the Cython function to add simplex noise to the imay
    _add_simplex_noise(im, frequency, octaves, persistence, amplitude, seed)


cdef void _add_simplex_noise(float[:,:] im, double frequency, int octaves, double persistence, double amplitude, int seed) nogil:
    cdef int rows = im.shape[0]
    cdef int cols = im.shape[1]
    cdef double noise
    cdef int r, c, o

    # Initialize the noise generator
    cdef OpenSimplexEnv *ose = initOpenSimplex()
    cdef OpenSimplexGradients *osg = newOpenSimplexGradients(ose, seed)

    # Loop through each element in the imay
    for r in range(rows):
        for c in range(cols):
            # Calculate the simplex noise for the current x,y position
            noise = 0
            for o in range(octaves):
                noise += (noise2(ose, osg, r * frequency * 2**o, c * frequency * 2**o) + 1) / 2 * persistence**o

            # Add the noise to the current imay element
            im[r,c] += noise * amplitude

    # Free the memory used by the OpenSimplex environment and gradients
    free(osg)
    free(ose)
