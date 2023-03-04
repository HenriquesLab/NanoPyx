# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

# Import necessary libraries
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

from cython.parallel import prange

def add_simplex_noise(dataset: np.ndarray, double frequency = 0.01, int octaves = 4, double persistence = 2, double amplitude = 1000, double offset = 1000, int seed = -1):
    """
    Add simplex noise to a 2D numpy array.
    :param im: The 2D numpy array to add noise to.
    :param frequency: The frequency of the noise.
    :param octaves: The number of octaves to use.
    :param persistence: The persistence of the noise.
    :param amplitude: The amplitude of the noise.
    :param offset: The offset of the noise.
    :param seed: The seed to use for the noise generator, if -1 a random seed will be used.
    """
    if seed == -1:
        seed = np.random.randint(0, 1000000)

    assert dataset.ndim == 2, "The image must be 2D."
    assert frequency > 0, "The frequency must be greater than 0."
    assert octaves > 0, "The number of octaves must be greater than 0."
    assert persistence > 0, "The persistence must be greater than 0."

    # Convert the imay to a 2D numpy array
    dataset = dataset.view(np.float32)

    if dataset.ndim == 2:
        _add_simplex_noise_2D(dataset, frequency, octaves, persistence, amplitude, offset, seed)
    elif dataset.ndim == 3:
        _add_simplex_noise_3D(dataset, frequency, octaves, persistence, amplitude, offset, seed)
    elif dataset.ndim == 4:
        _add_simplex_noise_4D(dataset, frequency, octaves, persistence, amplitude, offset, seed)
    else:
        raise ValueError("The image must be 2D, 3D, or 4D.")

cdef void _add_simplex_noise_2D(float[:,:] im, double frequency, int octaves, double persistence, double amplitude, double offset, int seed) nogil:
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
            im[r,c] += noise * amplitude + offset

    # Free the memory used by the OpenSimplex environment and gradients
    free(osg)
    free(ose)

cdef void _add_simplex_noise_3D(float[:,:,:] im, double frequency, int octaves, double persistence, double amplitude, double offset, int seed) nogil:
    cdef int rows = im.shape[0]
    cdef int cols = im.shape[1]
    cdef int depth = im.shape[2]
    cdef double noise
    cdef int r, c, d, o

    # Initialize the noise generator
    cdef OpenSimplexEnv *ose = initOpenSimplex()
    cdef OpenSimplexGradients *osg = newOpenSimplexGradients(ose, seed)

    # Loop through each element in the imay
    for r in range(rows):
        for c in range(cols):
            for d in range(depth):
                # Calculate the simplex noise for the current x,y,z position
                noise = 0
                for o in range(octaves):
                    noise += (noise3_Classic(ose, osg, r * frequency * 2**o, c * frequency * 2**o, d * frequency * 2**o) + 1) / 2 * persistence**o

                # Add the noise to the current imay element
                im[r,c,d] += noise * amplitude + offset

    # Free the memory used by the OpenSimplex environment and gradients
    free(osg)
    free(ose)

cdef void _add_simplex_noise_4D(float[:,:,:,:] im, double frequency, int octaves, double persistence, double amplitude, double offset, int seed) nogil:
    cdef int rows = im.shape[0]
    cdef int cols = im.shape[1]
    cdef int depth = im.shape[2]
    cdef int time = im.shape[3]
    cdef double noise
    cdef int r, c, d, t, o

    # Initialize the noise generator
    cdef OpenSimplexEnv *ose = initOpenSimplex()
    cdef OpenSimplexGradients *osg = newOpenSimplexGradients(ose, seed)

    # Loop through each element in the imay
    for r in range(rows):
        for c in range(cols):
            for d in range(depth):
                for t in range(time):
                    # Calculate the simplex noise for the current x,y,z,t position
                    noise = 0
                    for o in range(octaves):
                        noise += (noise4_Classic(ose, osg, r * frequency * 2**o, c * frequency * 2**o, d * frequency * 2**o, t * frequency * 2**o) + 1) / 2 * persistence**o

                    # Add the noise to the current imay element
                    im[r,c,d,t] += noise * amplitude + offset

    # Free the memory used by the OpenSimplex environment and gradients
    free(osg)
    free(ose)
