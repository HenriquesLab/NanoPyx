# Import necessary libraries
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

# Define a function that adds 2D simplex noise to a 2D numpy array
def add_simplex_noise(float[:,:] im, int seed):
    cdef int rows = im.shape[0]
    cdef int cols = im.shape[1]
    cdef double noise

    # Initialize the noise generator
    cdef OpenSimplexEnv *ose = initOpenSimplex()
    cdef OpenSimplexGradients *osg = newOpenSimplexGradients(ose, seed)

    # # Loop through each element in the imay
    # for i in range(rows):
    #     for j in range(cols):
    #         # Calculate the simplex noise for the current x,y position
    #         noise = noise2(ose, osg, j, i)

    #         # Add the noise to the current imay element
    #         im[i,j] += noise

    # # Free the memory used by the OpenSimplex environment and gradients
    # free(osg)
    # free(ose)
