# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

import numpy as np
from cython.parallel import prange

from . import cl, cl_array, ctx, get_kernel_txt, queue

cimport numpy as np


cdef int DEFAULT_PROCESSOR = 0

cpdef mandelbrot(int size, int max_iter, float divergence = 4):
    """
    Creates a mandelbrot set
    :param size: Size of the mandelbrot set
    :param max_iter: Maximum number of iterations
    :param divergence: The divergence threshold
    :return: The mandelbrot set
    """
    if DEFAULT_PROCESSOR == 0 and cl is not None:
        return _cl_mandelbrot(size, max_iter, divergence)
    elif DEFAULT_PROCESSOR == 1:
        return _t_mandelbrot(size, max_iter, divergence)
    elif DEFAULT_PROCESSOR == 2:
        return _nt_mandelbrot(size, max_iter, divergence)
    return _t_mandelbrot(size, max_iter, divergence)

# liquid CL
cdef int[:,:] _cl_mandelbrot(int size, int max_iter, double divergence):
    # Create the mandelbrot set
    im_mandelbrot = cl_array.zeros(queue, (size, size), dtype=np.int32)

    # Create the kernel
    kernel_txt = get_kernel_txt(__file__)
    prg = cl.Program(ctx, kernel_txt).build()

    # Run the kernel
    prg.mandelbrot(
        queue,
        im_mandelbrot.shape,
        None,
        im_mandelbrot.data,
        np.int32(max_iter),
        np.float64(divergence),
    )
    queue.finish()

    return im_mandelbrot.get()

# liquid cython-threaded
cdef int[:,:] _t_mandelbrot(int size, int max_iter, double divergence) nogil:
    cdef int i, j
    cdef int[:, :] image
    with gil:
        image = np.zeros((size, size), dtype=np.int32)
    for i in prange(size):
        for j in range(size):
            image[i, j] = _c_mandelbrot(i, j, max_iter, divergence)
    return image

# liquid cython-nonthreaded
cdef int[:,:] _nt_mandelbrot(int size, int max_iter, double divergence) nogil:
    cdef int i, j
    cdef int[:, :] image
    with gil:
        image = np.zeros((size, size), dtype=np.int32)
    for i in range(size):
        for j in range(size):
            image[i, j] = _c_mandelbrot(i, j, max_iter, divergence)
    return image
