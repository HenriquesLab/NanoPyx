cdef extern from "_le_mandelbrot_benchmark_.h":
    int _c_mandelbrot(double row, double col) nogil


# pyx2pxd: starting point
# Code below is autogenerated by pyx2pxd