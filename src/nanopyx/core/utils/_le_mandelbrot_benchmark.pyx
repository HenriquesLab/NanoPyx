# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np

cimport numpy as np

from cython.parallel import prange

from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array
from ._le_mandelbrot_benchmark_ import mandelbrot as _py_mandelbrot
from ._le_mandelbrot_benchmark_ import njit_mandelbrot as _njit_mandelbrot


cdef extern from "_c_mandelbrot_benchmark.h":
    int _c_mandelbrot(float row, float col) nogil

class MandelbrotBenchmark(LiquidEngine):
    """
    Mandelbrot Benchmark using the NanoPyx Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "Mandelbrot_Benchmark"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, 
                        opencl_=True, unthreaded_=True, threaded_=True, threaded_static_=True, 
                        threaded_dynamic_=True, threaded_guided_=True, python_=True, njit_=True)

    def run(self, int size=1000, float r_start=-1.5, float r_end=0.5, float c_start=-1, float c_end=1, run_type=None) -> np.ndarray:
        """
        Run the mandelbrot benchmark
        :param size: Size of the image to generate (size x size)
        :param r_start: Start of the row axis
        :param r_end: End of the row axis
        :param c_start: Start of the column axis
        :param c_end: End of the column axis
        :return: The mandelbrot set as a numpy array
        """
        return self._run(size, r_start, r_end, c_start, c_end, run_type=run_type)

    def benchmark(self, int size, float r_start=-1.5, float r_end=0.5, float c_start=-1, float c_end=1):
        return super().benchmark(size, r_start, r_end, c_start, c_end)

    def _run_opencl(self, int size, float r_start, float r_end, float c_start, float c_end, dict device) -> np.ndarray:

        # QUEUE AND CONTEXT
        cl_ctx = cl.Context([device['device']])
        cl_queue = cl.CommandQueue(cl_ctx)

        code = self._get_cl_code("_le_mandelbrot_benchmark_.cl", device['DP'])

        # Create array for mandelbrot set
        im_mandelbrot = cl_array.zeros(cl_queue, (size, size), dtype=np.int32)

        # Create the program
        prg = cl.Program(cl_ctx, code).build()

        # Run the kernel
        prg.mandelbrot(
            cl_queue,
            im_mandelbrot.shape,
            None,
            im_mandelbrot.data,
            np.float32(r_start),
            np.float32(r_end),
            np.float32(c_start),
            np.float32(c_end)
        )

        # Wait for queue to finish
        cl_queue.finish()

        return im_mandelbrot.get()

    def _run_unthreaded(self, int size, float r_start, float r_end, float c_start, float c_end) -> np.ndarray:
        im_mandelbrot = np.empty((size, size), dtype=np.int32)
        cdef int[:,:] _im_mandelbrot = im_mandelbrot

        # Calculate the mandelbrot set
        cdef int i, j
        cdef float row, col
        with nogil:
            for j in range(size):
                col = c_start + j * (c_end - c_start) / size
                for i in range(size):
                    row = r_start + i * (r_end - r_start) / size
                    _im_mandelbrot[i, j] = _c_mandelbrot(row, col)

        return im_mandelbrot

    def _run_threaded(self, int size, float r_start, float r_end, float c_start, float c_end) -> np.ndarray:
        im_mandelbrot = np.empty((size, size), dtype=np.int32)
        cdef int[:,:] _im_mandelbrot = im_mandelbrot

        # Calculate the mandelbrot set
        cdef int i, j
        cdef float row, col
        with nogil:
            for j in prange(size):
                col = c_start + j * (c_end - c_start) / size
                for i in range(size):
                    row = r_start + i * (r_end - r_start) / size
                    _im_mandelbrot[i, j] = _c_mandelbrot(row, col)

        return im_mandelbrot
    def _run_threaded_guided(self, int size, float r_start, float r_end, float c_start, float c_end) -> np.ndarray:
        im_mandelbrot = np.empty((size, size), dtype=np.int32)
        cdef int[:,:] _im_mandelbrot = im_mandelbrot

        # Calculate the mandelbrot set
        cdef int i, j
        cdef float row, col
        with nogil:
            for j in prange(size, schedule="guided"):
                col = c_start + j * (c_end - c_start) / size
                for i in range(size):
                    row = r_start + i * (r_end - r_start) / size
                    _im_mandelbrot[i, j] = _c_mandelbrot(row, col)

        return im_mandelbrot
    def _run_threaded_dynamic(self, int size, float r_start, float r_end, float c_start, float c_end) -> np.ndarray:
        im_mandelbrot = np.empty((size, size), dtype=np.int32)
        cdef int[:,:] _im_mandelbrot = im_mandelbrot

        # Calculate the mandelbrot set
        cdef int i, j
        cdef float row, col
        with nogil:
            for j in prange(size, schedule="dynamic"):
                col = c_start + j * (c_end - c_start) / size
                for i in range(size):
                    row = r_start + i * (r_end - r_start) / size
                    _im_mandelbrot[i, j] = _c_mandelbrot(row, col)

        return im_mandelbrot
    def _run_threaded_static(self, int size, float r_start, float r_end, float c_start, float c_end) -> np.ndarray:
        im_mandelbrot = np.empty((size, size), dtype=np.int32)
        cdef int[:,:] _im_mandelbrot = im_mandelbrot

        # Calculate the mandelbrot set
        cdef int i, j
        cdef float row, col
        with nogil:
            for j in prange(size, schedule="static"):
                col = c_start + j * (c_end - c_start) / size
                for i in range(size):
                    row = r_start + i * (r_end - r_start) / size
                    _im_mandelbrot[i, j] = _c_mandelbrot(row, col)

        return im_mandelbrot

    def _run_python(self, int size, float r_start, float r_end, float c_start, float c_end) -> np.ndarray:
        im_mandelbrot = np.empty((size, size), dtype=np.int32)
        _py_mandelbrot(im_mandelbrot, r_start, r_end, c_start, c_end)
        return im_mandelbrot

    def _run_njit(self, int size=10, float r_start=-1.5, float r_end=0.5, float c_start=-1, float c_end=1) -> np.ndarray:
        im_mandelbrot = np.empty((size, size), dtype=np.int32)
        _njit_mandelbrot(im_mandelbrot, r_start, r_end, c_start, c_end)
        return im_mandelbrot
