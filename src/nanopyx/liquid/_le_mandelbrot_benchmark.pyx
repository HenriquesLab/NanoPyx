# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

import numpy as np

cimport numpy as np

from pathlib import Path

from cython.parallel import parallel, prange

from . import cl, cl_array, cl_ctx, cl_queue
from .__liquid_engine__ import LiquidEngine
from ._le_mandelbrot_benchmark_ import mandelbrot as _py_mandelbrot


class MandelbrotBenchmark(LiquidEngine):
    """
    Mandelbrot Benchmark using the NanoPyx Liquid Engine
    """

    _has_opencl = True
    _has_threaded = True
    _has_threaded_static = True
    _has_threaded_dynamic = True
    _has_threaded_guided = True
    _has_unthreaded = True
    _has_python = True

    def run(self, int size=1000, double r_start=-1.5, double r_end=0.5, double c_start=-1, double c_end=1) -> np.ndarray:
        """
        Run the mandelbrot benchmark
        :param size: Size of the image to generate (size x size)
        :param r_start: Start of the row axis
        :param r_end: End of the row axis
        :param c_start: Start of the column axis
        :param c_end: End of the column axis
        :return: The mandelbrot set as a numpy array
        """
        return self._run(size, r_start, r_end, c_start, c_end)

    def benchmark(self, int size, double r_start=-1.5, double r_end=0.5, double c_start=-1, double c_end=1):
        return super().benchmark(size, r_start, r_end, c_start, c_end)

    def _run_opencl(self, int size, double r_start, double r_end, double c_start, double c_end) -> np.ndarray:
        code = self._get_cl_code(Path(__file__).parent / "_le_mandelbrot_benchmark_.cl")

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
            np.float64(r_start),
            np.float64(r_end),
            np.float64(c_start),
            np.float64(c_end)
        )

        # Wait for queue to finish
        cl_queue.finish()

        return im_mandelbrot.get()

    def _run_unthreaded(self, int size, double r_start, double r_end, double c_start, double c_end) -> np.ndarray:
        im_mandelbrot = np.empty((size, size), dtype=np.int32)
        cdef int[:,:] _im_mandelbrot = im_mandelbrot

        # Calculate the mandelbrot set
        cdef int i, j
        cdef double row, col
        with nogil:
            for j in range(size):
                col = c_start + j * (c_end - c_start) / size
                for i in range(size):
                    row = r_start + i * (r_end - r_start) / size
                    _im_mandelbrot[i, j] = _c_mandelbrot(row, col)

        return im_mandelbrot

    def _run_threaded(self, int size, double r_start, double r_end, double c_start, double c_end) -> np.ndarray:
        im_mandelbrot = np.empty((size, size), dtype=np.int32)
        cdef int[:,:] _im_mandelbrot = im_mandelbrot

        # Calculate the mandelbrot set
        cdef int i, j
        cdef double row, col
        with nogil:
            for j in prange(size):
                col = c_start + j * (c_end - c_start) / size
                for i in range(size):
                    row = r_start + i * (r_end - r_start) / size
                    _im_mandelbrot[i, j] = _c_mandelbrot(row, col)

        return im_mandelbrot

    def _run_threaded_static(self, int size, double r_start, double r_end, double c_start, double c_end) -> np.ndarray:
        im_mandelbrot = np.empty((size, size), dtype=np.int32)
        cdef int[:,:] _im_mandelbrot = im_mandelbrot

        # Calculate the mandelbrot set
        cdef int i, j
        cdef double row, col
        with nogil:
            for j in prange(size, schedule="static"):
                col = c_start + j * (c_end - c_start) / size
                for i in range(size):
                    row = r_start + i * (r_end - r_start) / size
                    _im_mandelbrot[i, j] = _c_mandelbrot(row, col)

        return im_mandelbrot

    def _run_threaded_dynamic(self, int size, double r_start, double r_end, double c_start, double c_end) -> np.ndarray:
        im_mandelbrot = np.empty((size, size), dtype=np.int32)
        cdef int[:,:] _im_mandelbrot = im_mandelbrot

        # Calculate the mandelbrot set
        cdef int i, j
        cdef double row, col
        with nogil:
            for j in prange(size, schedule="dynamic"):
                col = c_start + j * (c_end - c_start) / size
                for i in range(size):
                    row = r_start + i * (r_end - r_start) / size
                    _im_mandelbrot[i, j] = _c_mandelbrot(row, col)

        return im_mandelbrot

    def _run_threaded_guided(self, int size, double r_start, double r_end, double c_start, double c_end) -> np.ndarray:
        im_mandelbrot = np.empty((size, size), dtype=np.int32)
        cdef int[:,:] _im_mandelbrot = im_mandelbrot

        # Calculate the mandelbrot set
        cdef int i, j
        cdef double row, col
        with nogil:
            for j in prange(size, schedule="guided"):
                col = c_start + j * (c_end - c_start) / size
                for i in range(size):
                    row = r_start + i * (r_end - r_start) / size
                    _im_mandelbrot[i, j] = _c_mandelbrot(row, col)

        return im_mandelbrot

    def _run_python(self, int size, double r_start, double r_end, double c_start, double c_end) -> np.ndarray:
        im_mandelbrot = np.empty((size, size), dtype=np.int32)
        _py_mandelbrot(im_mandelbrot, r_start, r_end, c_start, c_end)
        return im_mandelbrot
