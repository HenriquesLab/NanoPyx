# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np

cimport numpy as np

from cython.parallel import prange

from ..__liquid_engine__ import LiquidEngine
from ..__opencl__ import cl, cl_array
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

        self._default_benchmarks = {'Numba': {"(['number(10)', 'number(-1.5)', 'number(0.5)', 'number(-1.0)', 'number(1.0)'], {})": [7.5, 0.3969438750000336, 0.0001701670000784361, 0.00015795900003467978], "(['number(20)', 'number(-1.5)', 'number(0.5)', 'number(-1.0)', 'number(1.0)'], {})": [15.0, 0.00023199999986900366, 0.00027608300001702446, 0.00027991599995402794]}, 'OpenCL': {"(['number(10)', 'number(-1.5)', 'number(0.5)', 'number(-1.0)', 'number(1.0)'], {})": [7.5, 0.008683583999982147, 0.005001416000141035, 0.004709541000011086], "(['number(20)', 'number(-1.5)', 'number(0.5)', 'number(-1.0)', 'number(1.0)'], {})": [15.0, 0.005239165999910256, 0.004791457999999693, 0.004966167000020505]}, 'Python': {"(['number(10)', 'number(-1.5)', 'number(0.5)', 'number(-1.0)', 'number(1.0)'], {})": [7.5, 0.00973437500010732, 0.007952500000101281, 0.007477000000108092], "(['number(20)', 'number(-1.5)', 'number(0.5)', 'number(-1.0)', 'number(1.0)'], {})": [15.0, 0.027458667000018977, 0.028410624999878564, 0.026648084000044037]}, 'Threaded': {"(['number(10)', 'number(-1.5)', 'number(0.5)', 'number(-1.0)', 'number(1.0)'], {})": [7.5, 0.0003130830000372953, 0.00022504100002151972, 0.00016691600012563867], "(['number(20)', 'number(-1.5)', 'number(0.5)', 'number(-1.0)', 'number(1.0)'], {})": [15.0, 0.0002568749998772546, 0.0002542080001148861, 0.00026070800004163175]}, 'Threaded_dynamic': {"(['number(10)', 'number(-1.5)', 'number(0.5)', 'number(-1.0)', 'number(1.0)'], {})": [7.5, 0.000146165999922232, 0.00015641699997104297, 0.0001227500001732551], "(['number(20)', 'number(-1.5)', 'number(0.5)', 'number(-1.0)', 'number(1.0)'], {})": [15.0, 0.00018033399987871235, 0.0002684999999473803, 0.00018504099989513634]}, 'Threaded_guided': {"(['number(10)', 'number(-1.5)', 'number(0.5)', 'number(-1.0)', 'number(1.0)'], {})": [7.5, 0.00017041599994627177, 0.0001660830000673741, 0.00014637499998571002], "(['number(20)', 'number(-1.5)', 'number(0.5)', 'number(-1.0)', 'number(1.0)'], {})": [15.0, 0.0002119999999194988, 0.00020254200012459478, 0.00020637500006159826]}, 'Threaded_static': {"(['number(10)', 'number(-1.5)', 'number(0.5)', 'number(-1.0)', 'number(1.0)'], {})": [7.5, 0.0002240000001165754, 0.00018258299996887217, 0.00018458400018062093], "(['number(20)', 'number(-1.5)', 'number(0.5)', 'number(-1.0)', 'number(1.0)'], {})": [15.0, 0.00026874999980464054, 0.0002514999998766143, 0.00028095900006519514]}, 'Unthreaded': {"(['number(10)', 'number(-1.5)', 'number(0.5)', 'number(-1.0)', 'number(1.0)'], {})": [7.5, 0.0001960000001872686, 0.00012925000010000076, 0.00012629100001504412], "(['number(20)', 'number(-1.5)', 'number(0.5)', 'number(-1.0)', 'number(1.0)'], {})": [15.0, 0.00044287500008977077, 0.0004312920000302256, 0.00043895799990423257]}}

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

    # tag-start: _le_mandelbrot_benchmark.MandelbrotBenchmark._run_unthreaded
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
    # tag-end

    # tag-copy: _le_mandelbrot_benchmark.MandelbrotBenchmark._run_unthreaded; replace('_run_unthreaded', '_run_threaded'); replace('j in range(size)', 'j in prange(size)')
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
    # tag-end

    # tag-copy: _le_mandelbrot_benchmark.MandelbrotBenchmark._run_unthreaded; replace('_run_unthreaded', '_run_threaded_static'); replace('j in range(size)', 'j in prange(size, schedule="static")')
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
    # tag-end

    # tag-copy: _le_mandelbrot_benchmark.MandelbrotBenchmark._run_unthreaded; replace('_run_unthreaded', '_run_threaded_dynamic'); replace('j in range(size)', 'j in prange(size, schedule="dynamic")')
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
    # tag-end

    # tag-copy: _le_mandelbrot_benchmark.MandelbrotBenchmark._run_unthreaded; replace('_run_unthreaded', '_run_threaded_guided'); replace('j in range(size)', 'j in prange(size, schedule="guided")')
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
    # tag-end

    def _run_python(self, int size, float r_start, float r_end, float c_start, float c_end) -> np.ndarray:
        im_mandelbrot = np.empty((size, size), dtype=np.int32)
        _py_mandelbrot(im_mandelbrot, r_start, r_end, c_start, c_end)
        return im_mandelbrot

    def _run_njit(self, int size=10, float r_start=-1.5, float r_end=0.5, float c_start=-1, float c_end=1) -> np.ndarray:
        im_mandelbrot = np.empty((size, size), dtype=np.int32)
        _njit_mandelbrot(im_mandelbrot, r_start, r_end, c_start, c_end)
        return im_mandelbrot
