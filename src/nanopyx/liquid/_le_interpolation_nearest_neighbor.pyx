# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np

cimport numpy as np

from cython.parallel import parallel, prange

from . import cl, cl_array, cl_ctx, cl_queue
from .__liquid_engine__ import LiquidEngine
from ._le_mandelbrot_benchmark_ import mandelbrot as _py_mandelbrot


cdef extern from "_c_interpolation_nearest_neighbor.h":
    float _c_interpolate(float *image, float row, float col, int rows, int cols) nogil


class ShiftAndMagnify(LiquidEngine):
    """
    Mandelbrot Benchmark using the NanoPyx Liquid Engine
    """

    _has_opencl = True
    _has_threaded = False
    _has_threaded_static = False
    _has_threaded_dynamic = False
    _has_threaded_guided = False
    _has_unthreaded = True
    _has_python = False

    def run(self, image, float shift_row = 0, float shift_col = 0, float magnification_row = 1, float magnification_col = 1) -> np.ndarray:
        """
        Shift and magnify an image using nearest neighbor interpolation
        :param image: The image to shift and magnify
        :param shift_row: The number of rows to shift the image
        :param shift_col: The number of columns to shift the image
        :param magnification_row: The magnification factor for the rows
        :param magnification_col: The magnification factor for the columns
        :return: The shifted and magnified image
        """
        assert image.ndim == 2, "Image must be 2D"
        assert image.dtype == np.float32, "Image must be of type np.float32"
        return self._run(image, shift_row, shift_col, magnification_row, magnification_col)

    def benchmark(self, image, float shift_row = 0, float shift_col = 0, float magnification_row = 1, float magnification_col = 1):
        return super().benchmark(image, shift_row, shift_col, magnification_row, magnification_col)

    def _run_opencl(self, image, float shift_row, float shift_col, float magnification_row, float magnification_col) -> np.ndarray:
        code = self._get_cl_code("_le_interpolation_nearest_neighbor_.cl")

        cdef int rowsM = <int>(image.shape[0] * magnification_row)
        cdef int colsM = <int>(image.shape[1] * magnification_col)

        image_in = cl_array.to_device(cl_queue, image)
        image_out = cl_array.zeros(cl_queue, (rowsM, colsM), dtype=np.float32)

        # Create the program
        prg = cl.Program(cl_ctx, code).build()

        # Run the kernel
        prg.shiftAndMagnify(
            cl_queue,
            image_out.shape,
            None,
            image_in.data,
            image_out.data,
            np.float32(shift_row),
            np.float32(shift_col),
            np.float32(magnification_row),
            np.float32(magnification_col),
        )

        # Wait for queue to finish
        cl_queue.finish()

        return image_out.get()

    def _run_unthreaded(self, image, float shift_row, float shift_col, float magnification_row, float magnification_col) -> np.ndarray:
        cdef int rows = image.shape[0]
        cdef int cols = image.shape[1]
        cdef int rowsM = image.shape[0] * magnification_row
        cdef int colsM = image.shape[1] * magnification_col

        cdef float[:,:] _image_in = image

        image_out = np.zeros((rowsM, colsM), dtype=np.float32)
        cdef float[:,:] _image_out = image_out

        cdef int i, j
        cdef float row, col

        with nogil:
            for j in range(colsM):
                col = j / magnification_col - shift_col
                for i in range(rowsM):
                    row = i / magnification_row - shift_row
                    _image_out[i, j] = _c_interpolate(&_image_in[0, 0], row, col, rows, cols)

        return image_out
