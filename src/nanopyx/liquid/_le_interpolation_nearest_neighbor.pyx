# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
cimport numpy as np

from cython.parallel import parallel, prange

from . import cl, cl_array, cl_ctx, cl_queue
from .__liquid_engine__ import LiquidEngine

from ._le_interpolation_nearest_neighbor_ import nearest_neighbor as _py_nearest_neighbor
from ._le_interpolation_nearest_neighbor_ import njit_nearest_neighbor as _njit_nearest_neighbor


cdef extern from "_c_interpolation_nearest_neighbor.h":
    float _c_interpolate(float *image, float row, float col, int rows, int cols) nogil


class ShiftAndMagnify(LiquidEngine):
    """
    Shift and Magnify Benchmark using the NanoPyx Liquid Engine
    """

    _has_opencl = True
    _has_threaded = True
    _has_threaded_static = True
    _has_threaded_dynamic = True
    _has_threaded_guided = True
    _has_unthreaded = True
    _has_python = True
    _has_njit = True


    def _parse_arguments(self, image, shift_row, shift_col):
        """
        Parse the arguments to the run function
        :param image: The image to shift and magnify
        :type image: np.ndarray
        :param shift_row: The number of rows to shift the image
        :type shift_row: int or float or np.ndarray
        :param shift_col: The number of columns to shift the image
        :type shift_col: int or float or np.ndarray
        :return: The parsed arguments
        :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        """
        if type(image) is not np.ndarray:
            raise TypeError("Image must be of type np.ndarray")
        if image.ndim != 2 and image.ndim != 3:
            raise ValueError("Image must be 2D and 3D (sequence of 2D images)")
        if image.dtype != np.float32:
            raise TypeError("Image must be of type np.float32")
        if image.ndim == 2:
            image = image.reshape((1, image.shape[0], image.shape[1])).astype(np.float32, copy=False)

        if type(shift_row) in (int, float):
            shift_row = np.ones(image.shape[0], dtype=np.float32) * shift_row
        if type(shift_col) in (int, float):
            shift_col = np.ones(image.shape[0], dtype=np.float32) * shift_col

        return image, shift_row, shift_col

    def run(self, image, shift_row, shift_col, float magnification_row, float magnification_col) -> np.ndarray:
        """
        Shift and magnify an image using nearest neighbor interpolation
        :param image: The image to shift and magnify
        :type image: np.ndarray
        :param shift_row: The number of rows to shift the image
        :type shift_row: int or float or np.ndarray
        :param shift_col: The number of columns to shift the image
        :type shift_col: int or float or np.ndarray
        :param magnification_row: The magnification factor for the rows
        :type magnification_row: flot
        :param magnification_col: The magnification factor for the columns
        :type magnification_col: float
        :return: The shifted and magnified image
        """
        image, shift_row, shift_col = self._parse_arguments(image, shift_row, shift_col)
        return self._run(image, shift_row, shift_col, magnification_row, magnification_col)

    def benchmark(self, image, shift_row, shift_col, float magnification_row, float magnification_col):
        """
        Benchmark the ShiftAndMagnify run function in multiple run types
        :param image: The image to shift and magnify
        :type image: np.ndarray
        :param shift_row: The number of rows to shift the image
        :type shift_row: int or float or np.ndarray
        :param shift_col: The number of columns to shift the image
        :type shift_col: int or float or np.ndarray
        :param magnification_row: The magnification factor for the rows
        :type magnification_row: float
        :param magnification_col: The magnification factor for the columns
        :type magnification_col: float
        :return: The benchmark results
        :rtype: [[run_time, run_type_name, return_value], ...]
        """
        image, shift_row, shift_col = self._parse_arguments(image, shift_row, shift_col)
        return super().benchmark(image, shift_row, shift_col, magnification_row, magnification_col)

    def _run_opencl(self, image, shift_row, shift_col, float magnification_row, float magnification_col) -> np.ndarray:
        code = self._get_cl_code("_le_interpolation_nearest_neighbor_.cl")

        assert image.dtype == np.float32, "Image must be of type np.float32"
        assert image.ndim == 3, "Image must be 3D (sequence of 2D images)"
        assert shift_row.dtype == np.float32, "Shift row must be of type np.float32"
        assert shift_row.ndim == 1, "Shift row must be 1D"
        assert shift_row.shape[0] == image.shape[0], "Shift row must have the same length as the number of frames"
        assert shift_col.dtype == np.float32, "Shift col must be of type np.float32"
        assert shift_col.ndim == 1, "Shift col must be 1D"
        assert shift_col.shape[0] == image.shape[0], "Shift col must have the same length as the number of frames"

        cdef int nFrames = image.shape[0]
        cdef int rowsM = <int>(image.shape[1] * magnification_row)
        cdef int colsM = <int>(image.shape[2] * magnification_col)

        image_in = cl_array.to_device(cl_queue, image)
        shift_col_in = cl_array.to_device(cl_queue, shift_col)
        shift_row_in = cl_array.to_device(cl_queue, shift_row)
        image_out = cl_array.zeros(cl_queue, (nFrames, rowsM, colsM), dtype=np.float32)

        # Create the program
        prg = cl.Program(cl_ctx, code).build()

        # Run the kernel
        prg.shiftAndMagnify(
            cl_queue,
            image_out.shape,
            None,
            image_in.data,
            image_out.data,
            shift_col_in.data,
            shift_row_in.data,
            np.float32(magnification_row),
            np.float32(magnification_col),
        )

        # Wait for queue to finish
        cl_queue.finish()

        return image_out.get()

    def _run_unthreaded(self, float[:,:,:] image, float[:] shift_row, float[:] shift_col, float magnification_row, float magnification_col) -> np.ndarray:
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]
        cdef int rowsM = <int>(rows * magnification_row)
        cdef int colsM = <int>(cols * magnification_col)

        image_out = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image

        cdef int f, i, j
        cdef float row, col

        with nogil:
            for f in range(nFrames):
                for j in range(colsM):
                    col = j / magnification_col - shift_col[f]
                    for i in range(rowsM):
                        row = i / magnification_row - shift_row[f]
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out

    def _run_threaded(self, float[:,:,:] image, float[:] shift_row, float[:] shift_col, float magnification_row, float magnification_col) -> np.ndarray:
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]
        cdef int rowsM = <int>(rows * magnification_row)
        cdef int colsM = <int>(cols * magnification_col)

        image_out = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image

        cdef int f, i, j
        cdef float row, col

        with nogil:
            for f in range(nFrames):
                for j in prange(colsM):
                    col = j / magnification_col - shift_col[f]
                    for i in range(rowsM):
                        row = i / magnification_row - shift_row[f]
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out

    def _run_threaded_static(self, float[:,:,:] image, float[:] shift_row, float[:] shift_col, float magnification_row, float magnification_col) -> np.ndarray:
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]
        cdef int rowsM = <int>(rows * magnification_row)
        cdef int colsM = <int>(cols * magnification_col)

        image_out = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image

        cdef int f, i, j
        cdef float row, col

        with nogil:
            for f in range(nFrames):
                for j in prange(colsM, schedule="static"):
                    col = j / magnification_col - shift_col[f]
                    for i in range(rowsM):
                        row = i / magnification_row - shift_row[f]
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out


    def _run_threaded_dynamic(self, float[:,:,:] image, float[:] shift_row, float[:] shift_col, float magnification_row, float magnification_col) -> np.ndarray:
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]
        cdef int rowsM = <int>(rows * magnification_row)
        cdef int colsM = <int>(cols * magnification_col)

        image_out = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image

        cdef int f, i, j
        cdef float row, col

        with nogil:
            for f in range(nFrames):
                for j in prange(colsM, schedule="dynamic"):
                    col = j / magnification_col - shift_col[f]
                    for i in range(rowsM):
                        row = i / magnification_row - shift_row[f]
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out


    def _run_threaded_guided(self, float[:,:,:] image, float[:] shift_row, float[:] shift_col, float magnification_row, float magnification_col) -> np.ndarray:
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]
        cdef int rowsM = <int>(rows * magnification_row)
        cdef int colsM = <int>(cols * magnification_col)

        image_out = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image

        cdef int f, i, j
        cdef float row, col

        with nogil:
            for f in range(nFrames):
                for j in prange(colsM, schedule="guided"):
                    col = j / magnification_col - shift_col[f]
                    for i in range(rowsM):
                        row = i / magnification_row - shift_row[f]
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out

    def _run_python(self, image, shift_row, shift_col, magnification_row, magnification_col) -> np.ndarray:
        image_out = _py_nearest_neighbor(image, shift_row, shift_col, magnification_row, magnification_col)
        return image_out

    def _run_njit(
        self, 
        image=np.zeros((1,10,10),dtype=np.float32), 
        shift_row=np.zeros((1,),dtype=np.float32), 
        shift_col=np.zeros((1,),dtype=np.float32), 
        magnification_row=1, magnification_col=1) -> np.ndarray:
        image_out = _njit_nearest_neighbor(image, shift_row, shift_col, magnification_row, magnification_col)
        return image_out
