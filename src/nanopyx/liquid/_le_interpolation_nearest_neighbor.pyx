# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np

cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport cos, sin

from .__interpolation_tools__ import check_image, value2array
from .__liquid_engine__ import LiquidEngine
from .__opencl__ import cl, cl_array, cl_ctx, cl_queue
from ._le_interpolation_nearest_neighbor_ import \
    njit_shift_magnify as _njit_shift_magnify
from ._le_interpolation_nearest_neighbor_ import \
    njit_shift_scale_rotate as _njit_shift_magnify_rotate
from ._le_interpolation_nearest_neighbor_ import \
    shift_magnify as _py_shift_magnify
from ._le_interpolation_nearest_neighbor_ import \
    shift_scale_rotate as _py_shift_magnify_rotate


cdef extern from "_c_interpolation_nearest_neighbor.h":
    float _c_interpolate(float *image, float row, float col, int rows, int cols) nogil


class ShiftAndMagnify(LiquidEngine):
    """
    Shift and Magnify using the NanoPyx Liquid Engine
    """

    _has_opencl = True
    _has_threaded = True
    _has_threaded_static = True
    _has_threaded_dynamic = True
    _has_threaded_guided = True
    _has_unthreaded = True
    _has_python = True
    _has_njit = True

    def run(self, image: np.ndarray, shift_row: np.ndarray | int | float, shift_col: np.ndarray | int | float, float magnification_row, float magnification_col) -> np.ndarray:
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
        image = check_image(image)
        shift_row = value2array(shift_row, image.shape[0])
        shift_col = value2array(shift_col, image.shape[0])
        return self._run(image, shift_row, shift_col, magnification_row, magnification_col)

    def benchmark(self, image: np.ndarray, shift_row: np.ndarray | int | float, shift_col: np.ndarray | int | float, float magnification_row, float magnification_col):
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
        image = check_image(image)
        shift_row = value2array(shift_row, image.shape[0])
        shift_col = value2array(shift_col, image.shape[0])
        return super().benchmark(image, shift_row, shift_col, magnification_row, magnification_col)

    def _run_opencl(self, image, shift_row, shift_col, float magnification_row, float magnification_col) -> np.ndarray:
        code = self._get_cl_code("_le_interpolation_nearest_neighbor_.cl")

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
        image_out = _py_shift_magnify(image, shift_row, shift_col, magnification_row, magnification_col)
        return image_out

    def _run_njit(
        self,
        image=np.zeros((1,10,10),dtype=np.float32),
        shift_row=np.zeros((1,),dtype=np.float32),
        shift_col=np.zeros((1,),dtype=np.float32),
        magnification_row=1, magnification_col=1) -> np.ndarray:
        image_out = _njit_shift_magnify(image, shift_row, shift_col, magnification_row, magnification_col)
        return image_out

class ShiftScaleRotate(LiquidEngine):
    """
    Shift, Scale and Rotate (affine transform) using the NanoPyx Liquid Engine
    """

    _has_opencl = True
    _has_threaded = True
    _has_threaded_static = True
    _has_threaded_dynamic = True
    _has_threaded_guided = True
    _has_unthreaded = True
    _has_python = True
    _has_njit = True

    def run(self, image, shift_row, shift_col, float scale_row, float scale_col, float angle) -> np.ndarray:
        """
        Shift and scale an image using nearest neighbor interpolation
        :param image: The image to shift and magnify
        :type image: np.ndarray
        :param shift_row: The number of rows to shift the image
        :type shift_row: int or float or np.ndarray
        :param shift_col: The number of columns to shift the image
        :type shift_col: int or float or np.ndarray
        :param scale_row: The scale factor for the rows
        :type scale_row: float
        :param scale_col: The scale factor for the columns
        :type scale_col: float
        :param angle: Angle of rotation in radians. Positive is counter clockwise
        :type angle: float
        :return: The shifted, magnified and rotated image
        """
        image = check_image(image)
        shift_row = value2array(shift_row, image.shape[0])
        shift_col = value2array(shift_col, image.shape[0])
        return self._run(image, shift_row, shift_col, scale_row, scale_col, angle)

    def benchmark(self, image, shift_row, shift_col, float scale_row, float scale_col, float angle):
        """
        Benchmark the ShiftMagnifyScale run function in multiple run types
        :param image: The image to shift, scale and rotate
        :type image: np.ndarray
        :param shift_row: The number of rows to shift the image
        :type shift_row: int or float or np.ndarray
        :param shift_col: The number of columns to shift the image
        :type shift_col: int or float or np.ndarray
        :param scale_row: The scale factor for the rows
        :type scale_row: float
        :param scale_col: The scale factor for the columns
        :type scale_col: float
        :param angle: Angle of rotation in radians. Positive is counter clockwise
        :type angle: float
        :return: The benchmark results
        :rtype: [[run_time, run_type_name, return_value], ...]
        """
        image = check_image(image)
        shift_row = value2array(shift_row, image.shape[0])
        shift_col = value2array(shift_col, image.shape[0])
        return super().benchmark(image, shift_row, shift_col, scale_row, scale_col, angle)

    def _run_opencl(self, image, shift_row, shift_col, float scale_row, float scale_col, float angle) -> np.ndarray:
        code = self._get_cl_code("_le_interpolation_nearest_neighbor_.cl")

        cdef int nFrames = image.shape[0]
        cdef int rowsM = image.shape[1]
        cdef int colsM = image.shape[2]

        image_in = cl_array.to_device(cl_queue, image)
        shift_col_in = cl_array.to_device(cl_queue, shift_col)
        shift_row_in = cl_array.to_device(cl_queue, shift_row)
        image_out = cl_array.zeros(cl_queue, (nFrames, rowsM, colsM), dtype=np.float32)

        # Create the program
        prg = cl.Program(cl_ctx, code).build()

        # Run the kernel
        prg.ShiftScaleRotate(
            cl_queue,
            image_out.shape,
            None,
            image_in.data,
            image_out.data,
            shift_col_in.data,
            shift_row_in.data,
            np.float32(scale_row),
            np.float32(scale_col),
            np.float32(angle)
        )

        # Wait for queue to finish
        cl_queue.finish()

        return image_out.get()

    def _run_unthreaded(self, float[:,:,:] image, float[:] shift_row, float[:] shift_col, float scale_row, float scale_col, float angle) -> np.ndarray:
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]

        image_out = np.zeros((nFrames, rows, cols), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image

        cdef int f, i, j
        cdef float row, col

        cdef float center_col = cols/2
        cdef float center_row = rows/2

        cdef float center_rowM = (rows * scale_row) / 2
        cdef float center_colM = (cols * scale_col) / 2

        cdef float a,b,c,d
        a = cos(angle)/scale_col
        b = -sin(angle)
        c = sin(angle)
        d = cos(angle)/scale_row

        with nogil:
            for f in range(nFrames):
                for j in range(cols):
                    for i in range(rows):
                        col = (a*(j-center_colM)+b*(i-center_rowM)) - shift_col[f] + center_col
                        row = (c*(j-center_colM)+d*(i-center_rowM)) - shift_row[f] + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out

    def _run_threaded(self, float[:,:,:] image, float[:] shift_row, float[:] shift_col, float scale_row, float scale_col, float angle) -> np.ndarray:
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]

        image_out = np.zeros((nFrames, rows, cols), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image

        cdef int f, i, j
        cdef float row, col

        cdef float center_col = cols/2
        cdef float center_row = rows/2

        cdef float center_rowM = (rows * scale_row) / 2
        cdef float center_colM = (cols * scale_col) / 2

        cdef float a,b,c,d
        a = cos(angle)/scale_col
        b = -sin(angle)
        c = sin(angle)
        d = cos(angle)/scale_row

        with nogil:
            for f in range(nFrames):
                for j in prange(cols):
                    for i in range(rows):
                        col = (a*(j-center_colM)+b*(i-center_rowM)) - shift_col[f] + center_col
                        row = (c*(j-center_colM)+d*(i-center_rowM)) - shift_row[f] + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out

    def _run_threaded_static(self, float[:,:,:] image, float[:] shift_row, float[:] shift_col, float scale_row, float scale_col, float angle) -> np.ndarray:
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]

        image_out = np.zeros((nFrames, rows, cols), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image

        cdef int f, i, j
        cdef float row, col

        cdef float center_col = cols/2
        cdef float center_row = rows/2

        cdef float center_rowM = (rows * scale_row) / 2
        cdef float center_colM = (cols * scale_col) / 2

        cdef float a,b,c,d
        a = cos(angle)/scale_col
        b = -sin(angle)
        c = sin(angle)
        d = cos(angle)/scale_row

        with nogil:
            for f in range(nFrames):
                for j in prange(cols, schedule="static"):
                    for i in range(rows):
                        col = (a*(j-center_colM)+b*(i-center_rowM)) - shift_col[f] + center_col
                        row = (c*(j-center_colM)+d*(i-center_rowM)) - shift_row[f] + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out


    def _run_threaded_dynamic(self, float[:,:,:] image, float[:] shift_row, float[:] shift_col, float scale_row, float scale_col, float angle) -> np.ndarray:
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]

        image_out = np.zeros((nFrames, rows, cols), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image

        cdef int f, i, j
        cdef float row, col

        cdef float center_col = cols/2
        cdef float center_row = rows/2

        cdef float center_rowM = (rows * scale_row) / 2
        cdef float center_colM = (cols * scale_col) / 2

        cdef float a,b,c,d
        a = cos(angle)/scale_col
        b = -sin(angle)
        c = sin(angle)
        d = cos(angle)/scale_row

        with nogil:
            for f in range(nFrames):
                for j in prange(cols, schedule="dynamic"):
                    for i in range(rows):
                        col = (a*(j-center_colM)+b*(i-center_rowM)) - shift_col[f] + center_col
                        row = (c*(j-center_colM)+d*(i-center_rowM)) - shift_row[f] + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out


    def _run_threaded_guided(self, float[:,:,:] image, float[:] shift_row, float[:] shift_col, float scale_row, float scale_col, float angle) -> np.ndarray:
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]

        image_out = np.zeros((nFrames, rows, cols), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image

        cdef int f, i, j
        cdef float row, col

        cdef float center_col = cols/2
        cdef float center_row = rows/2

        cdef float center_rowM = (rows * scale_row) / 2
        cdef float center_colM = (cols * scale_col) / 2

        cdef float a,b,c,d
        a = cos(angle)/scale_col
        b = -sin(angle)
        c = sin(angle)
        d = cos(angle)/scale_row

        with nogil:
            for f in range(nFrames):
                for j in prange(cols, schedule="guided"):
                    for i in range(rows):
                        col = (a*(j-center_colM)+b*(i-center_rowM)) - shift_col[f] + center_col
                        row = (c*(j-center_colM)+d*(i-center_rowM)) - shift_row[f] + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out

    def _run_python(self, image, shift_row, shift_col, scale_row, scale_col, angle) -> np.ndarray:
        image_out = _py_shift_magnify_rotate(image, shift_row, shift_col, scale_row, scale_col, angle)
        return image_out

    def _run_njit(
        self,
        image=np.zeros((1,10,10),dtype=np.float32),
        shift_row=np.zeros((1,),dtype=np.float32),
        shift_col=np.zeros((1,),dtype=np.float32),
        scale_row=1, scale_col=1, angle=0) -> np.ndarray:
        image_out = _njit_shift_magnify_rotate(image, shift_row, shift_col, scale_row, scale_col, angle)
        return image_out
