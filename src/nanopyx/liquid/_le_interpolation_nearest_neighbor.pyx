# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np

cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport cos, sin, pi, hypot, exp, log

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

    def __init__(self):
        super().__init__()

    # tag-start: _le_interpolation_nearest_neighbor.ShiftAndMagnify.run
    def run(self, image, shift_row, shift_col, float magnification_row, float magnification_col, run_type=None) -> np.ndarray:
        """
        Shift and magnify an image using Nearest-Neighbor interpolation
        :param image: The image to shift and magnify
        :type image: np.ndarray or memoryview
        :param shift_row: The number of rows to shift the image
        :type shift_row: int or float or np.ndarray
        :param shift_col: The number of columns to shift the image
        :type shift_col: int or float or np.ndarray
        :param magnification_row: The magnification factor for the rows
        :type magnification_row: float
        :param magnification_col: The magnification factor for the columns
        :type magnification_col: float
        :return: The shifted and magnified image
        """
        image = check_image(image)
        shift_row = value2array(shift_row, image.shape[0])
        shift_col = value2array(shift_col, image.shape[0])
        return self._run(image, shift_row, shift_col, magnification_row, magnification_col, run_type=run_type)
    # tag-end

    # tag-start: _le_interpolation_nearest_neighbor.ShiftAndMagnify.benchmark
    def benchmark(self, image, shift_row, shift_col, float magnification_row, float magnification_col):
        """
        Benchmark the ShiftAndMagnify run function in multiple run types
        :param image: The image to shift and magnify
        :type image: np.ndarray or memoryview
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
    # tag-end

    # tag-start: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_opencl
    def _run_opencl(self, image, shift_row, shift_col, float magnification_row, float magnification_col) -> np.ndarray:
        # Swap row and columns because opencl is strange and stores the
        # array in a buffer in fortran ordering despite the original
        # numpy array being in C order.
        image = np.ascontiguousarray(np.swapaxes(image, 1, 2), dtype=np.float32)

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

        # Swap rows and columns back
        return np.ascontiguousarray(np.swapaxes(image_out.get(), 1, 2), dtype=np.float32)
    # tag-end

    # tag-start: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_unthreaded
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
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_unthreaded; replace("_run_unthreaded", "_run_threaded"); replace("range(colsM)", "prange(colsM)")
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
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_unthreaded; replace("_run_unthreaded", "_run_threaded_static"); replace("range(colsM)", 'prange(colsM, schedule="static")')
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
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_unthreaded; replace("_run_unthreaded", "_run_threaded_dynamic"); replace("range(colsM)", 'prange(colsM, schedule="dynamic")')
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
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_unthreaded; replace("_run_unthreaded", "_run_threaded_guided"); replace("range(colsM)", 'prange(colsM, schedule="guided")')
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
    # tag-end

    # tag-start: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_python
    def _run_python(self, image, shift_row, shift_col, magnification_row, magnification_col) -> np.ndarray:
        image_out = _py_shift_magnify(image, shift_row, shift_col, magnification_row, magnification_col)
        return image_out
    # tag-end

    # tag-start: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_njit
    def _run_njit(
        self,
        image=np.zeros((1,10,10),dtype=np.float32),
        shift_row=np.zeros((1,),dtype=np.float32),
        shift_col=np.zeros((1,),dtype=np.float32),
        magnification_row=1, magnification_col=1) -> np.ndarray:
        image_out = _njit_shift_magnify(image, shift_row, shift_col, magnification_row, magnification_col)
        return image_out
    # tag-end

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

    def __init__(self):
        super().__init__()
        
    # tag-start: _le_interpolation_nearest_neighbor.ShiftScaleRotate.run
    def run(self, image, shift_row, shift_col, float scale_row, float scale_col, float angle, run_type=None) -> np.ndarray:
        """
        Shift and scale an image using Nearest-Neighbor interpolation
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
        return self._run(image, shift_row, shift_col, scale_row, scale_col, angle, run_type=run_type)
    # tag-end


    # tag-start: _le_interpolation_nearest_neighbor.ShiftScaleRotate.benchmark
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
    # tag-end


    # tag-start: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_opencl
    def _run_opencl(self, image, shift_row, shift_col, float scale_row, float scale_col, float angle) -> np.ndarray:

        # Swap row and columns because opencl is strange and stores the
        # array in a buffer in fortran ordering despite the original
        # numpy array being in C order.
        image = np.ascontiguousarray(np.swapaxes(image, 1, 2), dtype=np.float32)

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

        # Swap rows and columns back
        return np.ascontiguousarray(np.swapaxes(image_out.get(), 1, 2), dtype=np.float32)
    # tag-end

    # tag-start: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_unthreaded
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

        # cdef float center_rowM = (rows * scale_row) / 2
        # cdef float center_colM = (cols * scale_col) / 2

        cdef float a,b,c,d
        a = cos(angle)/scale_col
        b = -sin(angle)/scale_col
        c = sin(angle)/scale_row
        d = cos(angle)/scale_row

        with nogil:
            for f in range(nFrames):
                for j in range(cols):
                    for i in range(rows):
                        col = (a*(j-center_col-shift_col[f])+b*(i-center_row-shift_row[f])) + center_col
                        row = (c*(j-center_col-shift_col[f])+d*(i-center_row-shift_row[f])) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_unthreaded; replace("_run_unthreaded", "_run_threaded"); replace("range(cols)", "prange(cols)")
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

        # cdef float center_rowM = (rows * scale_row) / 2
        # cdef float center_colM = (cols * scale_col) / 2

        cdef float a,b,c,d
        a = cos(angle)/scale_col
        b = -sin(angle)/scale_col
        c = sin(angle)/scale_row
        d = cos(angle)/scale_row

        with nogil:
            for f in range(nFrames):
                for j in prange(cols):
                    for i in range(rows):
                        col = (a*(j-center_col-shift_col[f])+b*(i-center_row-shift_row[f])) + center_col
                        row = (c*(j-center_col-shift_col[f])+d*(i-center_row-shift_row[f])) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_unthreaded; replace("_run_unthreaded", "_run_threaded_static"); replace("range(cols)", "prange(cols, schedule='static')")
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

        # cdef float center_rowM = (rows * scale_row) / 2
        # cdef float center_colM = (cols * scale_col) / 2

        cdef float a,b,c,d
        a = cos(angle)/scale_col
        b = -sin(angle)/scale_col
        c = sin(angle)/scale_row
        d = cos(angle)/scale_row

        with nogil:
            for f in range(nFrames):
                for j in prange(cols, schedule='static'):
                    for i in range(rows):
                        col = (a*(j-center_col-shift_col[f])+b*(i-center_row-shift_row[f])) + center_col
                        row = (c*(j-center_col-shift_col[f])+d*(i-center_row-shift_row[f])) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_unthreaded; replace("_run_unthreaded", "_run_threaded_dynamic"); replace("range(cols)", "prange(cols, schedule='dynamic')")
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

        # cdef float center_rowM = (rows * scale_row) / 2
        # cdef float center_colM = (cols * scale_col) / 2

        cdef float a,b,c,d
        a = cos(angle)/scale_col
        b = -sin(angle)/scale_col
        c = sin(angle)/scale_row
        d = cos(angle)/scale_row

        with nogil:
            for f in range(nFrames):
                for j in prange(cols, schedule='dynamic'):
                    for i in range(rows):
                        col = (a*(j-center_col-shift_col[f])+b*(i-center_row-shift_row[f])) + center_col
                        row = (c*(j-center_col-shift_col[f])+d*(i-center_row-shift_row[f])) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_unthreaded; replace("_run_unthreaded", "_run_threaded_guided"); replace("range(cols)", "prange(cols, schedule='guided')")
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

        # cdef float center_rowM = (rows * scale_row) / 2
        # cdef float center_colM = (cols * scale_col) / 2

        cdef float a,b,c,d
        a = cos(angle)/scale_col
        b = -sin(angle)/scale_col
        c = sin(angle)/scale_row
        d = cos(angle)/scale_row

        with nogil:
            for f in range(nFrames):
                for j in prange(cols, schedule='guided'):
                    for i in range(rows):
                        col = (a*(j-center_col-shift_col[f])+b*(i-center_row-shift_row[f])) + center_col
                        row = (c*(j-center_col-shift_col[f])+d*(i-center_row-shift_row[f])) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end

    # tag-start: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_python
    def _run_python(self, image, shift_row, shift_col, scale_row, scale_col, angle) -> np.ndarray:
        image_out = _py_shift_magnify_rotate(image, shift_row, shift_col, scale_row, scale_col, angle)
        return image_out
    # tag-end


    # tag-start: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_njit
    def _run_njit(
        self,
        image=np.zeros((1,10,10),dtype=np.float32),
        shift_row=np.zeros((1,),dtype=np.float32),
        shift_col=np.zeros((1,),dtype=np.float32),
        scale_row=1, scale_col=1, angle=0) -> np.ndarray:
        image_out = _njit_shift_magnify_rotate(image, shift_row, shift_col, scale_row, scale_col, angle)
        return image_out
    # tag-end



class PolarTransform(LiquidEngine):
    """
    Polar Transformations using the NanoPyx Liquid Engine
    """

    _has_opencl = False
    _has_threaded = True
    _has_threaded_static = True
    _has_threaded_dynamic = True
    _has_threaded_guided = True
    _has_unthreaded = True
    _has_python = False
    _has_njit = False

    def __init__(self):
        super().__init__()

    # tag-start: _le_interpolation_nearest_neighbor.PolarTransform.run
    def run(self, image, tuple out_shape, str scale, run_type=None) -> np.ndarray:
        """
        Polar Transform an image using Nearest-Neighbor interpolation
        :param image: The image to transform
        :type image: np.ndarray or memoryview
        :param out_shape: Desired shape for the output image
        :type out_shape: tuple (n_row, n_col)
        :param scale: Linear or Log transform
        :type scale: str, either 'log' or 'linear'
        :return: The tranformed image in polar coordinates
        """
        image = check_image(image)
        nrow, ncol = out_shape
        if scale not in ['linear', 'log']:
            scale = 'linear'
        return self._run(image, nrow, ncol, scale, run_type=run_type)
    # tag-end

    # tag-start: _le_interpolation_nearest_neighbor.PolarTransform.benchmark
    def benchmark(self, image, tuple out_shape, str scale):
        """
        Benchmark the PolarTransform run function in multiple run types
        :param image: The image to transform
        :type image: np.ndarray or memoryview
        :param out_shape: Desired shape for the output image
        :type out_shape: tuple (n_row, n_col)
        :param scale: Linear or Log transform
        :type scale: str, either 'log' or 'linear'
        :return: The benchmark results
        :rtype: [[run_time, run_type_name, return_value], ...]
        """
        image = check_image(image)
        nrow, ncol = out_shape
        if scale not in ['linear', 'log']:
            scale = 'linear'
        return super().benchmark(image, nrow, ncol, scale)
    # tag-end

    # tag-start: _le_interpolation_nearest_neighbor.PolarTransform._run_opencl
    def _run_opencl(self, float[:,:,:] image, int nrow, int ncol, str scale):
        return 0
    # tag-end

    # tag-start: _le_interpolation_nearest_neighbor.PolarTransform._run_unthreaded
    def _run_unthreaded(self, float[:,:,:] image, int nrow, int ncol, str scale):
        
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]

        cdef float c_rows = rows / 2
        cdef float c_cols = cols / 2

        # angle on rows, radius on cols
        image_out = np.zeros((nFrames, nrow, ncol), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image
        
        # max_angle = 2*pi
        cdef float max_radius = hypot(c_rows, c_cols)

        cdef int f,i,j
        cdef float angle, radius, col, row

        with nogil:
            for f in range(nFrames):
                for i in range(ncol):
                    for j in range(nrow):
                        angle = j * 2 * pi  / (nrow-1)
                        if scale=='log':
                            radius = exp(i*log(max_radius)/(ncol-1))
                        else:
                            radius = i * max_radius / (ncol-1)
                        col = radius * cos(angle) + c_cols
                        row = radius * sin(angle) + c_rows
                        _image_out[f, j, i] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.PolarTransform._run_unthreaded; replace("_run_unthreaded", "_run_threaded"); replace("range(ncol)", "prange(ncol)")
    def _run_threaded(self, float[:,:,:] image, int nrow, int ncol, str scale):
        
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]

        cdef float c_rows = rows / 2
        cdef float c_cols = cols / 2

        # angle on rows, radius on cols
        image_out = np.zeros((nFrames, nrow, ncol), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image
        
        # max_angle = 2*pi
        cdef float max_radius = hypot(c_rows, c_cols)

        cdef int f,i,j
        cdef float angle, radius, col, row

        with nogil:
            for f in range(nFrames):
                for i in prange(ncol):
                    for j in range(nrow):
                        angle = j * 2 * pi  / (nrow-1)
                        if scale=='log':
                            radius = exp(i*log(max_radius)/(ncol-1))
                        else:
                            radius = i * max_radius / (ncol-1)
                        col = radius * cos(angle) + c_cols
                        row = radius * sin(angle) + c_rows
                        _image_out[f, j, i] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.PolarTransform._run_unthreaded; replace("_run_unthreaded", "_run_threaded_static"); replace("range(ncol)", 'prange(ncol, schedule="static")')  
    def _run_threaded_static(self, float[:,:,:] image, int nrow, int ncol, str scale):
        
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]

        cdef float c_rows = rows / 2
        cdef float c_cols = cols / 2

        # angle on rows, radius on cols
        image_out = np.zeros((nFrames, nrow, ncol), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image
        
        # max_angle = 2*pi
        cdef float max_radius = hypot(c_rows, c_cols)

        cdef int f,i,j
        cdef float angle, radius, col, row

        with nogil:
            for f in range(nFrames):
                for i in prange(ncol, schedule="static"):
                    for j in range(nrow):
                        angle = j * 2 * pi  / (nrow-1)
                        if scale=='log':
                            radius = exp(i*log(max_radius)/(ncol-1))
                        else:
                            radius = i * max_radius / (ncol-1)
                        col = radius * cos(angle) + c_cols
                        row = radius * sin(angle) + c_rows
                        _image_out[f, j, i] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.PolarTransform._run_unthreaded; replace("_run_unthreaded", "_run_threaded_dynamic"); replace("range(ncol)", 'prange(ncol, schedule="dynamic")')
    def _run_threaded_dynamic(self, float[:,:,:] image, int nrow, int ncol, str scale):
        
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]

        cdef float c_rows = rows / 2
        cdef float c_cols = cols / 2

        # angle on rows, radius on cols
        image_out = np.zeros((nFrames, nrow, ncol), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image
        
        # max_angle = 2*pi
        cdef float max_radius = hypot(c_rows, c_cols)

        cdef int f,i,j
        cdef float angle, radius, col, row

        with nogil:
            for f in range(nFrames):
                for i in prange(ncol, schedule="dynamic"):
                    for j in range(nrow):
                        angle = j * 2 * pi  / (nrow-1)
                        if scale=='log':
                            radius = exp(i*log(max_radius)/(ncol-1))
                        else:
                            radius = i * max_radius / (ncol-1)
                        col = radius * cos(angle) + c_cols
                        row = radius * sin(angle) + c_rows
                        _image_out[f, j, i] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.PolarTransform._run_unthreaded; replace("_run_unthreaded", "_run_threaded_guided"); replace("range(ncol)", 'prange(ncol, schedule="guided")')
    def _run_threaded_guided(self, float[:,:,:] image, int nrow, int ncol, str scale):
        
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]

        cdef float c_rows = rows / 2
        cdef float c_cols = cols / 2

        # angle on rows, radius on cols
        image_out = np.zeros((nFrames, nrow, ncol), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image
        
        # max_angle = 2*pi
        cdef float max_radius = hypot(c_rows, c_cols)

        cdef int f,i,j
        cdef float angle, radius, col, row

        with nogil:
            for f in range(nFrames):
                for i in prange(ncol, schedule="guided"):
                    for j in range(nrow):
                        angle = j * 2 * pi  / (nrow-1)
                        if scale=='log':
                            radius = exp(i*log(max_radius)/(ncol-1))
                        else:
                            radius = i * max_radius / (ncol-1)
                        col = radius * cos(angle) + c_cols
                        row = radius * sin(angle) + c_rows
                        _image_out[f, j, i] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end

    # tag-start: _le_interpolation_nearest_neighbor.PolarTransform._run_python
    def _run_python(self, image, nrow, ncol, scale):
        return 0
    # tag-end

    # tag-start: _le_interpolation_nearest_neighbor.PolarTransform._run_njit
    def _run_njit(self, image, nrow, ncol, scale):
        return 0
    # tag-end
