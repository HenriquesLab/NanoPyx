# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np

cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport cos, sin

from .__interpolation_tools__ import check_image, value2array
from .__liquid_engine__ import LiquidEngine
from .__opencl__ import cl, cl_array


cdef extern from "_c_interpolation_bicubic.h":
    float _c_interpolate(float *image, float row, float col, int rows, int cols) nogil


class ShiftAndMagnify(LiquidEngine):
    """
    Shift and Magnify using the NanoPyx Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "ShiftMagnify_BC"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, 
                        opencl_=True, unthreaded_=True, threaded_=True, threaded_static_=True, 
                        threaded_dynamic_=True, threaded_guided_=True)


    # tag-copy: _le_interpolation_nearest_neighbor.ShiftAndMagnify.run; replace("Nearest-Neighbor", "Bicubic")
    def run(self, image, shift_row, shift_col, float magnification_row, float magnification_col, run_type=None) -> np.ndarray:
        """
        Shift and magnify an image using Bicubic interpolation
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

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftAndMagnify.benchmark
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

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_opencl; replace("nearest_neighbor", "bicubic")
    def _run_opencl(self, image, shift_row, shift_col, float magnification_row, float magnification_col, dict device) -> np.ndarray:

        # QUEUE AND CONTEXT
        cl_ctx = cl.Context([device['device']])
        cl_queue = cl.CommandQueue(cl_ctx)

        # Swap row and columns because opencl is strange and stores the
        # array in a buffer in fortran ordering despite the original
        # numpy array being in C order.
        image = np.ascontiguousarray(np.swapaxes(image, 1, 2), dtype=np.float32)

        code = self._get_cl_code("_le_interpolation_bicubic_.cl", device['DP'])

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

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_unthreaded
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

class ShiftScaleRotate(LiquidEngine):
    """
    Shift, Scale and Rotate (affine transform) using the NanoPyx Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "ShiftScaleRotate_BC"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing,
                        opencl_=True, unthreaded_=True, threaded_=True, threaded_static_=True, 
                        threaded_dynamic_=True, threaded_guided_=True)


    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate.run; replace("Nearest-Neighbor", "Bicubic")
    def run(self, image, shift_row, shift_col, float scale_row, float scale_col, float angle, run_type=None) -> np.ndarray:
        """
        Shift and scale an image using Bicubic interpolation
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

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate.benchmark
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

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_opencl; replace("nearest_neighbor", "bicubic")
    def _run_opencl(self, image, shift_row, shift_col, float scale_row, float scale_col, float angle, dict device) -> np.ndarray:

        # QUEUE AND CONTEXT
        cl_ctx = cl.Context([device['device']])
        cl_queue = cl.CommandQueue(cl_ctx)

        # Swap row and columns because opencl is strange and stores the
        # array in a buffer in fortran ordering despite the original
        # numpy array being in C order.
        image = np.ascontiguousarray(np.swapaxes(image, 1, 2), dtype=np.float32)

        code = self._get_cl_code("_le_interpolation_bicubic_.cl", device['DP'])

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
        prg.shiftScaleRotate(
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

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_unthreaded
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

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_unthreaded; replace("_run_unthreaded", "_run_threaded"); replace("range(colsM)", "prange(colsM)")
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
                for j in range(cols):
                    for i in range(rows):
                        col = (a*(j-center_col-shift_col[f])+b*(i-center_row-shift_row[f])) + center_col
                        row = (c*(j-center_col-shift_col[f])+d*(i-center_row-shift_row[f])) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_unthreaded; replace("_run_unthreaded", "_run_threaded_static"); replace("range(colsM)", 'prange(colsM, schedule="static")')
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
                for j in range(cols):
                    for i in range(rows):
                        col = (a*(j-center_col-shift_col[f])+b*(i-center_row-shift_row[f])) + center_col
                        row = (c*(j-center_col-shift_col[f])+d*(i-center_row-shift_row[f])) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_unthreaded; replace("_run_unthreaded", "_run_threaded_dynamic"); replace("range(colsM)", 'prange(colsM, schedule="dynamic")')
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
                for j in range(cols):
                    for i in range(rows):
                        col = (a*(j-center_col-shift_col[f])+b*(i-center_row-shift_row[f])) + center_col
                        row = (c*(j-center_col-shift_col[f])+d*(i-center_row-shift_row[f])) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_unthreaded; replace("_run_unthreaded", "_run_threaded_guided"); replace("range(colsM)", 'prange(colsM, schedule="guided")')
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
                for j in range(cols):
                    for i in range(rows):
                        col = (a*(j-center_col-shift_col[f])+b*(i-center_row-shift_row[f])) + center_col
                        row = (c*(j-center_col-shift_col[f])+d*(i-center_row-shift_row[f])) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end
