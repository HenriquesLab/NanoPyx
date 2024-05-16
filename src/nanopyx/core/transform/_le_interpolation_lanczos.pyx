# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange
from libc.math cimport cos, sin, pi, hypot, exp, log

from .__interpolation_tools__ import check_image, value2array
from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array, _fastest_device


cdef extern from "_c_interpolation_lanczos.h":
    float _c_interpolate(float *image, float row, float col, int rows, int cols) nogil


class ShiftAndMagnify(LiquidEngine):
    """
    Shift and Magnify using the NanoPyx Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False, verbose=True):
        self._designation = "ShiftMagnify_lanczos"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, verbose=verbose)

    def run(self, image, shift_row, shift_col, float magnification_row, float magnification_col, run_type=None) -> np.ndarray:
        """
        Shift and magnify an image using lanczos interpolation
        :param image: The image to shift and magnify
        :type image: np.ndarray or memoryview
        :param shift_row: The number of rows to shift the image
        :type shift_row: int or float or np.ndarray
        :param shift_col: The number of columns to shift the image«
        :type shift_col: int or float or np.ndarray
        :param magnification_row: The magnification factor for the rows
        :type magnification_row: float
        :param magnification_col: The magnification factor for the columns
        :type magnification_col: float
        :return: The shifted and magnified image
        """
        image = check_image(image)
        return self._run(image, shift_row, shift_col, magnification_row, magnification_col, run_type=run_type)

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
        return super().benchmark(image, shift_row, shift_col, magnification_row, magnification_col)

    def _run_opencl(self, image, shift_row, shift_col, float magnification_row, float magnification_col, dict device=None, int mem_div=1) -> np.ndarray:
        """
        @gpu
        """
        if device is None:
            device = _fastest_device
        # QUEUE AND CONTEXT
        cl_ctx = cl.Context([device['device']])
        dc = device['device']
        cl_queue = cl.CommandQueue(cl_ctx)
        
        output_shape = (image.shape[0], int(image.shape[1]*magnification_row), int(image.shape[2]*magnification_col))
        image_out = np.zeros(output_shape, dtype=np.float32)

        max_slices = int((dc.global_mem_size // (image_out[0,:,:].nbytes + image[0,:,:].nbytes))/mem_div)
        max_slices = self._check_max_slices(image, max_slices)

        mf = cl.mem_flags
        input_opencl = cl.Buffer(cl_ctx, mf.READ_ONLY, image[0:max_slices,:,:].nbytes)
        output_opencl = cl.Buffer(cl_ctx, mf.WRITE_ONLY, image_out[0:max_slices,:,:].nbytes)
        cl.enqueue_copy(cl_queue, input_opencl, image[0:max_slices,:,:]).wait()

        code = self._get_cl_code("_le_interpolation_lanczos_.cl", device['DP'])
        prg = cl.Program(cl_ctx, code).build()
        knl = prg.shiftAndMagnify

        for i in range(0, image.shape[0], max_slices):
            if image.shape[0] - i >= max_slices:
                n_slices = max_slices
            else:
                n_slices = image.shape[0] - i
            knl(cl_queue,
                (n_slices, int(image.shape[1]*magnification_row), int(image.shape[2]*magnification_col)), 
                None, #self.get_work_group(dc, (n_slices, image.shape[1]*magnification_row, image.shape[2]*magnification_col)), 
                input_opencl, 
                output_opencl, 
                np.float32(shift_row), 
                np.float32(shift_col), 
                np.float32(magnification_row), 
                np.float32(magnification_col)).wait() 

            cl.enqueue_copy(cl_queue, image_out[i:i+n_slices,:,:], output_opencl).wait() 
            if i+n_slices<image.shape[0]:
                cl.enqueue_copy(cl_queue, input_opencl, image[i+n_slices:i+2*n_slices,:,:]).wait() 

            cl_queue.finish()

        input_opencl.release()
        output_opencl.release()
        
        return image_out

    def _run_unthreaded(self, float[:,:,:] image, float shift_row, float shift_col, float magnification_row, float magnification_col) -> np.ndarray:
        """
        @cpu
        @threaded
        @cython
        """
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
                    col = j / magnification_col - shift_col
                    for i in range(rowsM):
                        row = i / magnification_row - shift_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out

    def _run_threaded(self, float[:,:,:] image, float shift_row, float shift_col, float magnification_row, float magnification_col) -> np.ndarray:
        """
        @cpu
        @threaded
        @cython
        """
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
                    col = j / magnification_col - shift_col
                    for i in range(rowsM):
                        row = i / magnification_row - shift_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out

    def _run_threaded_guided(self, float[:,:,:] image, float shift_row, float shift_col, float magnification_row, float magnification_col) -> np.ndarray:
        """
        @cpu
        @threaded
        @cython
        """
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
                    col = j / magnification_col - shift_col
                    for i in range(rowsM):
                        row = i / magnification_row - shift_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out

    def _run_threaded_dynamic(self, float[:,:,:] image, float shift_row, float shift_col, float magnification_row, float magnification_col) -> np.ndarray:
        """
        @cpu
        @threaded
        @cython
        """
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
                    col = j / magnification_col - shift_col
                    for i in range(rowsM):
                        row = i / magnification_row - shift_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out

    def _run_threaded_static(self, float[:,:,:] image, float shift_row, float shift_col, float magnification_row, float magnification_col) -> np.ndarray:
        """
        @cpu
        @threaded
        @cython
        """
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
                    col = j / magnification_col - shift_col
                    for i in range(rowsM):
                        row = i / magnification_row - shift_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out



class ShiftScaleRotate(LiquidEngine):
    """
    Shift, Scale and Rotate (affine transform) using the NanoPyx Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False, verbose=True):
        self._designation = "ShiftScaleRotate_lanczos"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, verbose=verbose)
        
    def run(self, image, shift_row, shift_col, float scale_row, float scale_col, float angle, run_type=None) -> np.ndarray:
        """
        Shift and scale an image using lanczos interpolation
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
        return self._run(image, shift_row, shift_col, scale_row, scale_col, angle, run_type=run_type)

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
        return super().benchmark(image, shift_row, shift_col, scale_row, scale_col, angle)

    def _run_opencl(self, image, shift_row, shift_col, float scale_row, float scale_col, float angle, dict device=None, int mem_div=1) -> np.ndarray:

        if device is None:
            device = _fastest_device

        # QUEUE AND CONTEXT
        cl_ctx = cl.Context([device['device']])
        dc = device["device"]
        cl_queue = cl.CommandQueue(cl_ctx)

        output_shape = (image.shape[0], int(image.shape[1]), int(image.shape[2]))
        image_out = np.zeros(output_shape, dtype=np.float32)

        max_slices = int((dc.global_mem_size // (image_out[0,:,:].nbytes + image[0,:,:].nbytes))/mem_div)
        max_slices = self._check_max_slices(image, max_slices)

        mf = cl.mem_flags
        input_opencl = cl.Buffer(cl_ctx, mf.READ_ONLY, self._check_max_buffer_size(image[0:max_slices,:,:].nbytes, dc, max_slices))
        output_opencl = cl.Buffer(cl_ctx, mf.WRITE_ONLY, self._check_max_buffer_size(image_out[0:max_slices,:,:].nbytes, dc, max_slices))
        cl.enqueue_copy(cl_queue, input_opencl, image[0:max_slices,:,:]).wait()

        code = self._get_cl_code("_le_interpolation_lanczos_.cl", device['DP'])
        prg = cl.Program(cl_ctx, code).build()
        knl = prg.shiftScaleRotate

        for i in range(0, image.shape[0], max_slices):
            if image.shape[0] - i >= max_slices:
                n_slices = max_slices
            else:
                n_slices = image.shape[0] - i
            knl(
                cl_queue,
                (n_slices, int(image.shape[1]), int(image.shape[2])),
                None,#self.get_work_group(dc, (n_slices, image.shape[1], image.shape[2])),
                input_opencl,
                output_opencl,
                np.float32(shift_row), 
                np.float32(shift_col),
                np.float32(scale_row),
                np.float32(scale_col),
                np.float32(angle)
            ).wait()

            cl.enqueue_copy(cl_queue, image_out[i:i+n_slices,:,:], output_opencl).wait()
            if i+n_slices<image.shape[0]:
                cl.enqueue_copy(cl_queue, input_opencl, image[i+n_slices:i+2*n_slices,:,:]).wait()

            cl_queue.finish()

        input_opencl.release()
        output_opencl.release()

        return image_out

    def _run_unthreaded(self, float[:,:,:] image, float shift_row, float shift_col, float scale_row, float scale_col, float angle) -> np.ndarray:
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
                        col = (a*(j-center_col-shift_col)+b*(i-center_row-shift_row)) + center_col
                        row = (c*(j-center_col-shift_col)+d*(i-center_row-shift_row)) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
        
    def _run_threaded(self, float[:,:,:] image, float shift_row, float shift_col, float scale_row, float scale_col, float angle) -> np.ndarray:
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
                        col = (a*(j-center_col-shift_col)+b*(i-center_row-shift_row)) + center_col
                        row = (c*(j-center_col-shift_col)+d*(i-center_row-shift_row)) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
        
    def _run_threaded_guided(self, float[:,:,:] image, float shift_row, float shift_col, float scale_row, float scale_col, float angle) -> np.ndarray:
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
                for j in prange(cols, schedule="guided"):
                    for i in range(rows):
                        col = (a*(j-center_col-shift_col)+b*(i-center_row-shift_row)) + center_col
                        row = (c*(j-center_col-shift_col)+d*(i-center_row-shift_row)) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
        
    def _run_threaded_dynamic(self, float[:,:,:] image, float shift_row, float shift_col, float scale_row, float scale_col, float angle) -> np.ndarray:
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
                for j in prange(cols, schedule="dynamic"):
                    for i in range(rows):
                        col = (a*(j-center_col-shift_col)+b*(i-center_row-shift_row)) + center_col
                        row = (c*(j-center_col-shift_col)+d*(i-center_row-shift_row)) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
        
    def _run_threaded_static(self, float[:,:,:] image, float shift_row, float shift_col, float scale_row, float scale_col, float angle) -> np.ndarray:
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
                for j in prange(cols, schedule="static"):
                    for i in range(rows):
                        col = (a*(j-center_col-shift_col)+b*(i-center_row-shift_row)) + center_col
                        row = (c*(j-center_col-shift_col)+d*(i-center_row-shift_row)) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
        

class PolarTransform(LiquidEngine):
    """
    Polar Transformations using the NanoPyx Liquid Engine
    """
    
    def __init__(self, clear_benchmarks=False, testing=False, verbose=True):
        self._designation = "PolarTransform_lanczos"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, verbose=verbose)

    def run(self, image, tuple out_shape, str scale, run_type=None) -> np.ndarray:
        """
        Polar Transform an image using lanczos interpolation
        :param image: The image to transform
        :type image: np.ndarray or memoryview
        :param out_shape: Desired shape for the output image
        :type out_shape: tuple (n_row, n_col)
        :param scale: Linear or Log transform
        :type scale: str, either 'log' or 'linear'
        :return: The transformed image in polar coordinates
        """
        image = check_image(image)
        nrow, ncol = out_shape
        if scale not in ['linear', 'log']:
            scale = 'linear'
        return self._run(image, nrow, ncol, scale, run_type=run_type)

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

    def _run_opencl(self, image, int nrow, int ncol, str scale, dict device=None, int mem_div=1):

        if device is None:
            device = _fastest_device

        # QUEUE AND CONTEXT
        cl_ctx = cl.Context([device['device']])
        cl_queue = cl.CommandQueue(cl_ctx)

        cdef int nFrames = image.shape[0]
        cdef int nRows = image.shape[1]
        cdef int nCols = image.shape[2]

        output = np.zeros((nFrames, nrow, ncol), dtype=np.float32)

        max_slices = int((device["device"].global_mem_size // (output[0,:,:].nbytes + image[0,:,:].nbytes))/mem_div)
        max_slices = self._check_max_slices(image, max_slices)
        image_in = cl.Buffer(cl_ctx, cl.mem_flags.READ_ONLY, self._check_max_buffer_size(image[0:max_slices,:,:].nbytes, device['device'], max_slices))
        image_out = cl.Buffer(cl_ctx, cl.mem_flags.WRITE_ONLY, self._check_max_buffer_size(output[0:max_slices,:,:].nbytes, device['device'], max_slices))
        cl.enqueue_copy(cl_queue, image_in, image[0:max_slices,:,:]).wait()
        
        cdef int scale_int = 0
        if scale == 'log':
            scale_int = 1

        # Create the program        
        code = self._get_cl_code("_le_interpolation_lanczos_.cl", device['DP'])
        prg = cl.Program(cl_ctx, code).build()
        knl = prg.PolarTransform

        for i in range(0, image.shape[0], max_slices):
            if image.shape[0] - i >= max_slices:
                n_slices = max_slices
            else:
                n_slices = image.shape[0] - i

            knl(
                cl_queue,
                (n_slices, nrow, ncol),
                None,#self.get_work_group(device["device"], (n_slices, nrow, ncol)),
                image_in,
                image_out,
                np.int32(nRows),
                np.int32(nCols),
                np.int32(scale_int)
            )

            cl.enqueue_copy(cl_queue, output[i:i+n_slices,:,:], image_out).wait()
            if i+n_slices<image.shape[0]:
                cl.enqueue_copy(cl_queue, image_in, image[i+n_slices:i+2*n_slices,:,:]).wait()

            cl_queue.finish()

        image_in.release()
        image_out.release()

        return output
        
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
