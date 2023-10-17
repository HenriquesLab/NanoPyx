# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np

cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport cos, sin

from .__interpolation_tools__ import check_image, value2array
from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array


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

        self._default_benchmarks = {'OpenCL': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.016868334000719187, 0.012203041999782727, 0.011951250000493019], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.0377869579997423, 0.03821733299992047, 0.03995649999978923]}, 'Threaded': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.026508749999266, 0.024312042000019574, 0.024767499999143183], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.1646173329991143, 0.16613820800012036, 0.16288920799979678]}, 'Threaded_dynamic': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.022237749999476364, 0.020750124999722175, 0.021260999999867636], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.13584212500063586, 0.13166883299982146, 0.12987766600053874]}, 'Threaded_guided': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.02203195800029789, 0.021215750000010303, 0.02056575000005978], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.14439316600055463, 0.14346508300059213, 0.14401608299976942]}, 'Threaded_static': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.023586750000504253, 0.024729875000048196, 0.024204291999922134], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.16278725000029226, 0.16391270900021482, 0.16405791600027442]}, 'Unthreaded': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.049936166999941634, 0.04804191699986404, 0.04797699999926408], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.4472621250006341, 0.4428578750002998, 0.44251524999981484]}}

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
        return super().benchmark(image, shift_row, shift_col, magnification_row, magnification_col)
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_opencl; replace("nearest_neighbor", "bicubic")
    def _run_opencl(self, image, shift_row, shift_col, float magnification_row, float magnification_col, dict device, int mem_div = 1) -> np.ndarray:

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

        code = self._get_cl_code("_le_interpolation_bicubic_.cl", device['DP'])
        prg = cl.Program(cl_ctx, code).build()
        knl = prg.shiftAndMagnify

        for i in range(0, image.shape[0], max_slices):
            if image.shape[0] - i >= max_slices:
                n_slices = max_slices
            else:
                n_slices = image.shape[0] - i
            #TODO check that magnification_row and magnification_col are correct
            knl(cl_queue,
                (n_slices, int(image.shape[1]*magnification_row), int(image.shape[2]*magnification_col)), 
                self.get_work_group(dc, (n_slices, image.shape[1]*magnification_row, image.shape[2]*magnification_col)), 
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
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_unthreaded
    def _run_unthreaded(self, float[:,:,:] image, float shift_row, float shift_col, float magnification_row, float magnification_col) -> np.ndarray:
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
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_unthreaded; replace("_run_unthreaded", "_run_threaded"); replace("range(colsM)", "prange(colsM)")
    def _run_threaded(self, float[:,:,:] image, float shift_row, float shift_col, float magnification_row, float magnification_col) -> np.ndarray:
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
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_unthreaded; replace("_run_unthreaded", "_run_threaded_static"); replace("range(colsM)", 'prange(colsM, schedule="static")')
    def _run_threaded_static(self, float[:,:,:] image, float shift_row, float shift_col, float magnification_row, float magnification_col) -> np.ndarray:
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
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_unthreaded; replace("_run_unthreaded", "_run_threaded_dynamic"); replace("range(colsM)", 'prange(colsM, schedule="dynamic")')
    def _run_threaded_dynamic(self, float[:,:,:] image, float shift_row, float shift_col, float magnification_row, float magnification_col) -> np.ndarray:
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
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_unthreaded; replace("_run_unthreaded", "_run_threaded_guided"); replace("range(colsM)", 'prange(colsM, schedule="guided")')
    def _run_threaded_guided(self, float[:,:,:] image, float shift_row, float shift_col, float magnification_row, float magnification_col) -> np.ndarray:
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
        self._default_benchmarks = {'OpenCL': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.012808416999178007, 0.01451266699586995, 0.015523667010711506], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.06442633300321177, 0.0657457500055898, 0.07003754199831747]}, 'Threaded': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.013661334000062197, 0.013460167014272884, 0.013618666998809204], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.122014417022001, 0.11998670798493549, 0.12000020800041966]}, 'Threaded_dynamic': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.01399733399739489, 0.013608000008389354, 0.013484499999321997], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.12151024999911897, 0.12140866700792685, 0.11916912501328625]}, 'Threaded_guided': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.014119541010586545, 0.013741333998041227, 0.013378542003920302], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.12010641701635905, 0.11922020898782648, 0.11907866701949388]}, 'Threaded_static': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.013587041990831494, 0.013635958021041006, 0.013607125001726672], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.12013166601536795, 0.12017516701598652, 0.1189643329998944]}, 'Unthreaded': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.013573625008575618, 0.013579208025475964, 0.013639416021760553], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.12159570900257677, 0.11987237498397008, 0.11896608298411593]}}

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
        return super().benchmark(image, shift_row, shift_col, scale_row, scale_col, angle)
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_opencl; replace("nearest_neighbor", "bicubic")
    def _run_opencl(self, image, shift_row, shift_col, float scale_row, float scale_col, float angle, dict device, int mem_div = 1) -> np.ndarray:

        # QUEUE AND CONTEXT
        cl_ctx = cl.Context([device['device']])
        dc = device["device"]
        cl_queue = cl.CommandQueue(cl_ctx)

        output_shape = (image.shape[0], int(image.shape[1]), int(image.shape[2]))
        image_out = np.zeros(output_shape, dtype=np.float32)

        max_slices = int((dc.global_mem_size // (image_out[0,:,:].nbytes + image[0,:,:].nbytes))/mem_div)
        max_slices = self._check_max_slices(image, max_slices)

        mf = cl.mem_flags
        input_opencl = cl.Buffer(cl_ctx, mf.READ_ONLY, image[0:max_slices,:,:].nbytes)
        output_opencl = cl.Buffer(cl_ctx, mf.WRITE_ONLY, image_out[0:max_slices,:,:].nbytes)
        cl.enqueue_copy(cl_queue, input_opencl, image[0:max_slices,:,:]).wait()

        code = self._get_cl_code("_le_interpolation_bicubic_.cl", device['DP'])
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
                self.get_work_group(dc, (n_slices, image.shape[1], image.shape[2])),
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

        return np.asarray(image_out, dtype=np.float32)
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_unthreaded
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
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_unthreaded; replace("_run_unthreaded", "_run_threaded"); replace("range(colsM)", "prange(colsM)")
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
                for j in range(cols):
                    for i in range(rows):
                        col = (a*(j-center_col-shift_col)+b*(i-center_row-shift_row)) + center_col
                        row = (c*(j-center_col-shift_col)+d*(i-center_row-shift_row)) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_unthreaded; replace("_run_unthreaded", "_run_threaded_static"); replace("range(colsM)", 'prange(colsM, schedule="static")')
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
                for j in range(cols):
                    for i in range(rows):
                        col = (a*(j-center_col-shift_col)+b*(i-center_row-shift_row)) + center_col
                        row = (c*(j-center_col-shift_col)+d*(i-center_row-shift_row)) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_unthreaded; replace("_run_unthreaded", "_run_threaded_dynamic"); replace("range(colsM)", 'prange(colsM, schedule="dynamic")')
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
                for j in range(cols):
                    for i in range(rows):
                        col = (a*(j-center_col-shift_col)+b*(i-center_row-shift_row)) + center_col
                        row = (c*(j-center_col-shift_col)+d*(i-center_row-shift_row)) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_unthreaded; replace("_run_unthreaded", "_run_threaded_guided"); replace("range(colsM)", 'prange(colsM, schedule="guided")')
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
                for j in range(cols):
                    for i in range(rows):
                        col = (a*(j-center_col-shift_col)+b*(i-center_row-shift_row)) + center_col
                        row = (c*(j-center_col-shift_col)+d*(i-center_row-shift_row)) + center_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        return image_out
    # tag-end
