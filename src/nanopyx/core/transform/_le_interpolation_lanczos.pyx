# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np

cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport cos, sin

from .__interpolation_tools__ import check_image, value2array
from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array


cdef extern from "_c_interpolation_catmull_rom.h":
    float _c_interpolate(float *image, float row, float col, int rows, int cols) nogil


class ShiftAndMagnify(LiquidEngine):
    """
    Shift and Magnify using the NanoPyx Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "ShiftMagnify_LZ"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing,
                        opencl_=True, unthreaded_=True, threaded_=True, threaded_static_=True, 
                        threaded_dynamic_=True, threaded_guided_=True)
        self._default_benchmarks = {'OpenCL': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.16363358300009168, 0.03168145900008312, 0.03195112500020514], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.13761387500017008, 0.12955950000014127, 0.1233359999998811]}, 'Threaded': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.043898375000026135, 0.04184937499985608, 0.04176470899983542], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.31676054200011095, 0.31233449999990626, 0.30947470799992516]}, 'Threaded_dynamic': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.0345830000001115, 0.03378895800005921, 0.03426533299989387], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.23296766700013904, 0.23372599999993326, 0.23327137499995843]}, 'Threaded_guided': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.03484558300010576, 0.033381624999947235, 0.03443237499982388], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.2465884170001118, 0.24920333299996855, 0.2472722090001298]}, 'Threaded_static': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.0445847920000233, 0.04348195800002941, 0.04308850000006714], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.3090422909999688, 0.3100159589998839, 0.3112706250001338]}, 'Unthreaded': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.10468329200011794, 0.10376966599983461, 0.10351591599987842], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.9493457910000416, 0.9433519579999938, 0.9499724170000263]}}

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftAndMagnify.run; replace("Nearest-Neighbor", "Lanczos")
    def run(self, image, shift_row, shift_col, float magnification_row, float magnification_col, run_type=None) -> np.ndarray:
        """
        Shift and magnify an image using Lanczos interpolation
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

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_opencl; replace("nearest_neighbor", "lanczos")
    def _run_opencl(self, image, shift_row, shift_col, float magnification_row, float magnification_col, dict device, int mem_div=1) -> np.ndarray:

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
            if i<=image.shape[0]-max_slices:
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
        self._designation = "ShiftScaleRotate_LZ"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing,
                        opencl_=True, unthreaded_=True, threaded_=True, threaded_static_=True, 
                        threaded_dynamic_=True, threaded_guided_=True)
        self._default_benchmarks = {'OpenCL': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.020539166000162368, 0.01887329100009083, 0.018127374999949097], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.08613391700009743, 0.08152491700002429, 0.0788279999999304]}, 'Threaded': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.03088354199985588, 0.03240920799999003, 0.03100937499993961], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.27657187500017244, 0.27627350000011575, 0.27546158400014065]}, 'Threaded_dynamic': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.030809207999936916, 0.030729832999895734, 0.03082066699994357], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.2742402090000269, 0.27348333300005834, 0.27781258300001355]}, 'Threaded_guided': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.03071866600021167, 0.030636375000085536, 0.030323834000000716], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.2768276669999068, 0.27375016600012714, 0.27681395900003736]}, 'Threaded_static': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.030851999999867985, 0.03102012499994089, 0.030887082999925042], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.2733717500000239, 0.27371095800003786, 0.2762027499998112]}, 'Unthreaded': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.03058774999999514, 0.03372570800001995, 0.030668832999936058], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.27389116599988483, 0.27431166700012, 0.27594858299994485]}}

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate.run; replace("Nearest-Neighbor", "Lanczos")
    def run(self, image, shift_row, shift_col, float scale_row, float scale_col, float angle, run_type=None) -> np.ndarray:
        """
        Shift and scale an image using Lanczos interpolation
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

    # tag-copy: _le_interpolation_nearest_neighbor.ShiftScaleRotate._run_opencl; replace("nearest_neighbor", "lanczos")
    def _run_opencl(self, image, shift_row, shift_col, float scale_row, float scale_col, float angle, dict device, int mem_div=1) -> np.ndarray:

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

        code = self._get_cl_code("_le_interpolation_lanczos_.cl", False)
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
            if i<=image.shape[0]-max_slices:
                cl.enqueue_copy(cl_queue, input_opencl, image[i+n_slices:i+2*n_slices,:,:]).wait()

            cl_queue.finish()

        input_opencl.release()
        output_opencl.release()

        return image_out

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
