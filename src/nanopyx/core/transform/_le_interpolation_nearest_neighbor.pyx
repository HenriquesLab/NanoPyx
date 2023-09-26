# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np

cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport cos, sin, pi, hypot, exp, log

from .__interpolation_tools__ import check_image, value2array
from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array
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

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "ShiftMagnify_NN"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing,
                        opencl_=True, unthreaded_=True, threaded_=True, threaded_static_=True, 
                        threaded_dynamic_=True, threaded_guided_=True, python_=True, njit_=True)

        self._default_benchmarks = {'Numba': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.011261249979725108, 0.008658000006107613, 0.009081458993023261], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.038259124994510785, 0.03869416698580608, 0.038678791985148564]}, 'OpenCL': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.009874707990093157, 0.010942917026113719, 0.01088916600565426], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.031119999999646097, 0.03063349999138154, 0.030909374996554106]}, 'Python': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 4.6282880830112845, 4.621970458974829, 4.57845712499693], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 41.50990308300243, 41.28186725001433, 40.866472624999005]}, 'Threaded': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.013660958007676527, 0.010840707982424647, 0.011427457997342572], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.07226854198961519, 0.06988441600697115, 0.06987566701718606]}, 'Threaded_dynamic': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.012701833009487018, 0.01182133299880661, 0.012448374996893108], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.06524320901371539, 0.06451412499882281, 0.06617041601566598]}, 'Threaded_guided': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.012816083006327972, 0.012354833976132795, 0.012373666977509856], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.0736009580141399, 0.07187350001186132, 0.07309983397135511]}, 'Threaded_static': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.010962625005049631, 0.01140041701728478, 0.010838375019375235], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.07002141600241885, 0.06659716597641818, 0.06886829200084321]}, 'Unthreaded': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [40000000000.0, 0.006760124990250915, 0.005808415997307748, 0.005789166985778138], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)'], {})": [360000000000.0, 0.060527583002112806, 0.06079408299410716, 0.06013770800200291]}}

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
    #@LiquidEngine._logger(logger)
    def _run_opencl(self, image, shift_row, shift_col, float magnification_row, float magnification_col, dict device) -> np.ndarray:

        # QUEUE AND CONTEXT
        cl_ctx = cl.Context([device['device']])
        dc = device['device']
        cl_queue = cl.CommandQueue(cl_ctx)
        
        output_shape = (image.shape[0], int(image.shape[1]*magnification_row), int(image.shape[2]*magnification_col))
        image_out = np.zeros(output_shape, dtype=np.float32)

        # TODO 3 is a magic number 
        max_slices = int((dc.global_mem_size // (image_out[0,:,:].nbytes + image[0,:,:].nbytes))/3)
        # TODO add exception if max_slices < 1 

        mf = cl.mem_flags
        input_opencl = cl.Buffer(cl_ctx, mf.READ_ONLY, image[0:max_slices,:,:].nbytes)
        cl.enqueue_copy(cl_queue, input_opencl, image[0:max_slices,:,:]).wait()
        output_opencl = cl.Buffer(cl_ctx, mf.WRITE_ONLY, image_out[0:max_slices,:,:].nbytes)

        code = self._get_cl_code("_le_interpolation_nearest_neighbor_.cl", device['DP'])
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
                np.float32(shift_row[0]), 
                np.float32(shift_col[0]), 
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

    # tag-start: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_unthreaded
    #@LiquidEngine._logger(logger)
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
    #@LiquidEngine._logger(logger)
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
    #@LiquidEngine._logger(logger)
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
    #@LiquidEngine._logger(logger)
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
    #@LiquidEngine._logger(logger)
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
    #@LiquidEngine._logger(logger)
    def _run_python(self, image, shift_row, shift_col, magnification_row, magnification_col) -> np.ndarray:
        image_out = _py_shift_magnify(image, shift_row, shift_col, magnification_row, magnification_col)
        return image_out
    # tag-end

    # tag-start: _le_interpolation_nearest_neighbor.ShiftAndMagnify._run_njit
    #@LiquidEngine._logger(logger)
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
    
    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "ShiftScaleRotate_NN"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, 
                        opencl_=True, unthreaded_=True, threaded_=True, threaded_static_=True, 
                        threaded_dynamic_=True, threaded_guided_=True, python_=True, njit_=True)
                        
        self._default_benchmarks = {'Numba': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.010381959000369534, 0.008006833988474682, 0.00731566600734368], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.013968916988233104, 0.014351999998325482, 0.014282417017966509]}, 'OpenCL': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.01039199999650009, 0.009912667010212317, 0.010021291993325576], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.052067666983930394, 0.05267445798381232, 0.0507857910124585]}, 'Python': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 3.497419292019913, 3.5178624170075636, 3.529090833006194], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 31.53393600002164, 31.962537165993126, 32.02481729097781]}, 'Threaded': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.008445790997939184, 0.010681208019377664, 0.00998375000199303], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.02084358301362954, 0.021319999999832362, 0.02256958300131373]}, 'Threaded_dynamic': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.010332499979995191, 0.011359374999301508, 0.011331582994898781], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.02064062500721775, 0.01945033299853094, 0.01905904200975783]}, 'Threaded_guided': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.011013916984666139, 0.010239374998491257, 0.010085290996357799], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.02084879099857062, 0.02114379100385122, 0.022826500004157424]}, 'Threaded_static': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.009036000003106892, 0.009778459003427997, 0.01019124998128973], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.020108958007767797, 0.021447209001053125, 0.020637999987229705]}, 'Unthreaded': {"(['shape(100, 100, 100)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [40000000000.0, 0.0019341250008437783, 0.002130040986230597, 0.001987041992833838], "(['shape(100, 300, 300)', 'shape(100,)', 'shape(100,)', 'number(2.0)', 'number(2.0)', 'number(0.0)'], {})": [360000000000.0, 0.015529834025073797, 0.01632987501216121, 0.016242625017184764]}}

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
    def _run_opencl(self, image, shift_row, shift_col, float scale_row, float scale_col, float angle, dict device) -> np.ndarray:

        # QUEUE AND CONTEXT
        cl_ctx = cl.Context([device['device']])
        cl_queue = cl.CommandQueue(cl_ctx)

        code = self._get_cl_code("_le_interpolation_nearest_neighbor_.cl", device['DP'])

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
            shift_row_in.data,
            shift_col_in.data,
            np.float32(scale_row),
            np.float32(scale_col),
            np.float32(angle)
        )

        # Wait for queue to finish
        cl_queue.finish()

        return np.asarray(image_out.get(),dtype=np.float32)

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


'''
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
    _designation = "PolarTransform_NN"
    
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
    @LiquidEngine._logger(logger)
    def _run_opencl(self, float[:,:,:] image, int nrow, int ncol, str scale):
        
        # Swap row and columns because opencl is strange and stores the
        # array in a buffer in fortran ordering despite the original
        # numpy array being in C order.
        image = np.ascontiguousarray(np.swapaxes(image, 1, 2), dtype=np.float32)

        code = self._get_cl_code("_le_interpolation_nearest_neighbor_.cl")

        cdef int nFrames = image.shape[0]
        cdef int rowsM = image.shape[1]
        cdef int colsM = image.shape[2]

        image_in = cl_array.to_device(cl_queue, image)
        image_out = cl_array.zeros(cl_queue, (nFrames, nrow, ncol), dtype=np.float32)
        
        cdef int scale_int = 0
        if scale == 'log':
            scale_int = 1

        # Create the program
        prg = cl.Program(cl_ctx, code).build()

        # Run the kernel
        prg.PolarTransform(
            cl_queue,
            image_out.shape,
            None,
            image_in.data,
            image_out.data,
            scale_int
        )

        # Wait for queue to finish
        cl_queue.finish()

        # Swap rows and columns back
        return np.ascontiguousarray(np.swapaxes(image_out.get(), 1, 2), dtype=np.float32)
    # tag-end

    # tag-start: _le_interpolation_nearest_neighbor.PolarTransform._run_unthreaded
    @LiquidEngine._logger(logger)
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
    @LiquidEngine._logger(logger)
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
    @LiquidEngine._logger(logger)
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
    @LiquidEngine._logger(logger)
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
    @LiquidEngine._logger(logger)
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
    @LiquidEngine._logger(logger)
    def _run_python(self, image, nrow, ncol, scale):
        return 0
    # tag-end

    # tag-start: _le_interpolation_nearest_neighbor.PolarTransform._run_njit
    @LiquidEngine._logger(logger)
    def _run_njit(self, image, nrow, ncol, scale):
        return 0
    # tag-end
'''