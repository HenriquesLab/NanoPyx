# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport sqrt, pow
from .__opencl__ import cl, cl_array
from .__liquid_engine__ import LiquidEngine
from .__interpolation_tools__ import check_image

cdef extern from "_c_sr_radial_gradient_convergence.h":
    float _c_calculate_rgc(int xM, int yM, float* imIntGx, float* imIntGy, int colsM, int rowsM, int magnification, float Gx_Gy_MAGNIFICATION, float fwhm, float tSO, float tSS, float sensitivity) nogil

class RadialGradientConvergence(LiquidEngine):
    """
    Radial gradient convergence using the NanoPyx Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "RadialGradientConvergence"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing,
                        unthreaded_=True, threaded_=True, threaded_static_=True, 
                        threaded_dynamic_=True, threaded_guided_=True, opencl_=True)

        
        self._default_benchmarks = {'OpenCL': {"(['shape(100, 100, 100)', 'shape(100, 100, 100)', 'shape(100, 200, 200)', 'number(5)', 'number(1.5)', 'number(1.0)', True], {})": [3e+19, 0.016602458999841474, 0.014774957999634353, 0.014550083000358427], "(['shape(100, 300, 300)', 'shape(100, 300, 300)', 'shape(100, 600, 600)', 'number(5)', 'number(1.5)', 'number(1.0)', True], {})": [2.187e+22, 0.061947042000156216, 0.05570804199987833, 0.05627254199998788]}, 'Threaded': {"(['shape(100, 100, 100)', 'shape(100, 100, 100)', 'shape(100, 200, 200)', 'number(5)', 'number(1.5)', 'number(1.0)', True], {})": [3e+19, 0.01809087499987072, 0.018405166999855282, 0.020630749999781983], "(['shape(100, 300, 300)', 'shape(100, 300, 300)', 'shape(100, 600, 600)', 'number(5)', 'number(1.5)', 'number(1.0)', True], {})": [2.187e+22, 0.1779053750001367, 0.17156229100010023, 0.16890108300003703]}, 'Threaded_dynamic': {"(['shape(100, 100, 100)', 'shape(100, 100, 100)', 'shape(100, 200, 200)', 'number(5)', 'number(1.5)', 'number(1.0)', True], {})": [3e+19, 0.01563541699988491, 0.016560666999794194, 0.016359334000298986], "(['shape(100, 300, 300)', 'shape(100, 300, 300)', 'shape(100, 600, 600)', 'number(5)', 'number(1.5)', 'number(1.0)', True], {})": [2.187e+22, 0.1260679170000003, 0.1298676249998607, 0.12453404200005025]}, 'Threaded_guided': {"(['shape(100, 100, 100)', 'shape(100, 100, 100)', 'shape(100, 200, 200)', 'number(5)', 'number(1.5)', 'number(1.0)', True], {})": [3e+19, 0.016395916999954352, 0.016721625000172935, 0.01623758400000952], "(['shape(100, 300, 300)', 'shape(100, 300, 300)', 'shape(100, 600, 600)', 'number(5)', 'number(1.5)', 'number(1.0)', True], {})": [2.187e+22, 0.13648262499964403, 0.13698016699981963, 0.12756820799995694]}, 'Threaded_static': {"(['shape(100, 100, 100)', 'shape(100, 100, 100)', 'shape(100, 200, 200)', 'number(5)', 'number(1.5)', 'number(1.0)', True], {})": [3e+19, 0.021419832999981736, 0.019069041999955516, 0.01931129200011128], "(['shape(100, 300, 300)', 'shape(100, 300, 300)', 'shape(100, 600, 600)', 'number(5)', 'number(1.5)', 'number(1.0)', True], {})": [2.187e+22, 0.18173866699999053, 0.1682213750000301, 0.16703279200010002]}, 'Unthreaded': {"(['shape(100, 100, 100)', 'shape(100, 100, 100)', 'shape(100, 200, 200)', 'number(5)', 'number(1.5)', 'number(1.0)', True], {})": [3e+19, 0.03560816600020189, 0.03549925000015719, 0.03543366600024456], "(['shape(100, 300, 300)', 'shape(100, 300, 300)', 'shape(100, 600, 600)', 'number(5)', 'number(1.5)', 'number(1.0)', True], {})": [2.187e+22, 0.6588103340000089, 0.6491647500001818, 0.6530582079999476]}}


    def run(self, gradient_col_interp, gradient_row_interp, image_interp, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1 , doIntensityWeighting: bool = True, run_type = None): 
        gradient_col_interp = check_image(gradient_col_interp)
        gradient_row_interp = check_image(gradient_row_interp)
        image_interp = check_image(image_interp)
        return self._run(gradient_col_interp, gradient_row_interp, image_interp, magnification, radius, sensitivity, doIntensityWeighting, run_type=run_type)
    

    def benchmark(self, gradient_col_interp, gradient_row_interp, image_interp, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1 , doIntensityWeighting: bool = True):
        gradient_col_interp = check_image(gradient_col_interp)
        gradient_row_interp = check_image(gradient_row_interp)
        image_interp = check_image(image_interp)
        return super().benchmark(gradient_col_interp, gradient_row_interp, image_interp, magnification, radius, sensitivity, doIntensityWeighting)


    # tag-start: _le_radial_gradient_convergence.RadialGradientConvergence._run_unthreaded
    def _run_unthreaded(self, float[:,:,:] gradient_col_interp, float[:,:,:] gradient_row_interp, float[:,:,:] image_interp, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True):

        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float Gx_Gy_MAGNIFICATION = 2.0
        cdef int _magnification = magnification
        cdef float _sensitivity = sensitivity
        cdef int _doIntensityWeighting = doIntensityWeighting

        cdef int nFrames = gradient_col_interp.shape[0]
        cdef int rowsM = <int>(gradient_row_interp.shape[1] / Gx_Gy_MAGNIFICATION)
        cdef int colsM = <int>(gradient_row_interp.shape[2] / Gx_Gy_MAGNIFICATION)

        cdef float [:,:,:] rgc_map = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)

        cdef int f, rM, cM
        with nogil:
            for f in range(nFrames):
                    for rM in range(_magnification*2, rowsM - _magnification*2): 
                        for cM in range(_magnification*2, colsM - _magnification*2):
                            if _doIntensityWeighting:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity) * image_interp[f, rM, cM] 
                            else:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity)

        return np.asarray(rgc_map,dtype=np.float32)
        # tag-end

    # tag-copy:  _le_radial_gradient_convergence.RadialGradientConvergence._run_unthreaded; replace("_run_unthreaded", "_run_threaded"); replace("range(_magnification*2, rowsM - _magnification*2)", "prange(_magnification*2, rowsM - _magnification*2)")
    def _run_threaded(self, float[:,:,:] gradient_col_interp, float[:,:,:] gradient_row_interp, float[:,:,:] image_interp, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True):

        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float Gx_Gy_MAGNIFICATION = 2.0
        cdef int _magnification = magnification
        cdef float _sensitivity = sensitivity
        cdef int _doIntensityWeighting = doIntensityWeighting

        cdef int nFrames = gradient_col_interp.shape[0]
        cdef int rowsM = <int>(gradient_row_interp.shape[1] / Gx_Gy_MAGNIFICATION)
        cdef int colsM = <int>(gradient_row_interp.shape[2] / Gx_Gy_MAGNIFICATION)

        cdef float [:,:,:] rgc_map = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)

        cdef int f, rM, cM
        with nogil:
            for f in range(nFrames):
                    for rM in prange(_magnification*2, rowsM - _magnification*2): 
                        for cM in range(_magnification*2, colsM - _magnification*2):
                            if _doIntensityWeighting:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity) * image_interp[f, rM, cM] 
                            else:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity)
        return np.asarray(rgc_map,dtype=np.float32)
        # tag-end

    # tag-copy:  _le_radial_gradient_convergence.RadialGradientConvergence._run_unthreaded; replace("_run_unthreaded", "_run_threaded_static"); replace("range(_magnification*2, rowsM - _magnification*2)", 'prange(_magnification*2, rowsM - _magnification*2, schedule="static")')
    def _run_threaded_static(self, float[:,:,:] gradient_col_interp, float[:,:,:] gradient_row_interp, float[:,:,:] image_interp, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True):

        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float Gx_Gy_MAGNIFICATION = 2.0
        cdef int _magnification = magnification
        cdef float _sensitivity = sensitivity
        cdef int _doIntensityWeighting = doIntensityWeighting

        cdef int nFrames = gradient_col_interp.shape[0]
        cdef int rowsM = <int>(gradient_row_interp.shape[1] / Gx_Gy_MAGNIFICATION)
        cdef int colsM = <int>(gradient_row_interp.shape[2] / Gx_Gy_MAGNIFICATION)

        cdef float [:,:,:] rgc_map = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)

        cdef int f, rM, cM
        with nogil:
            for f in range(nFrames):
                    for rM in prange(_magnification*2, rowsM - _magnification*2, schedule="static"): 
                        for cM in range(_magnification*2, colsM - _magnification*2):
                            if _doIntensityWeighting:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity) * image_interp[f, rM, cM] 
                            else:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity)
        return np.asarray(rgc_map,dtype=np.float32)
        # tag-end

    # tag-copy:  _le_radial_gradient_convergence.RadialGradientConvergence._run_unthreaded; replace("_run_unthreaded", "_run_threaded_dynamic"); replace("range(_magnification*2, rowsM - _magnification*2)", 'prange(_magnification*2, rowsM - _magnification*2, schedule="dynamic")')
    def _run_threaded_dynamic(self, float[:,:,:] gradient_col_interp, float[:,:,:] gradient_row_interp, float[:,:,:] image_interp, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True):

        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float Gx_Gy_MAGNIFICATION = 2.0
        cdef int _magnification = magnification
        cdef float _sensitivity = sensitivity
        cdef int _doIntensityWeighting = doIntensityWeighting

        cdef int nFrames = gradient_col_interp.shape[0]
        cdef int rowsM = <int>(gradient_row_interp.shape[1] / Gx_Gy_MAGNIFICATION)
        cdef int colsM = <int>(gradient_row_interp.shape[2] / Gx_Gy_MAGNIFICATION)

        cdef float [:,:,:] rgc_map = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)

        cdef int f, rM, cM
        with nogil:
            for f in range(nFrames):
                    for rM in prange(_magnification*2, rowsM - _magnification*2, schedule="dynamic"): 
                        for cM in range(_magnification*2, colsM - _magnification*2):
                            if _doIntensityWeighting:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity) * image_interp[f, rM, cM] 
                            else:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity)
        return np.asarray(rgc_map,dtype=np.float32)
        # tag-end

    # tag-copy:  _le_radial_gradient_convergence.RadialGradientConvergence._run_unthreaded; replace("_run_unthreaded", "_run_threaded_guided"); replace("range(_magnification*2, rowsM - _magnification*2)", 'prange(_magnification*2, rowsM - _magnification*2, schedule="guided")')
    def _run_threaded_guided(self, float[:,:,:] gradient_col_interp, float[:,:,:] gradient_row_interp, float[:,:,:] image_interp, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True):

        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float Gx_Gy_MAGNIFICATION = 2.0
        cdef int _magnification = magnification
        cdef float _sensitivity = sensitivity
        cdef int _doIntensityWeighting = doIntensityWeighting

        cdef int nFrames = gradient_col_interp.shape[0]
        cdef int rowsM = <int>(gradient_row_interp.shape[1] / Gx_Gy_MAGNIFICATION)
        cdef int colsM = <int>(gradient_row_interp.shape[2] / Gx_Gy_MAGNIFICATION)

        cdef float [:,:,:] rgc_map = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)

        cdef int f, rM, cM
        with nogil:
            for f in range(nFrames):
                    for rM in prange(_magnification*2, rowsM - _magnification*2, schedule="guided"): 
                        for cM in range(_magnification*2, colsM - _magnification*2):
                            if _doIntensityWeighting:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity) * image_interp[f, rM, cM] 
                            else:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity)
        return np.asarray(rgc_map,dtype=np.float32)
        # tag-end

    
    def _run_opencl(self, gradient_col_interp, gradient_row_interp, image_interp, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True, device=None):

        # gradient gxgymag*mag*size
        # image_interp = mag*size
        # output = image_interp

        # Context and queue
        cl_ctx = cl.Context([device['device']])
        cl_queue = cl.CommandQueue(cl_ctx)

        # Parameters
        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float Gx_Gy_MAGNIFICATION = 2.0
        cdef int _magnification = magnification
        cdef float _sensitivity = sensitivity
        cdef int _doIntensityWeighting = doIntensityWeighting

        # Sizes
        cdef int nFrames = gradient_col_interp.shape[0]
        cdef int rows_interpolated = <int>(gradient_row_interp.shape[1] / Gx_Gy_MAGNIFICATION)
        cdef int cols_interpolated = <int>(gradient_row_interp.shape[2] / Gx_Gy_MAGNIFICATION)

        # Grid size of the global work space
        lowest_row = _magnification*2 
        highest_row = rows_interpolated - _magnification*2
        lowest_col = _magnification*2
        highest_col =  cols_interpolated - _magnification*2

        # Output 
        rgc_map = np.zeros((nFrames, rows_interpolated, cols_interpolated), dtype=np.float32)

        # Calculating max slices
        size_per_slice = gradient_col_interp[0,:,:].nbytes + gradient_row_interp[0,:,:].nbytes + image_interp[0,:,:].nbytes + rgc_map[0,:,:].nbytes
        max_slices = int((device['device'].global_mem_size // (size_per_slice))/4)   # TODO 3 is a magic number 

        # Initial buffers
        mf = cl.mem_flags
        grad_col_int_in = cl.Buffer(cl_ctx, mf.READ_ONLY, gradient_col_interp[0:max_slices,:,:].nbytes)
        cl.enqueue_copy(cl_queue, grad_col_int_in, gradient_col_interp[0:max_slices,:,:]).wait()

        grad_row_int_in = cl.Buffer(cl_ctx, mf.READ_ONLY, gradient_row_interp[0:max_slices,:,:].nbytes)
        cl.enqueue_copy(cl_queue, grad_row_int_in, gradient_row_interp[0:max_slices,:,:]).wait()

        image_interp_in = cl.Buffer(cl_ctx, mf.READ_ONLY, image_interp[0:max_slices,:,:].nbytes)
        cl.enqueue_copy(cl_queue, image_interp_in, image_interp[0:max_slices,:,:]).wait()

        rgc_map_out = cl.Buffer(cl_ctx, mf.WRITE_ONLY, rgc_map[0:max_slices,:,:].nbytes)

        # Code and building the kernel
        code = self._get_cl_code("_le_radial_gradient_convergence.cl", device['DP'])
        prg = cl.Program(cl_ctx, code).build() 
        knl = prg.calculate_rgc

        # Actual computation
        for i in range(0, nFrames-1, max_slices):
            if nFrames - i >= max_slices:
                n_slices = max_slices
            else:
                n_slices = nFrames - i

            knl(cl_queue, 
                (n_slices, highest_row - lowest_row, highest_col - lowest_col), 
                self.get_work_group(device['device'],(n_slices, highest_row - lowest_row, highest_col - lowest_col)), 
                grad_col_int_in,
                grad_row_int_in,
                image_interp_in,
                rgc_map_out,
                np.int32(cols_interpolated), 
                np.int32(rows_interpolated), 
                np.int32(_magnification), 
                np.float32(Gx_Gy_MAGNIFICATION), 
                np.float32(fwhm), 
                np.float32(tSO), 
                np.float32(tSS), 
                np.float32(_sensitivity), 
                np.int32(_doIntensityWeighting)).wait()
            
            # Copy output
            cl.enqueue_copy(cl_queue, rgc_map[i:i+n_slices,:,:], rgc_map_out).wait()

            # Copy input
            if i<=nFrames-max_slices:
                cl.enqueue_copy(cl_queue, grad_col_int_in, gradient_col_interp[i+n_slices:i+2*n_slices,:,:]).wait() 
                cl.enqueue_copy(cl_queue, grad_row_int_in, gradient_row_interp[i+n_slices:i+2*n_slices,:,:]).wait() 
                cl.enqueue_copy(cl_queue, image_interp_in, image_interp[i+n_slices:i+2*n_slices,:,:]).wait() 

            cl_queue.finish()

        
        return rgc_map
        




