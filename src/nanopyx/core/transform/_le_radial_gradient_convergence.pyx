# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport sqrt, pow
from ...__opencl__ import cl, cl_array, _fastest_device
from ...__liquid_engine__ import LiquidEngine
from .__interpolation_tools__ import check_image

cdef extern from "_c_sr_radial_gradient_convergence.h":
    float _c_calculate_rgc(int xM, int yM, float* imIntGx, float* imIntGy, int colsM, int rowsM, int magnification, float Gx_Gy_MAGNIFICATION, float fwhm, float tSO, float tSS, float sensitivity, float offset, float xyoffset, float angle) nogil

class RadialGradientConvergence(LiquidEngine):
    """
    Radial gradient convergence using the NanoPyx Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False, verbose=True):
        self._designation = "RadialGradientConvergence"
        super().__init__(
            clear_benchmarks=clear_benchmarks, testing=testing,
            verbose=verbose)


    def run(self, gradient_col_interp, gradient_row_interp, image_interp, magnification: int = 5, grad_magnification: int = 1, radius: float = 1.5, sensitivity: float = 1 , doIntensityWeighting: bool = True, offset: float = 0, xyoffset: float = 0, angle: float = 0, run_type = None): 
        gradient_col_interp = check_image(gradient_col_interp)
        gradient_row_interp = check_image(gradient_row_interp)
        image_interp = check_image(image_interp)
        return self._run(gradient_col_interp, gradient_row_interp, image_interp, magnification, grad_magnification, radius, sensitivity, doIntensityWeighting, offset, xyoffset, angle, run_type=run_type)
    

    def benchmark(self, gradient_col_interp, gradient_row_interp, image_interp, magnification: int = 5, grad_magnification: int = 2, radius: float = 1.5, sensitivity: float = 1 , doIntensityWeighting: bool = True, offset: float = 0, xyoffset: float = 0, angle: float = 0):
        gradient_col_interp = check_image(gradient_col_interp)
        gradient_row_interp = check_image(gradient_row_interp)
        image_interp = check_image(image_interp)
        return super().benchmark(gradient_col_interp, gradient_row_interp, image_interp, magnification, grad_magnification, radius, sensitivity, doIntensityWeighting, offset, xyoffset, angle)

    def _run_unthreaded(self, float[:,:,:] gradient_col_interp, float[:,:,:] gradient_row_interp, float[:,:,:] image_interp, magnification=5, grad_magnification=1, radius=1.5, sensitivity=1, doIntensityWeighting=True, offset: float =0, xyoffset: float =0, angle: float =0):
        """
        @cpu
        @cython
        """
        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef int margin = int(fwhm*magnification)
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float Gx_Gy_MAGNIFICATION = grad_magnification
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
                    for rM in range(margin, rowsM - margin): 
                        for cM in range(margin, colsM - margin):
                            if _doIntensityWeighting:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity, offset, xyoffset, angle) * image_interp[f, rM, cM] 
                            else:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity, offset, xyoffset, angle)

        return np.asarray(rgc_map,dtype=np.float32)

    def _run_threaded(self, float[:,:,:] gradient_col_interp, float[:,:,:] gradient_row_interp, float[:,:,:] image_interp, magnification=5, grad_magnification=1, radius=1.5, sensitivity=1, doIntensityWeighting=True, offset: float =0, xyoffset: float =0, angle: float =0):
        """
        @cpu
        @threaded
        @cython
        """
        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef int margin = int(fwhm) * magnification
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float Gx_Gy_MAGNIFICATION = grad_magnification
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
                for rM in prange(margin, rowsM - margin):
                    for cM in range(margin, colsM - margin):
                        if _doIntensityWeighting:
                            rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity, offset, xyoffset, angle) * image_interp[f, rM, cM] 
                        else:
                            rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity, offset, xyoffset, angle)
        return np.asarray(rgc_map,dtype=np.float32)
    def _run_threaded_guided(self, float[:,:,:] gradient_col_interp, float[:,:,:] gradient_row_interp, float[:,:,:] image_interp, magnification=5, grad_magnification=1, radius=1.5, sensitivity=1, doIntensityWeighting=True, offset: float =0, xyoffset: float =0, angle: float =0):
        """
        @cpu
        @threaded
        @cython
        """
        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef int margin = int(fwhm) * magnification
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float Gx_Gy_MAGNIFICATION = grad_magnification
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
                for rM in prange(margin, rowsM - margin, schedule="guided"):
                    for cM in range(margin, colsM - margin):
                        if _doIntensityWeighting:
                            rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity, offset, xyoffset, angle) * image_interp[f, rM, cM] 
                        else:
                            rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity, offset, xyoffset, angle)
        return np.asarray(rgc_map,dtype=np.float32)
    def _run_threaded_dynamic(self, float[:,:,:] gradient_col_interp, float[:,:,:] gradient_row_interp, float[:,:,:] image_interp, magnification=5, grad_magnification=1, radius=1.5, sensitivity=1, doIntensityWeighting=True, offset: float =0, xyoffset: float =0, angle: float =0):
        """
        @cpu
        @threaded
        @cython
        """
        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef int margin = int(fwhm) * magnification
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float Gx_Gy_MAGNIFICATION = grad_magnification
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
                for rM in prange(margin, rowsM - margin, schedule="dynamic"):
                    for cM in range(margin, colsM - margin):
                        if _doIntensityWeighting:
                            rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity, offset, xyoffset, angle) * image_interp[f, rM, cM] 
                        else:
                            rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity, offset, xyoffset, angle)
        return np.asarray(rgc_map,dtype=np.float32)
    def _run_threaded_static(self, float[:,:,:] gradient_col_interp, float[:,:,:] gradient_row_interp, float[:,:,:] image_interp, magnification=5, grad_magnification=1, radius=1.5, sensitivity=1, doIntensityWeighting=True, offset: float =0, xyoffset: float =0, angle: float =0):
        """
        @cpu
        @threaded
        @cython
        """
        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef int margin = int(fwhm) * magnification
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float Gx_Gy_MAGNIFICATION = grad_magnification
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
                for rM in prange(margin, rowsM - margin, schedule="static"):
                    for cM in range(margin, colsM - margin):
                        if _doIntensityWeighting:
                            rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity, offset, xyoffset, angle) * image_interp[f, rM, cM] 
                        else:
                            rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity, offset, xyoffset, angle)
        return np.asarray(rgc_map,dtype=np.float32)

    
    def _run_opencl(self, gradient_col_interp, gradient_row_interp, image_interp, magnification=5, grad_magnification=1, radius=1.5, sensitivity=1, doIntensityWeighting=True, offset=0, xyoffset=0, angle=0, device=None, int mem_div=1):
        """
        @gpu
        """
        if device is None:
            device = _fastest_device

        # gradient gxgymag*mag*size
        # image_interp = mag*size
        # output = image_interp

        # Context and queue
        cl_ctx = cl.Context([device['device']])
        cl_queue = cl.CommandQueue(cl_ctx)

        # Parameters
        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef int margin = int(fwhm*magnification)
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float Gx_Gy_MAGNIFICATION = grad_magnification
        cdef int _magnification = magnification
        cdef float _sensitivity = sensitivity
        cdef int _doIntensityWeighting = doIntensityWeighting
        cdef float _offset = offset
        cdef float _xyoffset = xyoffset
        cdef float _angle = angle

        # Sizes
        cdef int nFrames = gradient_col_interp.shape[0]
        cdef int rows_interpolated = <int>(gradient_row_interp.shape[1] / Gx_Gy_MAGNIFICATION)
        cdef int cols_interpolated = <int>(gradient_row_interp.shape[2] / Gx_Gy_MAGNIFICATION)

        # Grid size of the global work space
        lowest_row = margin # TODO discuss edges calculation
        highest_row = rows_interpolated - margin
        lowest_col = margin
        highest_col =  cols_interpolated - margin

        # Output 
        rgc_map = np.zeros((nFrames, rows_interpolated, cols_interpolated), dtype=np.float32)

        # Calculating max slices
        size_per_slice = gradient_col_interp[0,:,:].nbytes + gradient_row_interp[0,:,:].nbytes + image_interp[0,:,:].nbytes + rgc_map[0,:,:].nbytes
        max_slices = int((device['device'].global_mem_size // (size_per_slice))/mem_div)
        max_slices = self._check_max_slices(image_interp, max_slices)

        # Initial buffers
        mf = cl.mem_flags
        grad_col_int_in = cl.Buffer(cl_ctx, mf.READ_ONLY, self._check_max_buffer_size(gradient_col_interp[0:max_slices,:,:].nbytes, device['device'], max_slices))
        grad_row_int_in = cl.Buffer(cl_ctx, mf.READ_ONLY, self._check_max_buffer_size(gradient_row_interp[0:max_slices,:,:].nbytes, device['device'], max_slices))
        image_interp_in = cl.Buffer(cl_ctx, mf.READ_ONLY, self._check_max_buffer_size(image_interp[0:max_slices,:,:].nbytes, device['device'], max_slices))
        rgc_map_out = cl.Buffer(cl_ctx, mf.WRITE_ONLY, self._check_max_buffer_size(rgc_map[0:max_slices,:,:].nbytes, device['device'], max_slices))


        cl.enqueue_copy(cl_queue, grad_col_int_in, gradient_col_interp[0:max_slices,:,:]).wait()
        cl.enqueue_copy(cl_queue, grad_row_int_in, gradient_row_interp[0:max_slices,:,:]).wait()
        cl.enqueue_copy(cl_queue, image_interp_in, image_interp[0:max_slices,:,:]).wait()

        # Code and building the kernel
        code = self._get_cl_code("_le_radial_gradient_convergence.cl", device['DP'])
        prg = cl.Program(cl_ctx, code).build() 
        knl = prg.calculate_rgc

        # Actual computation
        for i in range(0, nFrames, max_slices):
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
                np.int32(_doIntensityWeighting),
                np.float32(_offset),
                np.float32(_xyoffset),
                np.float32(_angle)).wait()
            
            # Copy output
            cl.enqueue_copy(cl_queue, rgc_map[i:i+n_slices,:,:], rgc_map_out).wait()

            # Copy input
            if i+n_slices<nFrames:
                cl.enqueue_copy(cl_queue, grad_col_int_in, gradient_col_interp[i+n_slices:i+2*n_slices,:,:]).wait() 
                cl.enqueue_copy(cl_queue, grad_row_int_in, gradient_row_interp[i+n_slices:i+2*n_slices,:,:]).wait() 
                cl.enqueue_copy(cl_queue, image_interp_in, image_interp[i+n_slices:i+2*n_slices,:,:]).wait() 

            cl_queue.finish()

        
        return rgc_map
