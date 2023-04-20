# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np

cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport sqrt, pow
from .__liquid_engine__ import LiquidEngine
from .__interpolation_tools__ import check_image
from nanopyx.liquid import CRShiftAndMagnify

cdef extern from "_c_sr_radial_gradient_convergence.h":
    float _c_calculate_rgc(int xM, int yM, float* imIntGx, float* imIntGy, float* imInt, int w, int h, int magnification, float Gx_Gy_MAGNIFICATION, float fwhm, float tSO, float tSS, float sensitivity) nogil

cdef extern from "_c_gradients.h":
    void _c_gradient_roberts_cross(float* pixels, float* GxArray, float* GyArray, int w, int h) nogil

# cdef float Gx_Gy_MAGNIFICATION = 2.0

class RadialGradientConvergence(LiquidEngine):
    """
    Radial gradient convergence using the NanoPyx Liquid Engine
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
    
    def run(self, image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1 , doIntensityWeighting: bool = True, run_type = 2): # TODO: add the benchmark and change the run type 

        image = check_image(image)
        return self._run(image, magnification, radius, sensitivity, doIntensityWeighting, run_type=run_type)
    
    # tag-start: _le_radial_gradient_convergence.RadialGradientConvergence._run_unthreaded
    def _run_unthreaded(self, float[:,:,:] image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1 , doIntensityWeighting: bool = True):

        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float Gx_Gy_MAGNIFICATION = 2.0
        cdef int _magnification = magnification
        cdef float _sensitivity = sensitivity
        cdef int _doIntensityWeighting = doIntensityWeighting

        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]
        cdef int rowsM = <int>(rows * magnification)
        cdef int colsM = <int>(cols * magnification)

        crsm = CRShiftAndMagnify()
        cdef float [:,:,:] image_interp = crsm.run(image, 0, 0, magnification, magnification)

        cdef float [:,:,:] gradient_col = np.zeros_like(image) 
        cdef float [:,:,:] gradient_row = np.zeros_like(image)

        cdef int n
        with nogil: 
            for n in prange(nFrames):
                _c_gradient_roberts_cross(&image[n,0,0], &gradient_col[n,0,0], &gradient_row[n,0,0], image.shape[1], image.shape[2])

        cdef float [:,:,:] gradient_col_interp = crsm.run(gradient_col, 0, 0, magnification*Gx_Gy_MAGNIFICATION, magnification*Gx_Gy_MAGNIFICATION)
        cdef float [:,:,:] gradient_row_interp = crsm.run(gradient_row, 0, 0, magnification*Gx_Gy_MAGNIFICATION, magnification*Gx_Gy_MAGNIFICATION)
    
        cdef float [:,:,:] rgc_map = np.zeros((image.shape[0], image.shape[1]*magnification, image.shape[2]*magnification), dtype=np.float32)

        cdef int f, rM, cM
        with nogil:
            for f in range(nFrames):
                    for rM in range(rowsM): 
                        for cM in range(colsM):
                            if _doIntensityWeighting:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], &image_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity) * image_interp[f, rM, cM] 
                            else:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], &image_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity)
        return rgc_map
        # tag-end

    # tag-copy:  _le_radial_gradient_convergence.RadialGradientConvergence._run_unthreaded; replace("_run_unthreaded", "_run_threaded"); replace("range(rowsM)", "prange(rowsM)")
    def _run_threaded(self, float[:,:,:] image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1 , doIntensityWeighting: bool = True):

        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float Gx_Gy_MAGNIFICATION = 2.0
        cdef int _magnification = magnification
        cdef float _sensitivity = sensitivity
        cdef int _doIntensityWeighting = doIntensityWeighting

        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]
        cdef int rowsM = <int>(rows * magnification)
        cdef int colsM = <int>(cols * magnification)

        crsm = CRShiftAndMagnify()
        cdef float [:,:,:] image_interp = crsm.run(image, 0, 0, magnification, magnification)

        cdef float [:,:,:] gradient_col = np.zeros_like(image) 
        cdef float [:,:,:] gradient_row = np.zeros_like(image)

        cdef int n
        with nogil: 
            for n in prange(nFrames):
                _c_gradient_roberts_cross(&image[n,0,0], &gradient_col[n,0,0], &gradient_row[n,0,0], image.shape[1], image.shape[2])

        cdef float [:,:,:] gradient_col_interp = crsm.run(gradient_col, 0, 0, magnification*Gx_Gy_MAGNIFICATION, magnification*Gx_Gy_MAGNIFICATION)
        cdef float [:,:,:] gradient_row_interp = crsm.run(gradient_row, 0, 0, magnification*Gx_Gy_MAGNIFICATION, magnification*Gx_Gy_MAGNIFICATION)
    
        cdef float [:,:,:] rgc_map = np.zeros((image.shape[0], image.shape[1]*magnification, image.shape[2]*magnification), dtype=np.float32)

        cdef int f, rM, cM
        with nogil:
            for f in range(nFrames):
                    for rM in prange(rowsM): 
                        for cM in range(colsM):
                            if _doIntensityWeighting:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], &image_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity) * image_interp[f, rM, cM] 
                            else:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], &image_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity)
        return rgc_map
        # tag-end

    # tag-copy:  _le_radial_gradient_convergence.RadialGradientConvergence._run_unthreaded; replace("_run_unthreaded", "_run_threaded_static"); replace("range(rowsM)", 'prange(rowsM, schedule="static")')
    def _run_threaded_static(self, float[:,:,:] image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1 , doIntensityWeighting: bool = True):

        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float Gx_Gy_MAGNIFICATION = 2.0
        cdef int _magnification = magnification
        cdef float _sensitivity = sensitivity
        cdef int _doIntensityWeighting = doIntensityWeighting

        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]
        cdef int rowsM = <int>(rows * magnification)
        cdef int colsM = <int>(cols * magnification)

        crsm = CRShiftAndMagnify()
        cdef float [:,:,:] image_interp = crsm.run(image, 0, 0, magnification, magnification)

        cdef float [:,:,:] gradient_col = np.zeros_like(image) 
        cdef float [:,:,:] gradient_row = np.zeros_like(image)

        cdef int n
        with nogil: 
            for n in prange(nFrames):
                _c_gradient_roberts_cross(&image[n,0,0], &gradient_col[n,0,0], &gradient_row[n,0,0], image.shape[1], image.shape[2])

        cdef float [:,:,:] gradient_col_interp = crsm.run(gradient_col, 0, 0, magnification*Gx_Gy_MAGNIFICATION, magnification*Gx_Gy_MAGNIFICATION)
        cdef float [:,:,:] gradient_row_interp = crsm.run(gradient_row, 0, 0, magnification*Gx_Gy_MAGNIFICATION, magnification*Gx_Gy_MAGNIFICATION)
    
        cdef float [:,:,:] rgc_map = np.zeros((image.shape[0], image.shape[1]*magnification, image.shape[2]*magnification), dtype=np.float32)

        cdef int f, rM, cM
        with nogil:
            for f in range(nFrames):
                    for rM in prange(rowsM, schedule="static"): 
                        for cM in range(colsM):
                            if _doIntensityWeighting:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], &image_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity) * image_interp[f, rM, cM] 
                            else:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], &image_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity)
        return rgc_map
        # tag-end

    # tag-copy:  _le_radial_gradient_convergence.RadialGradientConvergence._run_unthreaded; replace("_run_unthreaded", "_run_threaded_dynamic"); replace("range(rowsM)", 'prange(rowsM, schedule="dynamic")')
    def _run_threaded_dynamic(self, float[:,:,:] image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1 , doIntensityWeighting: bool = True):

        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float Gx_Gy_MAGNIFICATION = 2.0
        cdef int _magnification = magnification
        cdef float _sensitivity = sensitivity
        cdef int _doIntensityWeighting = doIntensityWeighting

        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]
        cdef int rowsM = <int>(rows * magnification)
        cdef int colsM = <int>(cols * magnification)

        crsm = CRShiftAndMagnify()
        cdef float [:,:,:] image_interp = crsm.run(image, 0, 0, magnification, magnification)

        cdef float [:,:,:] gradient_col = np.zeros_like(image) 
        cdef float [:,:,:] gradient_row = np.zeros_like(image)

        cdef int n
        with nogil: 
            for n in prange(nFrames):
                _c_gradient_roberts_cross(&image[n,0,0], &gradient_col[n,0,0], &gradient_row[n,0,0], image.shape[1], image.shape[2])

        cdef float [:,:,:] gradient_col_interp = crsm.run(gradient_col, 0, 0, magnification*Gx_Gy_MAGNIFICATION, magnification*Gx_Gy_MAGNIFICATION)
        cdef float [:,:,:] gradient_row_interp = crsm.run(gradient_row, 0, 0, magnification*Gx_Gy_MAGNIFICATION, magnification*Gx_Gy_MAGNIFICATION)
    
        cdef float [:,:,:] rgc_map = np.zeros((image.shape[0], image.shape[1]*magnification, image.shape[2]*magnification), dtype=np.float32)

        cdef int f, rM, cM
        with nogil:
            for f in range(nFrames):
                    for rM in prange(rowsM, schedule="dynamic"): 
                        for cM in range(colsM):
                            if _doIntensityWeighting:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], &image_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity) * image_interp[f, rM, cM] 
                            else:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], &image_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity)
        return rgc_map
        # tag-end

    # tag-copy:  _le_radial_gradient_convergence.RadialGradientConvergence._run_unthreaded; replace("_run_unthreaded", "_run_threaded_guided"); replace("range(rowsM)", 'prange(rowsM, schedule="guided")')
    def _run_threaded_guided(self, float[:,:,:] image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1 , doIntensityWeighting: bool = True):

        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float Gx_Gy_MAGNIFICATION = 2.0
        cdef int _magnification = magnification
        cdef float _sensitivity = sensitivity
        cdef int _doIntensityWeighting = doIntensityWeighting

        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]
        cdef int rowsM = <int>(rows * magnification)
        cdef int colsM = <int>(cols * magnification)

        crsm = CRShiftAndMagnify()
        cdef float [:,:,:] image_interp = crsm.run(image, 0, 0, magnification, magnification)

        cdef float [:,:,:] gradient_col = np.zeros_like(image) 
        cdef float [:,:,:] gradient_row = np.zeros_like(image)

        cdef int n
        with nogil: 
            for n in prange(nFrames):
                _c_gradient_roberts_cross(&image[n,0,0], &gradient_col[n,0,0], &gradient_row[n,0,0], image.shape[1], image.shape[2])

        cdef float [:,:,:] gradient_col_interp = crsm.run(gradient_col, 0, 0, magnification*Gx_Gy_MAGNIFICATION, magnification*Gx_Gy_MAGNIFICATION)
        cdef float [:,:,:] gradient_row_interp = crsm.run(gradient_row, 0, 0, magnification*Gx_Gy_MAGNIFICATION, magnification*Gx_Gy_MAGNIFICATION)
    
        cdef float [:,:,:] rgc_map = np.zeros((image.shape[0], image.shape[1]*magnification, image.shape[2]*magnification), dtype=np.float32)

        cdef int f, rM, cM
        with nogil:
            for f in range(nFrames):
                    for rM in prange(rowsM, schedule="guided"): 
                        for cM in range(colsM):
                            if _doIntensityWeighting:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], &image_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity) * image_interp[f, rM, cM] 
                            else:
                                rgc_map[f, rM, cM] = _c_calculate_rgc(cM, rM, &gradient_col_interp[f,0,0], &gradient_row_interp[f,0,0], &image_interp[f,0,0], colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION,  fwhm, tSO, tSS, _sensitivity)
        return rgc_map
        # tag-end

    




