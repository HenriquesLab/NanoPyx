# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
cimport numpy as np
from .__opencl__ import cl, cl_array
from .__liquid_engine__ import LiquidEngine
from cython.parallel import prange
from .__interpolation_tools__ import check_image

cdef extern from "_c_gradients.h":
    void _c_gradient_roberts_cross(float* pixels, float* GxArray, float* GyArray, int w, int h) nogil

class GradientRobertsCross(LiquidEngine):

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "GradientRobertsCross"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing,
                        unthreaded_=True, threaded_=True, threaded_static_=True, 
                        threaded_dynamic_=True, threaded_guided_=True, opencl_=False)
        
    def run(self, image, run_type = None):
        image = check_image(image)
        return self._run(image, run_type=run_type)
    
    def benchmark(self, image):
        image = check_image(image)
        return super().benchmark(image)
    
    
    # tag-start: _le_roberts_cross_gradients.GradientRobertsCross._run_unthreaded
    def _run_unthreaded(self, float[:,:,:] image):

        cdef int nFrames = image.shape[0]
        cdef float [:,:,:] gradient_col = np.zeros_like(image) 
        cdef float [:,:,:] gradient_row = np.zeros_like(image)

        cdef int n
        with nogil: 
            for n in range(nFrames):
                _c_gradient_roberts_cross(&image[n,0,0], &gradient_col[n,0,0], &gradient_row[n,0,0], image.shape[1], image.shape[2])
        
        return gradient_col, gradient_row
    # tag-end
    

    # tag-copy: _le_roberts_cross_gradients.GradientRobertsCross._run_unthreaded; replace("_run_unthreaded", "_run_threaded"); replace("range(nFrames)", "prange(nFrames)")
    def _run_threaded(self, float[:,:,:] image):

        cdef int nFrames = image.shape[0]
        cdef float [:,:,:] gradient_col = np.zeros_like(image) 
        cdef float [:,:,:] gradient_row = np.zeros_like(image)

        cdef int n
        with nogil: 
            for n in prange(nFrames):
                _c_gradient_roberts_cross(&image[n,0,0], &gradient_col[n,0,0], &gradient_row[n,0,0], image.shape[1], image.shape[2])
        
        return gradient_col, gradient_row
    # tag-end


    # tag-copy: _le_roberts_cross_gradients.GradientRobertsCross._run_unthreaded; replace("_run_unthreaded", "_run_threaded_static"); replace("range(nFrames)", 'prange(nFrames, schedule="static")')
    def _run_threaded_static(self, float[:,:,:] image):

        cdef int nFrames = image.shape[0]
        cdef float [:,:,:] gradient_col = np.zeros_like(image) 
        cdef float [:,:,:] gradient_row = np.zeros_like(image)

        cdef int n
        with nogil: 
            for n in prange(nFrames, schedule="static"):
                _c_gradient_roberts_cross(&image[n,0,0], &gradient_col[n,0,0], &gradient_row[n,0,0], image.shape[1], image.shape[2])
        
        return gradient_col, gradient_row
    # tag-end


    # tag-copy: _le_roberts_cross_gradients.GradientRobertsCross._run_unthreaded; replace("_run_unthreaded", "_run_threaded_dynamic"); replace("range(nFrames)", 'prange(nFrames, schedule="dynamic")')
    def _run_threaded_dynamic(self, float[:,:,:] image):

        cdef int nFrames = image.shape[0]
        cdef float [:,:,:] gradient_col = np.zeros_like(image) 
        cdef float [:,:,:] gradient_row = np.zeros_like(image)

        cdef int n
        with nogil: 
            for n in prange(nFrames, schedule="dynamic"):
                _c_gradient_roberts_cross(&image[n,0,0], &gradient_col[n,0,0], &gradient_row[n,0,0], image.shape[1], image.shape[2])
        
        return gradient_col, gradient_row
    # tag-end


    # tag-copy: _le_roberts_cross_gradients.GradientRobertsCross._run_unthreaded; replace("_run_unthreaded", "_run_threaded_guided"); replace("range(nFrames)", 'prange(nFrames, schedule="guided")')
    def _run_threaded_guided(self, float[:,:,:] image):

        cdef int nFrames = image.shape[0]
        cdef float [:,:,:] gradient_col = np.zeros_like(image) 
        cdef float [:,:,:] gradient_row = np.zeros_like(image)

        cdef int n
        with nogil: 
            for n in prange(nFrames, schedule="guided"):
                _c_gradient_roberts_cross(&image[n,0,0], &gradient_col[n,0,0], &gradient_row[n,0,0], image.shape[1], image.shape[2])
        
        return gradient_col, gradient_row
    # tag-end




