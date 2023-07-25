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
                        threaded_dynamic_=True, threaded_guided_=True, opencl_=True)
        
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

    def _run_opencl(self, float[:,:,:] image, dict device):

        # QUEUE AND CONTEXT
        cl_ctx = cl.Context([device['device']])
        dc = device['device']
        cl_queue = cl.CommandQueue(cl_ctx)

        # Swap row and columns because opencl is strange and stores the
        # array in a buffer in fortran ordering despite the original
        # numpy array being in C order.
        #image = np.ascontiguousarray(np.swapaxes(image, 1, 2), dtype=np.float32)

        cdef int nFrames = image.shape[0]
        cdef int row = image.shape[1]
        cdef int col = image.shape[2]
        cdef float [:,:,:] gradient_col = np.zeros_like(image) 
        cdef float [:,:,:] gradient_row = np.zeros_like(image)

        max_slices = int((dc.global_mem_size // (image[0,:,:].nbytes + gradient_col[0,:,:].nbytes + gradient_row[0,:,:].nbytes))/3)
        # TODO add exception if max_slices < 1 

        mf = cl.mem_flags

        input_opencl = cl.Buffer(cl_ctx, mf.READ_ONLY, image[0:max_slices,:,:].nbytes)
        cl.enqueue_copy(cl_queue, input_opencl, image[0:max_slices,:,:]).wait()
        output_opencl_col = cl.Buffer(cl_ctx, mf.WRITE_ONLY, gradient_col[0:max_slices,:,:].nbytes)
        output_opencl_row = cl.Buffer(cl_ctx, mf.WRITE_ONLY, gradient_row[0:max_slices, :, :].nbytes)

        code = self._get_cl_code("_le_roberts_cross_gradients.cl", device['DP'])
        prg = cl.Program(cl_ctx, code).build()
        knl = prg.gradient_roberts_cross

        for i in range(0, image.shape[0]-1, max_slices):
            if image.shape[0] - i >= max_slices:
                n_slices = max_slices
            else:
                n_slices = image.shape[0] - i
            knl(cl_queue,
                (n_slices,), 
                None, 
                input_opencl, 
                output_opencl_col,
                output_opencl_row, 
                np.int32(row), 
                np.int32(col)).wait() 

            cl.enqueue_copy(cl_queue, gradient_col[i:i+n_slices,:,:], output_opencl_col).wait()
            cl.enqueue_copy(cl_queue, gradient_row[i:i+n_slices,:,:], output_opencl_row).wait()  
            if i<=image.shape[0]-max_slices:
                cl.enqueue_copy(cl_queue, input_opencl, image[i+n_slices:i+2*n_slices,:,:]).wait() 

            cl_queue.finish()

        input_opencl.release()
        output_opencl_col.release()
        output_opencl_row.release()
        # Swap rows and columns back
        #return np.ascontiguousarray(np.swapaxes(gradient_col, 1, 2), dtype=np.float32), np.ascontiguousarray(np.swapaxes(gradient_row, 1, 2), dtype=np.float32)
        return gradient_col, gradient_row