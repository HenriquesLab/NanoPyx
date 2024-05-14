<%!
schedulers = ['threaded','threaded_guided','threaded_dynamic','threaded_static']
%># cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
cimport numpy as np
from ...__opencl__ import cl, cl_array, _fastest_device
from ...__liquid_engine__ import LiquidEngine

from cython.parallel import prange
from .__interpolation_tools__ import check_image

cdef extern from "_c_gradients.h":
    void _c_gradient_roberts_cross(float* pixels, float* GxArray, float* GyArray, int w, int h) nogil

class GradientRobertsCross(LiquidEngine):

    def __init__(self, clear_benchmarks=False, testing=False, verbose=True):
        self._designation = "GradientRobertsCross"
        super().__init__(
            clear_benchmarks=clear_benchmarks, testing=testing,
            verbose=verbose)

    def run(self, image, run_type = None):
        image = check_image(image)
        return self._run(image, run_type=run_type)
    
    def benchmark(self, image):
        image = check_image(image)
        return super().benchmark(image)
    
    def _run_unthreaded(self, float[:,:,:] image):

        cdef int nFrames = image.shape[0]
        cdef float [:,:,:] gradient_col = np.zeros_like(image) 
        cdef float [:,:,:] gradient_row = np.zeros_like(image)

        cdef int n
        with nogil: 
            for n in range(nFrames):
                _c_gradient_roberts_cross(&image[n,0,0], &gradient_col[n,0,0], &gradient_row[n,0,0], image.shape[1], image.shape[2])
        
        return gradient_col, gradient_row
    
    % for sch in schedulers:
    def _run_${sch}(self, float[:,:,:] image):

        cdef int nFrames = image.shape[0]
        cdef float [:,:,:] gradient_col = np.zeros_like(image) 
        cdef float [:,:,:] gradient_row = np.zeros_like(image)

        cdef int n
        with nogil:
            % if sch=='threaded':
            for n in prange(nFrames):
            % else:
            for n in prange(nFrames, schedule="${sch.split('_')[1]}"):
            % endif
                _c_gradient_roberts_cross(&image[n,0,0], &gradient_col[n,0,0], &gradient_row[n,0,0], image.shape[1], image.shape[2])
        
        return gradient_col, gradient_row
    % endfor

    def _run_opencl(self, float[:,:,:] image, dict device=None, int mem_div=1):

        if device is None:
            device = _fastest_device

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

        max_slices = int((dc.global_mem_size // (image[0,:,:].nbytes + gradient_col[0,:,:].nbytes + gradient_row[0,:,:].nbytes))/mem_div)
        max_slices = self._check_max_slices(image, max_slices) 

        mf = cl.mem_flags

        input_opencl = cl.Buffer(cl_ctx, mf.READ_ONLY, self._check_max_buffer_size(image[0:max_slices,:,:].nbytes, device['device'], max_slices))
        output_opencl_col = cl.Buffer(cl_ctx, mf.WRITE_ONLY, self._check_max_buffer_size(gradient_col[0:max_slices,:,:].nbytes, device['device'] , max_slices))
        output_opencl_row = cl.Buffer(cl_ctx, mf.WRITE_ONLY, self._check_max_buffer_size(gradient_row[0:max_slices, :, :].nbytes, device['device'], max_slices))

        cl.enqueue_copy(cl_queue, input_opencl, image[0:max_slices,:,:]).wait()

        code = self._get_cl_code("_le_roberts_cross_gradients.cl", device['DP'])
        prg = cl.Program(cl_ctx, code).build()
        knl = prg.gradient_roberts_cross

        for i in range(0, image.shape[0], max_slices):
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
            if i+n_slices<image.shape[0]:
                cl.enqueue_copy(cl_queue, input_opencl, image[i+n_slices:i+2*n_slices,:,:]).wait() 

            cl_queue.finish()

        input_opencl.release()
        output_opencl_col.release()
        output_opencl_row.release()

        return gradient_col, gradient_row