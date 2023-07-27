import numpy as np

cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport cos, sin

from .__interpolation_tools__ import check_image, value2array
from .__liquid_engine__ import LiquidEngine
from .__opencl__ import cl, cl_array


class Convolution(LiquidEngine):
    """
    2D convolution
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "Conv2D"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, 
                        opencl_=True, unthreaded_=True, threaded_=True, threaded_static_=True, 
                        threaded_dynamic_=True, threaded_guided_=True)

    def run(self, image, kernel, run_type=None):
        return self._run(image, kernel, run_type=run_type)

    def benchmark(self, image, kernel):
        return super().benchmark(image, kernel)

    def _run_unthreaded(self, float[:,:] image, float[:,:] kernel):

        cdef int nRows = image.shape[0]
        cdef int nCols = image.shape[1]

        cdef int nRows_kernel = kernel.shape[0]
        cdef int nCols_kernel = kernel.shape[1]

        cdef int center_r = (nRows_kernel-1) // 2
        cdef int center_c = (nCols_kernel-1) // 2

        cdef int r,c
        cdef int kr,kc

        cdef int local_row, local_col

        cdef float acc = 0


        conv_out = np.zeros((nRows, nCols), dtype=np.float32)
        cdef float[:,:] _conv_out = conv_out

        with nogil:
            for r in range(nRows):
                for c in range(nCols):
                    acc = 0
                    for kr in range(nRows_kernel):
                        for kc in range(nCols_kernel):
                            local_row = min(max(r+(kr-center_r),0),nRows-1)
                            local_col = min(max(c+(kc-center_c),0),nCols-1)
                            acc = acc + kernel[kr,kc] * image[local_row, local_col]
                    _conv_out[r,c] = acc

        return conv_out

    def _run_threaded(self, float[:,:] image, float[:,:] kernel):

        cdef int nRows = image.shape[0]
        cdef int nCols = image.shape[1]

        cdef int nRows_kernel = kernel.shape[0]
        cdef int nCols_kernel = kernel.shape[1]

        cdef int center_r = (nRows_kernel-1) // 2
        cdef int center_c = (nCols_kernel-1) // 2

        cdef int r,c
        cdef int kr,kc

        cdef int local_row, local_col

        cdef float acc = 0

        conv_out = np.zeros((nRows, nCols), dtype=np.float32)
        cdef float[:,:] _conv_out = conv_out

        with nogil:
            for r in prange(nRows):
                for c in prange(nCols):
                    acc = 0
                    for kr in range(nRows_kernel):
                        for kc in range(nCols_kernel):
                            local_row = min(max(r+(kr-center_r),0),nRows-1)
                            local_col = min(max(c+(kc-center_c),0),nCols-1)
                            acc = acc + kernel[kr,kc] * image[local_row, local_col]
                    _conv_out[r,c] = acc

        return conv_out

    def _run_threaded_static(self, float[:,:] image, float[:,:] kernel):

        cdef int nRows = image.shape[0]
        cdef int nCols = image.shape[1]

        cdef int nRows_kernel = kernel.shape[0]
        cdef int nCols_kernel = kernel.shape[1]

        cdef int center_r = (nRows_kernel-1) // 2
        cdef int center_c = (nCols_kernel-1) // 2

        cdef int r,c
        cdef int kr,kc

        cdef int local_row, local_col

        cdef float acc = 0

        conv_out = np.zeros((nRows, nCols), dtype=np.float32)
        cdef float[:,:] _conv_out = conv_out

        with nogil:
            for r in prange(nRows, schedule="static"):
                for c in prange(nCols, schedule="static"):
                    acc = 0
                    for kr in range(nRows_kernel):
                        for kc in range(nCols_kernel):
                            local_row = min(max(r+(kr-center_r),0),nRows-1)
                            local_col = min(max(c+(kc-center_c),0),nCols-1)
                            acc = acc + kernel[kr,kc] * image[local_row, local_col]
                    _conv_out[r,c] = acc

        return conv_out

    def _run_threaded_dynamic(self, float[:,:] image, float[:,:] kernel):

        cdef int nRows = image.shape[0]
        cdef int nCols = image.shape[1]

        cdef int nRows_kernel = kernel.shape[0]
        cdef int nCols_kernel = kernel.shape[1]

        cdef int center_r = (nRows_kernel-1) // 2
        cdef int center_c = (nCols_kernel-1) // 2

        cdef int r,c
        cdef int kr,kc

        cdef int local_row, local_col

        cdef float acc = 0

        conv_out = np.zeros((nRows, nCols), dtype=np.float32)
        cdef float[:,:] _conv_out = conv_out

        with nogil:
            for r in prange(nRows, schedule="dynamic"):
                for c in prange(nCols, schedule="dynamic"):
                    acc = 0
                    for kr in range(nRows_kernel):
                        for kc in range(nCols_kernel):
                            local_row = min(max(r+(kr-center_r),0),nRows-1)
                            local_col = min(max(c+(kc-center_c),0),nCols-1)
                            acc = acc + kernel[kr,kc] * image[local_row, local_col]
                    _conv_out[r,c] = acc

        return conv_out

    def _run_threaded_guided(self, float[:,:] image, float[:,:] kernel):

        cdef int nRows = image.shape[0]
        cdef int nCols = image.shape[1]

        cdef int nRows_kernel = kernel.shape[0]
        cdef int nCols_kernel = kernel.shape[1]

        cdef int center_r = (nRows_kernel-1) // 2
        cdef int center_c = (nCols_kernel-1) // 2

        cdef int r,c
        cdef int kr,kc

        cdef int local_row, local_col

        cdef float acc = 0

        conv_out = np.zeros((nRows, nCols), dtype=np.float32)
        cdef float[:,:] _conv_out = conv_out

        with nogil:
            for r in prange(nRows, schedule="guided"):
                for c in prange(nCols, schedule="guided"):
                    acc = 0
                    for kr in range(nRows_kernel):
                        for kc in range(nCols_kernel):
                            local_row = min(max(r+(kr-center_r),0),nRows-1)
                            local_col = min(max(c+(kc-center_c),0),nCols-1)
                            acc = acc + kernel[kr,kc] * image[local_row, local_col]
                    _conv_out[r,c] = acc

        return conv_out


    def _run_opencl(self, image, kernel, device):

        print(image.shape)
        
        # QUEUE AND CONTEXT
        cl_ctx = cl.Context([device['device']])
        dc = device['device']
        cl_queue = cl.CommandQueue(cl_ctx)

        nRows_kernel = kernel.shape[0]
        nCols_kernel = kernel.shape[1]

        center_r = (nRows_kernel-1) // 2
        center_c = (nCols_kernel-1) // 2

        image_out = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        mf = cl.mem_flags

        input_image = cl.Buffer(cl_ctx, mf.READ_ONLY, image.nbytes)
        cl.enqueue_copy(cl_queue, input_image, image).wait()

        input_kernel = cl.Buffer(cl_ctx, mf.READ_ONLY, kernel.nbytes)
        cl.enqueue_copy(cl_queue, input_kernel, kernel).wait()

        output_opencl = cl.Buffer(cl_ctx, mf.WRITE_ONLY, image_out.nbytes)

        code = self._get_cl_code("_le_convolution.cl", device['DP'])
        prg = cl.Program(cl_ctx, code).build()
        knl = prg.conv2d

        knl(cl_queue,
            (image.shape[0], image.shape[1]), 
            None, 
            input_image, 
            output_opencl, 
            input_kernel,
            np.int32(nRows_kernel), 
            np.int32(nCols_kernel), 
            np.int32(center_r), 
            np.int32(center_c)).wait() 

        cl.enqueue_copy(cl_queue, image_out, output_opencl).wait() 

        cl_queue.finish()

        return image_out