# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
import time


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
        self._default_benchmarks = {'OpenCL': {"(['shape(100, 100)', 'shape(5, 5)'], {})": [250000, 0.009790166019229218, 0.003298707975773141, 0.003652999992482364], "(['shape(500, 500)', 'shape(11, 11)'], {})": [30250000, 0.007485667010769248, 0.007351625012233853, 0.007932541979243979]}, 'Threaded': {"(['shape(100, 100)', 'shape(5, 5)'], {})": [250000, 0.00022562500089406967, 0.00018245799583382905, 0.00017095799557864666], "(['shape(500, 500)', 'shape(11, 11)'], {})": [30250000, 0.007719500019447878, 0.0060990000201854855, 0.008289082994451746]}, 'Threaded_dynamic': {"(['shape(100, 100)', 'shape(5, 5)'], {})": [250000, 0.0001590839819982648, 0.0001595419889781624, 0.00020799998310394585], "(['shape(500, 500)', 'shape(11, 11)'], {})": [30250000, 0.005388707999372855, 0.005681082984665409, 0.005188624985748902]}, 'Threaded_guided': {"(['shape(100, 100)', 'shape(5, 5)'], {})": [250000, 0.00018945799092762172, 0.0001723750028759241, 0.00017620899598114192], "(['shape(500, 500)', 'shape(11, 11)'], {})": [30250000, 0.005136792024131864, 0.005925459001446143, 0.005063916993094608]}, 'Threaded_static': {"(['shape(100, 100)', 'shape(5, 5)'], {})": [250000, 0.00018066700431518257, 0.0001790839887689799, 0.00020249999943189323], "(['shape(500, 500)', 'shape(11, 11)'], {})": [30250000, 0.005990457982989028, 0.006161083991173655, 0.006471291999332607]}, 'Unthreaded': {"(['shape(100, 100)', 'shape(5, 5)'], {})": [250000, 0.00021341597312130034, 0.00020179100101813674, 0.00020345899974927306], "(['shape(500, 500)', 'shape(11, 11)'], {})": [30250000, 0.031798416981473565, 0.031586291996063665, 0.031558375019812956]}}

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
        
        # QUEUE AND CONTEXT
        cl_ctx = cl.Context([device['device']])
        dc = device['device']
        cl_queue = cl.CommandQueue(cl_ctx)

        image_out = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        mf = cl.mem_flags

        input_image = cl.image_from_array(cl_ctx, image, mode='r')
        input_kernel = cl.image_from_array(cl_ctx, kernel, mode='r')
        output_opencl = cl.image_from_array(cl_ctx, image_out, mode='w')
        cl_queue.finish()
        
        code = self._get_cl_code("_le_convolution.cl", device['DP'])
        prg = cl.Program(cl_ctx, code).build()
        knl = prg.conv2d

        knl(cl_queue,
            (1,image.shape[0], image.shape[1]), 
            self.get_work_group(device['device'],(1,image.shape[0], image.shape[1])),
            input_image, 
            output_opencl, 
            input_kernel).wait() 

        cl.enqueue_copy(cl_queue, image_out, output_opencl,origin=(0,0), region=(image.shape[0], image.shape[1])).wait() 

        cl_queue.finish()
        return image_out