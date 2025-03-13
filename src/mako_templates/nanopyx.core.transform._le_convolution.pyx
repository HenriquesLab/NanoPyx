<%!
schedulers = ['unthreaded','threaded','threaded_guided','threaded_dynamic','threaded_static']
%># cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
import time


cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport cos, sin

from .__interpolation_tools__ import check_image, value2array
from .convolution import convolution2D_cuda, convolution2D_dask, convolution2D_numba, convolution2D_python, convolution2D_transonic
from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array, _fastest_device


class Convolution(LiquidEngine):
    """
    2D convolution
    """

    def __init__(self, clear_benchmarks=False, testing=False, verbose=True):
        self._designation = "Conv2D"
        super().__init__(
            clear_benchmarks=clear_benchmarks, testing=testing, verbose=verbose)
        
    def run(self, image, kernel, run_type=None):
        image = check_image(image)
        return self._run(image, kernel, run_type=run_type)

    def benchmark(self, image, kernel):
        image = check_image(image)
        return super().benchmark(image, kernel)

    % for sch in schedulers:
    def _run_${sch}(self, float[:,:,:] image, float[:,:] kernel):
        """
        @cpu
        % if sch!='unthreaded':
        @threaded
        % endif
        @cython
        """
        cdef int nFrames = image.shape[0]
        cdef int nRows = image.shape[1]
        cdef int nCols = image.shape[2]

        cdef int nRows_kernel = kernel.shape[0]
        cdef int nCols_kernel = kernel.shape[1]

        cdef int center_r = (nRows_kernel-1) // 2
        cdef int center_c = (nCols_kernel-1) // 2

        cdef int r,c
        cdef int kr,kc

        cdef int local_row, local_col

        cdef float acc = 0


        conv_out = np.zeros((nFrames, nRows, nCols), dtype=np.float32)
        cdef float[:,:,:] _conv_out = conv_out

        with nogil:
            for f in range(nFrames):
                % if sch=='unthreaded':
                for r in range(nRows):
                    for c in range(nCols):
                % elif sch=='threaded':
                for r in prange(nRows):
                    for c in prange(nCols):
                % else:
                for r in prange(nRows,schedule="${sch.split('_')[1]}"):
                    for c in prange(nCols,schedule="${sch.split('_')[1]}"):
                % endif
                        acc = 0
                        for kr in range(nRows_kernel):
                            for kc in range(nCols_kernel):
                                local_row = min(max(r+(kr-center_r),0),nRows-1)
                                local_col = min(max(c+(kc-center_c),0),nCols-1)
                                acc = acc + kernel[kr,kc] * image[f,local_row, local_col]
                        _conv_out[f,r,c] = acc

        return conv_out

    % endfor

    def _run_opencl(self, image, kernel, device=None):
        """
        @gpu
        """
        if device is None:
            device = _fastest_device

        # QUEUE AND CONTEXT
        cl_ctx = cl.Context([device['device']])
        dc = device['device']
        cl_queue = cl.CommandQueue(cl_ctx)

        image_out = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
        mf = cl.mem_flags

        input_image = cl.Buffer(cl_ctx, mf.READ_ONLY, image.nbytes)
        cl.enqueue_copy(cl_queue, input_image, image).wait()

        input_kernel = cl.Buffer(cl_ctx, mf.READ_ONLY, kernel.nbytes)
        cl.enqueue_copy(cl_queue, input_kernel, kernel).wait()

        output_opencl = cl.Buffer(cl_ctx, mf.WRITE_ONLY, image_out.nbytes)

        kernelsize = kernel.shape[0]
        cl_queue.finish()
        
        code = self._get_cl_code("_le_convolution.cl", device['DP'])
        prg = cl.Program(cl_ctx, code).build()
        knl = prg.conv2d_2

        knl(cl_queue,
            (image.shape[0],image.shape[1],image.shape[2]), 
            None,#self.get_work_group(device['device'],(image.shape[0], image.shape[1], image.shape[2])),
            input_image, 
            output_opencl, 
            input_kernel,
            np.int32(kernelsize)).wait() 

        cl.enqueue_copy(cl_queue, image_out, output_opencl).wait() 

        cl_queue.finish()
        return image_out

    def _run_python(self, image, kernel):
        """
        @cpu
        """
        return convolution2D_python(image, kernel).astype(np.float32)

    def _run_transonic(self, image, kernel):
        """
        @cpu
        @threaded
        """
        return convolution2D_transonic(image, kernel).astype(np.float32)

    def _run_dask(self, image, kernel):
        """
        @cpu
        @threaded
        """
        return convolution2D_dask(image, kernel).astype(np.float32)

    def _run_cuda(self, image, kernel):
        """
        @gpu
        """
        return convolution2D_cuda(image, kernel).astype(np.float32)

    def _run_numba(self, image, kernel):
        """
        @cpu
        @threaded
        @numba
        """
        return convolution2D_numba(image, kernel).astype(np.float32)
