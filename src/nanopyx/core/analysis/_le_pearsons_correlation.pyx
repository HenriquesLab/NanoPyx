# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
import time


cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport cos, sin

from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array


class PearsonsCorrelation(LiquidEngine):
    """
    Pearson's Correlation of two dimensional arrays using the Liquid Engine"""

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "PearsonsCorrelation"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, 
                        opencl_=True, unthreaded_=True, threaded_=True, threaded_static_=True, 
                        threaded_dynamic_=True, threaded_guided_=True, python_=True)

    def run(self, array_1, array_2, run_type=None):
        return self._run(array_1, array_2, run_type=run_type)

    def benchmark(self, array_1, array_2):
        return super().benchmark(array_1, array_2)

    def _compare_runs(self, output_1, output_2):

        if (abs(output_1-output_2)) < abs(output_1*0.05):
            return True
        else:
            return False

    def _run_python(self, im1, im2):
        w = im1.shape[1]
        h = im1.shape[0]
        wh = w*h

        mean_im1 = 0.0
        mean_im2 = 0.0
        sum_im12 = 0.0
        sum_im11 = 0.0
        sum_im22 = 0.0

        for j in range(h):
            for i in range(w):
                mean_im1 += im1[j, i]
                mean_im2 += im2[j, i]
        mean_im1 /= wh
        mean_im2 /= wh
        for j in range(h):
            for i in range(w):
                d_im1 = im1[j, i] - mean_im1
                d_im2 = im2[j, i] - mean_im2
                sum_im12 += d_im1 * d_im2
                sum_im11 += d_im1 * d_im1
                sum_im22 += d_im2 * d_im2
        if sum_im11 == 0 or sum_im22 == 0:
            return 0
        else:
            return sum_im12 / (sum_im11 * sum_im22)**0.5

    def _run_unthreaded(self, float[:, :] im1, float[:, :] im2):
        cdef int w = im1.shape[1]
        cdef int h = im1.shape[0]
        cdef int wh = w*h

        cdef float mean_im1 = 0.0
        cdef float mean_im2 = 0.0
        cdef float sum_im12 = 0.0
        cdef float sum_im11 = 0.0
        cdef float sum_im22 = 0.0

        cdef int i, j
        cdef float d_im1, d_im2

        for j in range(h):
            for i in range(w):
                mean_im1 += im1[j, i]
                mean_im2 += im2[j, i]
        mean_im1 /= wh
        mean_im2 /= wh
        for j in range(h):
            for i in range(w):
                d_im1 = im1[j, i] - mean_im1
                d_im2 = im2[j, i] - mean_im2
                sum_im12 += d_im1 * d_im2
                sum_im11 += d_im1 * d_im1
                sum_im22 += d_im2 * d_im2
        if sum_im11 == 0 or sum_im22 == 0:
            return 0
        else:
            return sum_im12 / (sum_im11 * sum_im22)**0.5

    def _run_threaded(self, float[:, :] im1, float[:, :] im2):
        
        cdef int w = im1.shape[1]
        cdef int h = im1.shape[0]
        cdef int wh = w*h

        cdef float mean_im1 = 0.0
        cdef float mean_im2 = 0.0
        cdef float sum_im12 = 0.0
        cdef float sum_im11 = 0.0
        cdef float sum_im22 = 0.0

        cdef int i, j
        cdef float d_im1, d_im2
        cdef float def_val = 0.0

        with nogil:
            for j in prange(h):
                for i in range(w):
                    mean_im1 += im1[j, i]
                    mean_im2 += im2[j, i]
        mean_im1 /= wh
        mean_im2 /= wh
        with nogil:
            for j in prange(h):
                for i in range(w):
                    d_im1 = im1[j, i] - mean_im1
                    d_im2 = im2[j, i] - mean_im2
                    sum_im12 += d_im1 * d_im2
                    sum_im11 += d_im1 * d_im1
                    sum_im22 += d_im2 * d_im2
        if sum_im11 == 0 or sum_im22 == 0:
            return def_val
        else:
            return sum_im12 / (sum_im11 * sum_im22)**0.5

    def _run_threaded_guided(self, float[:, :] im1, float[:, :] im2):
        
        cdef int w = im1.shape[1]
        cdef int h = im1.shape[0]
        cdef int wh = w*h

        cdef float mean_im1 = 0.0
        cdef float mean_im2 = 0.0
        cdef float sum_im12 = 0.0
        cdef float sum_im11 = 0.0
        cdef float sum_im22 = 0.0

        cdef int i, j
        cdef float d_im1, d_im2
        cdef float def_val = 0.0

        with nogil:
            for j in prange(w, schedule="guided"):
                for i in range(w):
                    mean_im1 += im1[j, i]
                    mean_im2 += im2[j, i]
        mean_im1 /= wh
        mean_im2 /= wh
        with nogil:
            for j in prange(w, schedule="guided"):
                for i in range(w):
                    d_im1 = im1[j, i] - mean_im1
                    d_im2 = im2[j, i] - mean_im2
                    sum_im12 += d_im1 * d_im2
                    sum_im11 += d_im1 * d_im1
                    sum_im22 += d_im2 * d_im2
        if sum_im11 == 0 or sum_im22 == 0:
            return def_val
        else:
            return sum_im12 / (sum_im11 * sum_im22)**0.5

    def _run_threaded_dynamic(self, float[:, :] im1, float[:, :] im2):
        
        cdef int w = im1.shape[1]
        cdef int h = im1.shape[0]
        cdef int wh = w*h

        cdef float mean_im1 = 0.0
        cdef float mean_im2 = 0.0
        cdef float sum_im12 = 0.0
        cdef float sum_im11 = 0.0
        cdef float sum_im22 = 0.0

        cdef int i, j
        cdef float d_im1, d_im2
        cdef float def_val = 0.0

        with nogil:
            for j in prange(w, schedule="dynamic"):
                for i in range(w):
                    mean_im1 += im1[j, i]
                    mean_im2 += im2[j, i]
        mean_im1 /= wh
        mean_im2 /= wh
        with nogil:
            for j in prange(w, schedule="dynamic"):
                for i in range(w):
                    d_im1 = im1[j, i] - mean_im1
                    d_im2 = im2[j, i] - mean_im2
                    sum_im12 += d_im1 * d_im2
                    sum_im11 += d_im1 * d_im1
                    sum_im22 += d_im2 * d_im2
        if sum_im11 == 0 or sum_im22 == 0:
            return def_val
        else:
            return sum_im12 / (sum_im11 * sum_im22)**0.5

    def _run_threaded_static(self, float[:, :] im1, float[:, :] im2):
        
        cdef int w = im1.shape[1]
        cdef int h = im1.shape[0]
        cdef int wh = w*h

        cdef float mean_im1 = 0.0
        cdef float mean_im2 = 0.0
        cdef float sum_im12 = 0.0
        cdef float sum_im11 = 0.0
        cdef float sum_im22 = 0.0

        cdef int i, j
        cdef float d_im1, d_im2
        cdef float def_val = 0.0

        with nogil:
            for j in prange(w, schedule="static"):
                for i in range(w):
                    mean_im1 += im1[j, i]
                    mean_im2 += im2[j, i]
        mean_im1 /= wh
        mean_im2 /= wh
        with nogil:
            for j in prange(w, schedule="static"):
                for i in range(w):
                    d_im1 = im1[j, i] - mean_im1
                    d_im2 = im2[j, i] - mean_im2
                    sum_im12 += d_im1 * d_im2
                    sum_im11 += d_im1 * d_im1
                    sum_im22 += d_im2 * d_im2
        if sum_im11 == 0 or sum_im22 == 0:
            return def_val
        else:
            return sum_im12 / (sum_im11 * sum_im22)**0.5


    def _run_opencl(self, im1, im2, device):

        cl_ctx = cl.Context([device['device']])
        dc = device['device']
        cl_queue = cl.CommandQueue(cl_ctx)

        mf = cl.mem_flags

        sum11 = np.empty_like(im1)
        sum12 = np.empty_like(im1)
        sum22 = np.empty_like(im1)

        input_array_1 = cl.Buffer(cl_ctx, mf.READ_ONLY, im1.nbytes)
        input_array_2 = cl.Buffer(cl_ctx, mf.READ_ONLY, im2.nbytes)
        sum11_buffer = cl.Buffer(cl_ctx, mf.WRITE_ONLY, im1.nbytes)
        sum12_buffer = cl.Buffer(cl_ctx, mf.WRITE_ONLY, im1.nbytes)
        sum22_buffer = cl.Buffer(cl_ctx, mf.WRITE_ONLY, im1.nbytes)

        cl.enqueue_copy(cl_queue, input_array_1, im1).wait()
        cl.enqueue_copy(cl_queue, input_array_2, im2).wait()

        code = self._get_cl_code("_le_pearsons_correlation_.cl", device['DP'])
        prg = cl.Program(cl_ctx, code).build()
        knl = prg.pearsons_correlation

        knl(
            cl_queue,
            (im1.shape[0], im1.shape[1]),
            None,
            input_array_1,
            input_array_2,
            sum11_buffer,
            sum12_buffer,
            sum22_buffer,
            np.float32(np.mean(im1)),
            np.float32(np.mean(im2))
        ).wait()

        cl.enqueue_copy(cl_queue, sum11, sum11_buffer)
        cl.enqueue_copy(cl_queue, sum12, sum12_buffer)
        cl.enqueue_copy(cl_queue, sum22, sum22_buffer)

        cl_queue.finish()

        input_array_1.release()
        input_array_2.release()
        sum11_buffer.release()
        sum12_buffer.release()
        sum22_buffer.release()

        return np.sum(sum12) / (np.sum(sum11) * np.sum(sum22))**0.5