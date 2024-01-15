<%!
schedulers = ['unthreaded','threaded','threaded_guided','threaded_dynamic','threaded_static']
%># cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import time
import numpy as np
cimport numpy as np

from skimage.restoration import denoise_nl_means
from matplotlib import pyplot as plt

from libc.math cimport exp, isnan, fmax
from cython.parallel import parallel, prange

from .__interpolation_tools__ import check_image
from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array


cdef extern from "_c_patch_distance.h":
    float _c_patch_distance(float* image, float* integral, float* w, int patch_size, int iglobal, int jglobal,  int n_col, float var) nogil


class NLMDenoising(LiquidEngine):
    """
    Non-local means Denoising using the NanoPyx Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "NLMDenoising_pixel"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing,
                        unthreaded_=True, threaded_=True, threaded_static_=True,
                        threaded_dynamic_=True, threaded_guided_=True, opencl_=True,
                        python_=True)

    def run(self, np.ndarray image, int patch_size=7, int patch_distance=11, float h=0.1, float sigma=0.0, run_type=None) -> np.ndarray:
        """
        Run the non-local means denoising algorithm
        :param image: np.ndarray or memoryview
        :param patch_size: int, default 7, the size of the patch
        :param patch_distance: int, default 11, the maximum patch distance
        :param h: float, default 0.1, Cut-off distance (in gray levels).
        :param sigma: float, default 0.0, the standard deviation of the gaussian kernel
        :return: np.ndarray
        """

        image = check_image(image)

        return self._run(image, patch_size=patch_size, patch_distance=patch_distance, h=h, sigma=sigma, run_type=run_type)

    def benchmark(self, np.ndarray image, int patch_size=7, int patch_distance=11, float h=0.1, float sigma=0.0, run_type=None):
        """
        Run the non-local means denoising algorithm
        :param image: np.ndarray or memoryview
        :param patch_size: int, default 7, the size of the patch
        :param patch_distance: int, default 11, the maximum patch distance
        :param h: float, default 0.1, Cut-off distance (in gray levels).
        :param sigma: float, default 0.0, the standard deviation of the gaussian kernel
        :return: The benchmark results
        :rtype: [[run_time, run_type_name, return_value], ...]
        """

        image = check_image(image)
        
        return super().benchmark(image, patch_size=patch_size, patch_distance=patch_distance, h=h, sigma=sigma) 

    def _run_python(self, np.ndarray image, int patch_size=7, int patch_distance=11, float h=0.1, float sigma=0.0) -> np.ndarray:
        out = np.zeros_like(image)
        for i in range(image.shape[0]):
            out[i] = denoise_nl_means(image[i], patch_size=patch_size, patch_distance=patch_distance, h=h, sigma=sigma, fast_mode=False)

        return np.squeeze(out)

    % for sch in schedulers:
    def _run_${sch}(self, float[:, :, :] image, int patch_size=7, int patch_distance=11, float h=0.1, float sigma=0.0) -> np.ndarray:
        if patch_size % 2 == 0:
            patch_size = patch_size + 1  # odd value for symmetric patch

        cdef int n_frames, n_row, n_col,
        n_frames, n_row, n_col = image.shape[0], image.shape[1], image.shape[2]
        cdef int offset = patch_size / 2
        cdef int row, col, i, j, i_start, i_end, j_start, j_end

        cdef float[:, :, :] padded = np.ascontiguousarray(
            np.pad(
                image,
                ((0, 0), (offset, offset), (offset, offset)),
                mode='reflect'
            ).astype(np.float32))
        cdef float[:,:,:] result = np.zeros_like(image)
        cdef float weight_sum, weight

        cdef float A = ((patch_size - 1.) / 4.)
        cdef float[:] range_vals = np.arange(-offset, offset + 1, dtype=np.float32)
        xg_row, xg_col = np.meshgrid(range_vals, range_vals, indexing='ij')
        cdef float[:,:] w = np.ascontiguousarray(
            np.exp(-(xg_row * xg_row + xg_col * xg_col) / (2 * A * A)))
        w = w * (1. / (np.sum(w) * h * h))

        cdef float var = 2*(sigma * sigma)
        cdef float new_value
        
        with nogil:
            for f in range(n_frames):
                % if sch=='unthreaded':
                for row in range(n_row):
                % elif sch=='threaded':
                for row in prange(n_row):
                % else:
                for row in prange(n_row, schedule="${sch.split('_')[1]}"):
                % endif
                    # Iterate over columns, taking padding into account
                    i_start = row - min(patch_distance, row)
                    i_end = row + min(patch_distance + 1, n_row - row)

                    for col in range(n_col):
                        # Reset weights for each local region
                        new_value = 0 
                        weight_sum = 0

                        j_start = col - min(patch_distance, col)
                        j_end = col + min(patch_distance + 1, n_col - col)

                        # Iterate over local 2d patch for each pixel
                        for i in range(i_start, i_end):
                            for j in range(j_start, j_end):

                                weight = _c_patch_distance(
                                    &padded[f, row, col],
                                    &padded[f, 0, 0],
                                    &w[0, 0], patch_size, i, j, n_col+2*offset, var)

                                # Collect results in weight sum
                                weight_sum = weight_sum + weight
                                
                                new_value = new_value + weight * padded[f, i+offset, j+offset]
                        result[f, row, col] = new_value / weight_sum
                        
        return np.squeeze(np.asarray(result))

        %endfor
    
        
    def _run_opencl(self, image, int patch_size, int patch_distance, float h, float sigma, dict device, int mem_div=1) -> np.ndarray:
        cl_ctx = cl.Context([device['device']])
        dc = device['device']
        cl_queue = cl.CommandQueue(cl_ctx)

        # assure patch size is odd
        if patch_size % 2 == 0:
            patch_size = patch_size + 1

        nframes, nrow, ncol = image.shape
        offset = patch_size // 2

        padded = np.ascontiguousarray(np.pad(image, ((0,0), (offset, offset), (offset, offset)), mode='reflect'))
        result = np.empty_like(image)
        result_per_frame = np.empty_like(result[0,:,:])

        A = ((patch_size - 1.) / 4.)
        range_vals = np.arange(-offset, offset + 1, dtype=np.float32)
        xg_row, xg_col = np.meshgrid(range_vals, range_vals, indexing='ij')
        w = np.ascontiguousarray(np.exp(-(xg_row * xg_row + xg_col * xg_col) / (2 * A * A)))
        w *= 1. / ( np.sum(w) * h * h)

        var = 2 * sigma * sigma

        code = self._get_cl_code("_le_nlm_denoising_.cl", device['DP'])
        prg = cl.Program(cl_ctx, code).build()
        knl = prg.nlm_denoising

        padded_cl = cl.image_from_array(cl_ctx, padded[0,:,:], mode='r')
        result_cl = cl.image_from_array(cl_ctx, result_per_frame, mode='w')
        w_cl = cl.image_from_array(cl_ctx, w, mode='r')
        cl_queue.finish()


        for f in range(nframes):
            
            knl(cl_queue,
                (nrow,ncol),
                None,
                padded_cl,
                result_cl,
                w_cl,
                np.int32(patch_distance),
                np.int32(patch_size),
                np.int32(offset),
                np.float32(var)).wait()

            cl.enqueue_copy(cl_queue, result_per_frame, result_cl, origin=(0,0), region=(nrow,ncol)).wait()
            result[f,:,:] = result_per_frame

            if f<(nframes-1):
                cl.enqueue_copy(cl_queue, padded_cl, padded[f+1,:,:],origin=(0,0),region=(nrow+offset*2,ncol+offset*2)).wait()

            cl_queue.finish()

        return np.squeeze(result)
