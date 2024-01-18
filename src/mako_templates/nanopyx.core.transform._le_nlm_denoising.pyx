<%!
schedulers = ['threaded','threaded_guided','threaded_dynamic','threaded_static']
%># cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
cimport numpy as np

from itertools import product

from skimage.restoration import denoise_nl_means

from libc.math cimport exp, isnan, fmax
from cython.parallel import parallel, prange

from .__interpolation_tools__ import check_image
from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array

import os
os.environ['PYOPENCL_NO_CACHE']='1'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

cdef extern from "_c_integral_image.h":
    void _c_integral_image(float* image, float* integral, int n_row, int n_col, int t_row, int r_col, float var_diff) nogil

cdef extern from "_c_integral_to_distance.h":
    float _c_integral_to_distance(float* integral, int rows, int cols, int row, int col, int offset, float h2s2) nogil

class NLMDenoising(LiquidEngine):
    """
    Non-local means Denoising using the NanoPyx Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "NLMDenoising"
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
            out[i] = denoise_nl_means(image[i], patch_size=patch_size, patch_distance=patch_distance, h=h, sigma=sigma, fast_mode=True)

        return np.squeeze(out)

    def _run_unthreaded(self, float[:, :, :] image, int patch_size=7, int patch_distance=11, float h=0.1, float sigma=0.0) -> np.ndarray:
        cdef float distance_cutoff = 5.0
        cdef float var = sigma * sigma

        if patch_size % 2 == 0:
            patch_size += 1
        
        cdef int n_row, n_col, t_row, t_col, row, col, row_start, row_end, row_shift, col_shift, f
        cdef int offset = patch_size // 2
        cdef int pad_size = offset + patch_distance + 1
        cdef float[:, :, :] padded = np.ascontiguousarray(
            np.pad(
                image,
                ((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
                mode='reflect'
            ).astype(np.float32))
        cdef float[:, :] weights = np.zeros_like(padded[0])
        cdef float[:, :] integral = np.zeros_like(weights)
        cdef float[:, :, :] result = np.zeros_like(padded)

        cdef float distance, h2s2, weight, alpha

        n_frames, n_row, n_col = padded.shape[0], padded.shape[1], padded.shape[2]
        h2s2 = patch_size * patch_size * h * h
        var *=  2
        
        with nogil:
            for f in range(n_frames):
                for t_row in range(-patch_distance, patch_distance + 1):
                    row_start = max(offset, offset - t_row)
                    row_end = min(n_row - offset, n_row - offset - t_row)
                    # Iterate over shifts along the column axis
                    for t_col in range(0, patch_distance + 1):
                        # alpha is to account for patches on the same column
                        # distance is computed twice in this case
                        alpha = 0.5 if t_col == 0 else 1

                        # Compute integral image of the squared difference between
                        # padded and the same image shifted by (t_row, t_col)
                        _c_integral_image(&padded[f, 0, 0], &integral[0, 0],
                                        n_row, n_col, t_row, t_col, var)

                        # Inner loops on pixel coordinates
                        # Iterate over rows, taking offset and shift into account
                        for row in range(row_start, row_end):
                            row_shift = row + t_row
                            # Iterate over columns, taking offset and shift into account
                            for col in range(offset, n_col - offset - t_col):
                                # Compute squared distance between shifted patches
                                distance = _c_integral_to_distance(
                                    &integral[0, 0], n_row, n_col, row, col, offset, h2s2)
                                # exp of large negative numbers is close to zero
                                if distance > distance_cutoff:
                                    continue
                                col_shift = col + t_col
                                weight = alpha * exp(-distance)
                                # Accumulate weights corresponding to different shifts
                                weights[row, col] += weight
                                weights[row_shift, col_shift] += weight

                                result[f, row, col] = result[f, row, col] + weight * padded[f, row_shift, col_shift]
                                result[f, row_shift, col_shift] = result[f, row_shift, col_shift] + weight * padded[f, row, col]

                # Normalize pixel values using sum of weights of contributing patches
                for row in range(pad_size, n_row - pad_size):
                    for col in range(pad_size, n_col - pad_size):
                        # No risk of division by zero, since the contribution
                        # of a null shift is strictly positive
                        result[f, row, col] /= weights[row, col]

                with gil:
                    weights = np.zeros_like(padded[0])

        # Return cropped result, undoing padding
        return np.squeeze(np.asarray(result[:, pad_size: -pad_size,
                                            pad_size: -pad_size]).astype(np.float32))

    % for sch in schedulers:
    def _run_${sch}(self, float[:, :, :] image, int patch_size=7, int patch_distance=11, float h=0.1, float sigma=0.0) -> np.ndarray:

        cdef float distance_cutoff = 5.0

        if patch_size%2==0:
            patch_size+=1

        cdef int offset = patch_size / 2
        cdef int pad_size = offset + patch_distance + 1

        cdef float[:,:,:] padded = np.ascontiguousarray(np.pad(image,((0, 0), (pad_size, pad_size), (pad_size, pad_size)),mode='reflect').astype(np.float32))
        cdef int n_frames, n_row, n_col
        n_frames, n_row, n_col = padded.shape[0], padded.shape[1], padded.shape[2]
        
        cdef int[:,:] shifts = np.concatenate(np.dstack(np.mgrid[-patch_distance:patch_distance+1,0:patch_distance+1])).astype(np.int32)
        cdef int n_shifts = shifts.shape[0]

        cdef float[:,:,:,:] weights = np.zeros((n_shifts,n_frames,n_row,n_col)).astype(np.float32)
        
        cdef float[:,:,:,:] integral = np.zeros((n_shifts,n_frames,n_row,n_col)).astype(np.float32)

        cdef float[:,:,:,:] result = np.zeros((n_shifts,n_frames,n_row,n_col)).astype(np.float32)
        cdef float[:,:,:] output_result = np.zeros_like(padded)

        cdef float h2s2 = h*h*patch_size*patch_size
        cdef float var = sigma*sigma*2

        cdef int f, shift, t_row, t_col, row_start, row_end, row_shift, col_shift, row, col, f_row, f_col
        cdef float alpha, distance, final_result, final_weight, weight
        with nogil:
            for f in range(n_frames):
                % if sch=='threaded':
                for shift in prange(n_shifts):
                % else:
                for shift in prange(n_shifts,schedule="${sch.split('_')[1]}"):
                % endif
                    t_row = shifts[shift,0]
                    t_col = shifts[shift,1]

                    row_start = max(offset, offset-t_row)
                    row_end = min(n_row-offset,n_row-offset-t_row)
                    alpha = 0.5 if t_col == 0 else 1
                    
                    _c_integral_image(&padded[f,0,0],&integral[shift,f,0,0],n_row,n_col,t_row,t_col,var)
                    
                    for row in range(row_start,row_end):
                        for col in range(offset,n_col-offset-t_col):
                            row_shift = row + t_row
                            col_shift = col + t_col
                            distance = _c_integral_to_distance(&integral[shift,f,0,0],n_row,n_col,row,col,offset,h2s2)
                            if distance > distance_cutoff:
                                continue
                            weight = alpha * exp(-distance)

                            weights[shift,f,row,col] = weights[shift,f,row,col] + weight
                            weights[shift,f,row_shift,col_shift] = weights[shift,f,row_shift,col_shift] + weight

                            result[shift,f,row,col] = result[shift,f,row,col] + weight * padded[f,row_shift,col_shift]
                            result[shift,f,row_shift,col_shift] = result[shift,f,row_shift,col_shift] + weight * padded[f,row,col]
                    
                for f_row in range(pad_size, n_row - pad_size):
                    % if sch=='threaded':
                    for f_col in prange(pad_size, n_col - pad_size):
                    % else:
                    for f_col in prange(pad_size, n_col - pad_size, schedule="${sch.split('_')[1]}"):
                    % endif
                        final_result = 0
                        final_weight = 0
                        for shift in range(n_shifts):
                            final_result = final_result + result[shift,f,f_row,f_col]
                            final_weight = final_weight + weights[shift,f,f_row,f_col]
                        # No risk of division by zero, since the contribution
                        # of a null shift is strictly positive
                        output_result[f, f_row, f_col] = final_result / final_weight
                        
        return np.squeeze(np.asarray(output_result[:, pad_size: -pad_size,pad_size: -pad_size]).astype(np.float32))
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
        result = np.zeros_like(image)
        result_per_frame = np.zeros_like(result[0,:,:])

        A = ((patch_size - 1.) / 4.)
        range_vals = np.arange(-offset, offset + 1, dtype=np.float32)
        xg_row, xg_col = np.meshgrid(range_vals, range_vals, indexing='ij')
        w = np.ascontiguousarray(np.exp(-(xg_row * xg_row + xg_col * xg_col) / (2 * A * A)))
        w *= 1. / ( np.sum(w) * h * h)

        var = 2 * sigma * sigma

        code = self._get_cl_code("_le_nlm_denoising_.cl", device['DP'])
        prg = cl.Program(cl_ctx, code).build(options="-cl-opt-disable -Werror -g")
        knl = prg.nlm_denoising

        padded_cl = cl.image_from_array(cl_ctx, np.asarray(padded[0,:,:],order='C', dtype=np.float32), mode='r')
        result_cl = cl.image_from_array(cl_ctx, np.asarray(result_per_frame,order='C', dtype=np.float32), mode='w')
        w_cl = cl.image_from_array(cl_ctx, np.asarray(w,order='C',dtype=np.float32), mode='r')
        cl_queue.finish()


        for f in range(nframes):
            for bl in product([0,1],[0,1]):
                knl(cl_queue,
                    (nrow//2,ncol//2),
                    (1,1),
                    padded_cl,
                    result_cl,
                    w_cl,
                    np.int32(patch_distance),
                    np.int32(patch_size),
                    np.int32(offset),
                    np.float32(var),
                    np.int32(bl[0]),
                    np.int32(bl[1]))
            
            cl_queue.finish()
            
            cl.enqueue_copy(cl_queue, result_per_frame, result_cl, origin=(0,0), region=(nrow,ncol)).wait()
            result[f,:,:] = result_per_frame

            if f<(nframes-1):
                cl.enqueue_copy(cl_queue, padded_cl, padded[f+1,:,:],origin=(0,0),region=(nrow+offset*2,ncol+offset*2)).wait()

            cl_queue.finish()

        return np.squeeze(result)
