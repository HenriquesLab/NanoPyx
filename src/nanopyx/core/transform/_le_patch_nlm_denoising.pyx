# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
cimport numpy as np

from skimage.restoration import denoise_nl_means

from libc.math cimport exp
from cython.parallel import parallel, prange

from .__interpolation_tools__ import check_image
from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array


cdef extern from "_c_integral_image.h":
    void _c_integral_image(float* image, float* integral, int n_row, int n_col, int t_row, int r_col, float var_diff) nogil

cdef extern from "_c_integral_to_distance.h":
    float _c_integral_to_distance(float* integral, int rows, int cols, int row, int col, int offset, float h2s2) nogil


class NLMDenoising(LiquidEngine):
    """
    Non-local means Denoising using the NanoPyx Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "NLMDenoising_patch"
        super().__init__(
            clear_benchmarks=clear_benchmarks, testing=testing,
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

    def _run_threaded(self, float[:, :, :] image, int patch_size=7, int patch_distance=11, float h=0.1, float sigma=0.0) -> np.ndarray:

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
                for shift in prange(n_shifts):
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
                    for f_col in prange(pad_size, n_col - pad_size):
                        final_result = 0
                        final_weight = 0
                        for shift in range(n_shifts):
                            final_result = final_result + result[shift,f,f_row,f_col]
                            final_weight = final_weight + weights[shift,f,f_row,f_col]
                        # No risk of division by zero, since the contribution
                        # of a null shift is strictly positive
                        output_result[f, f_row, f_col] = final_result / final_weight
                        
        return np.squeeze(np.asarray(output_result[:, pad_size: -pad_size,pad_size: -pad_size]).astype(np.float32))
    def _run_threaded_guided(self, float[:, :, :] image, int patch_size=7, int patch_distance=11, float h=0.1, float sigma=0.0) -> np.ndarray:

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
                for shift in prange(n_shifts,schedule="guided"):
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
                    for f_col in prange(pad_size, n_col - pad_size, schedule="guided"):
                        final_result = 0
                        final_weight = 0
                        for shift in range(n_shifts):
                            final_result = final_result + result[shift,f,f_row,f_col]
                            final_weight = final_weight + weights[shift,f,f_row,f_col]
                        # No risk of division by zero, since the contribution
                        # of a null shift is strictly positive
                        output_result[f, f_row, f_col] = final_result / final_weight
                        
        return np.squeeze(np.asarray(output_result[:, pad_size: -pad_size,pad_size: -pad_size]).astype(np.float32))
    def _run_threaded_dynamic(self, float[:, :, :] image, int patch_size=7, int patch_distance=11, float h=0.1, float sigma=0.0) -> np.ndarray:

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
                for shift in prange(n_shifts,schedule="dynamic"):
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
                    for f_col in prange(pad_size, n_col - pad_size, schedule="dynamic"):
                        final_result = 0
                        final_weight = 0
                        for shift in range(n_shifts):
                            final_result = final_result + result[shift,f,f_row,f_col]
                            final_weight = final_weight + weights[shift,f,f_row,f_col]
                        # No risk of division by zero, since the contribution
                        # of a null shift is strictly positive
                        output_result[f, f_row, f_col] = final_result / final_weight
                        
        return np.squeeze(np.asarray(output_result[:, pad_size: -pad_size,pad_size: -pad_size]).astype(np.float32))
    def _run_threaded_static(self, float[:, :, :] image, int patch_size=7, int patch_distance=11, float h=0.1, float sigma=0.0) -> np.ndarray:

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
                for shift in prange(n_shifts,schedule="static"):
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
                    for f_col in prange(pad_size, n_col - pad_size, schedule="static"):
                        final_result = 0
                        final_weight = 0
                        for shift in range(n_shifts):
                            final_result = final_result + result[shift,f,f_row,f_col]
                            final_weight = final_weight + weights[shift,f,f_row,f_col]
                        # No risk of division by zero, since the contribution
                        # of a null shift is strictly positive
                        output_result[f, f_row, f_col] = final_result / final_weight
                        
        return np.squeeze(np.asarray(output_result[:, pad_size: -pad_size,pad_size: -pad_size]).astype(np.float32))
    
        
    def _run_opencl(self, image, int patch_size, int patch_distance, float h, float sigma, dict device) -> np.ndarray:
        # QUEUE AND CONTEXT
        cl_ctx = cl.Context([device['device']])
        dc = device['device']
        cl_queue = cl.CommandQueue(cl_ctx)

        distance_cutoff = 5.0

        if patch_size%2==0:
            patch_size += 1

        offset = patch_size // 2
        pad_size = offset + patch_distance + 1

        padded = np.ascontiguousarray(np.pad(image,((0, 0), (pad_size, pad_size), (pad_size, pad_size)),mode='reflect').astype(np.float32))
        padded_opencl = cl.Buffer(cl_ctx, cl.mem_flags.READ_ONLY, padded.nbytes)
        cl.enqueue_copy(cl_queue,padded_opencl, padded)

        n_frames, n_row, n_col = padded.shape[0], padded.shape[1], padded.shape[2]

        shifts = np.concatenate(np.dstack(np.mgrid[-patch_distance:patch_distance+1,0:patch_distance+1])).astype(np.int32)
        shifts_opencl = cl.Buffer(cl_ctx, cl.mem_flags.READ_ONLY, shifts.nbytes)
        cl.enqueue_copy(cl_queue, shifts_opencl,shifts).wait()

        n_shifts = shifts.shape[0]

        weights = np.zeros((n_shifts,n_frames,n_row,n_col)).astype(np.float32)
        weights_opencl = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, weights.nbytes)
        cl.enqueue_copy(cl_queue,weights_opencl,weights)

        integral = np.zeros((n_shifts,n_frames,n_row,n_col)).astype(np.float32)
        integral_opencl = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, integral.nbytes)
        cl.enqueue_copy(cl_queue,integral_opencl,integral)

        result = np.zeros((n_shifts,n_frames,n_row,n_col)).astype(np.float32)
        result_opencl = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, result.nbytes)
        cl.enqueue_copy(cl_queue,result_opencl,result)

        output_result = np.zeros_like(padded)
        output_opencl = cl.Buffer(cl_ctx, cl.mem_flags.WRITE_ONLY, output_result.nbytes)
        cl.enqueue_copy(cl_queue,output_opencl,output_result)

        h2s2 = h*h*patch_size*patch_size
        var = sigma*sigma*2

        code = self._get_cl_code("_le_patch_nlm_denoising_.cl", device['DP'])
        prg = cl.Program(cl_ctx, code).build()
        knl_denoising = prg.nlm_denoising
        knl_normalization = prg.nlm_normalizer
        
        knl_denoising(cl_queue,
                     (n_frames,n_shifts), 
                     None,
                     padded_opencl,
                     shifts_opencl,
                     weights_opencl,
                     integral_opencl,
                     result_opencl,
                     np.int32(n_row),
                     np.int32(n_col),
                     np.int32(patch_size),
                     np.int32(patch_distance),
                     np.float32(h2s2),
                     np.float32(var)).wait() 

        knl_normalization(cl_queue,
                          (n_frames,n_row-2*pad_size,n_col-2*pad_size),
                          None,
                          weights_opencl,
                          result_opencl,
                          output_opencl,
                          np.int32(pad_size),
                          np.int32(n_shifts),
                          np.int32(n_row),
                          np.int32(n_col)).wait()

        cl_queue.finish()

        cl.enqueue_copy(cl_queue, output_result, output_opencl).wait()

        return np.squeeze(np.asarray(output_result[:,pad_size:-pad_size,pad_size:-pad_size]).astype(np.float32))