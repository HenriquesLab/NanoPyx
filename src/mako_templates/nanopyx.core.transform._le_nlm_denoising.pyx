<%!
schedulers = ['threaded','threaded_guided','threaded_dynamic','threaded_static']
%># cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import time
import numpy as np
cimport numpy as np

from skimage.restoration import denoise_nl_means

from libc.math cimport exp
from cython.parallel import parallel, prange

from .__interpolation_tools__ import check_image
from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array


cdef extern from "_c_patch_distance.h":
    float _c_patch_distance(float* image, float* integral, float* w, int patch_distance, int n_col, float var) nogil


class NLMDenoising(LiquidEngine):
    """
    Non-local means Denoising using the NanoPyx Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "NLMDenoising"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing,
                        unthreaded_=True, threaded_=True, threaded_static_=True,
                        threaded_dynamic_=True, threaded_guided_=True, opencl_=False,
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

    def _run_unthreaded(self, float[:, :, :] image, int patch_size=7, int patch_distance=11, float h=0.1, float sigma=0.0) -> np.ndarray:
        """
        Run the non-local means denoising algorithm
        :param image: np.ndarray or memoryview
        :param patch_size: int, default 7, the size of the patch
        :param patch_distance: int, default 11, the maximum patch distance
        :param h: float, default 0.1, Cut-off distance (in gray levels).
        :param sigma: float, default 0.0, the standard deviation of the gaussian kernel
        :return: np.ndarray
        """

        if patch_size % 2 == 0:
            patch_size += 1  # odd value for symmetric patch

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
        cdef float[:] range_vals = np.arange(-offset, offset + 1,
                                                    dtype=np.float32)
        xg_row, xg_col = np.meshgrid(range_vals, range_vals, indexing='ij')
        cdef float[ :,:] w = np.ascontiguousarray(
            np.exp(-(xg_row * xg_row + xg_col * xg_col) / (2 * A * A)))
        w *= 1. / (np.sum(w) * h * h)

        cdef float[:, :, :] central_patch  = np.zeros((image.shape[0], patch_size, patch_size), dtype=np.float32)
        cdef float var = sigma * sigma
        var *= 2

        # Iterate over rows, taking padding into account
        with nogil:
            for f in range(n_frames):
                for row in range(n_row):
                    # Iterate over columns, taking padding into account
                    i_start = row - min(patch_distance, row)
                    i_end = row + min(patch_distance + 1, n_row - row)

                    for col in range(n_col):
                        # Reset weights for each local region
                        weight_sum = 0
                        weight = 0

                        central_patch[f] = padded[f, row:row+patch_size, col:col+patch_size]
                        j_start = col - min(patch_distance, col)
                        j_end = col + min(patch_distance + 1, n_col - col)

                        # Iterate over local 2d patch for each pixel
                        for i in range(i_start, i_end):
                            for j in range(j_start, j_end):
                                weight = _c_patch_distance(
                                    &central_patch[f, 0, 0],
                                    &padded[f, i, j],
                                    &w[0, 0], patch_size, n_col, var)

                                # Collect results in weight sum
                                weight_sum = weight_sum + weight
                                result[f, row, col] = result[f, row, col] + weight * padded[f, i+offset, j+offset]
                        result[f, row, col] = result[f, row, col] / weight_sum

        return np.squeeze(np.asarray(result))

    % for sch in schedulers:
    def _run_${sch}(self, float[:, :, :] image, int patch_size=7, int patch_distance=11, float h=0.1, float sigma=0.0) -> np.ndarray:
        if patch_size % 2 == 0:
            patch_size += 1  # odd value for symmetric patch

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
        cdef float[:] range_vals = np.arange(-offset, offset + 1,
                                                    dtype=np.float32)
        xg_row, xg_col = np.meshgrid(range_vals, range_vals, indexing='ij')
        cdef float[:,:] w = np.ascontiguousarray(
            np.exp(-(xg_row * xg_row + xg_col * xg_col) / (2 * A * A)))
        w *= 1. / (np.sum(w) * h * h)

        cdef float[:, :, :] central_patch = np.zeros((image.shape[0], patch_size, patch_size), dtype=np.float32)
        cdef float var = sigma * sigma
        var *= 2
        
        with nogil:
            for f in range(n_frames):
                % if sch=='threaded':
                for row in prange(n_row):
                % else:
                for row in prange(n_row, schedule="${sch.split('_')[1]}"):
                % endif
                    # Iterate over columns, taking padding into account
                    i_start = row - min(patch_distance, row)
                    i_end = row + min(patch_distance + 1, n_row - row)

                    for col in range(n_col):
                        # Reset weights for each local region
                        weight_sum = 0
                        weight = 0

                        central_patch[f] = padded[f, row:row+patch_size, col:col+patch_size]
                        j_start = col - min(patch_distance, col)
                        j_end = col + min(patch_distance + 1, n_col - col)

                        # Iterate over local 2d patch for each pixel
                        for i in range(i_start, i_end):
                            for j in range(j_start, j_end):
                                weight = _c_patch_distance(
                                    &central_patch[f, 0, 0],
                                    &padded[f, i, j],
                                    &w[0, 0], patch_size, n_col, var)

                                # Collect results in weight sum
                                weight_sum = weight_sum + weight
                                result[f, row, col] = result[f, row, col] + weight * padded[f, i+offset, j+offset]
                        result[f, row, col] = result[f, row, col] / weight_sum

        return np.squeeze(np.asarray(result))

        %endfor
    
        
    def _run_opencl(self, image, int patch_size, int patch_distance, float h, float sigma, dict device) -> np.ndarray:
        pass
