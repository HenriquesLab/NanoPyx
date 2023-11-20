# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
cimport numpy as np

from libc.math cimport exp
from cython.parallel import parallel, prange

from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array


cdef extern from "_c_integral_image.h":
    void _c_integral_image(float *image, float *integral, int n_row, int n_col, int n_channel, int t_row, int r_col, float var_diff)

cdef extern from "_c_integral_to_distance.h":
    float _c_integral_to_distance(float *integral, int rows, int cols, int row, int col, int offset, float h2s2)


class NLMDenoising(LiquidEngine):
    """
    Non-local means Denoising using the NanoPyx Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "NLMDenoising"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing,
                        unthreaded_=True, threaded_=True, threaded_static_=True,
                        threaded_dynamic_=True, threaded_guided_=True, opencl_=True)

    def run(self, np.ndarray image, int patch_size=7, int patch_distance=11, float h=0.1, float sigma=0.0, run_type=None) -> np.ndarray:
        """
        Run the non-local means denoising algorithm
        :param image: np.ndarray or memoryview
        :param patch_size: int, default 7, the size of the patch
        :param patch_distance: int, default 11, the maximum patch distance
        :param sigma: float, default 0.0, the standard deviation of the gaussian kernel
        :return: np.ndarray
        """

        if int(len(image.shape)) == 2:
            image = np.asarray([image])
        
        return self._run(image, sigma=sigma, smoothing_factor=smoothing_factor, run_type=run_type)

    def benchmark(self, np.ndarray image, int patch_size=7, int patch_distance=11, float h=0.1, float sigma=0.0, run_type=None) -> np.ndarray:
        """
        Run the non-local means denoising algorithm
        :param image: np.ndarray or memoryview
        :param patch_size: int, default 7, the size of the patch
        :param patch_distance: int, default 11, the maximum patch distance
        :param sigma: float, default 0.0, the standard deviation of the gaussian kernel
        :return: The benchmark results
        :rtype: [[run_time, run_type_name, return_value], ...]
        """

        if int(len(image.shape)) == 2:
            image = np.asarray([image])
        
        return super()._benchmark(image, sigma=sigma, smoothing_factor=smoothing_factor)

    def _run_threaded(self, float[:,:,:] image, float patch_size, float max_distance, float cut_off_distance, float noise_variance) -> np.ndarray:
        pass

