import numpy as np
from .__njit__ import njit, prange
from .__liquid_engine__ import LiquidEngine
from .__interpolation_tools__ import check_image


class Template(LiquidEngine):
    """
    Template to implement new methods using the Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "Template" # change to name of your method
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing,
                        unthreaded_=True, threaded_=False, threaded_static_=False, 
                        threaded_dynamic_=False, threaded_guided_=False, opencl_=False,
                        njit_=True)  # change implemented run types to True 

    def run(self, image, run_type=None):
        image = check_image(image)
        return self._run(image)

    def benchmark(self, image):
        return super().benchmark(image)

    def _run_python(self, image: np.ndarray):

        for i in range(image.shape[0]):
            pass
        return np.asarray(image)

    @njit(cache=True, parallel=True)
    def _run_njit(self, image: np.ndarray):

        for i in prange(image.shape[0]):
            pass
        return np.asarray(image)
