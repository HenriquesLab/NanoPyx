import numpy as np
from ...__njit__ import njit, prange
from ...__liquid_engine__ import LiquidEngine

from ._le_template_simple_ import njit_template as _njit_template
from ._le_template_simple_ import py_template as _py_template

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
        return self._run(image)

    def benchmark(self, image):
        return super().benchmark(image)

    def _run_python(self, image: np.ndarray):
        image_out = _py_template(image)
        return image_out

    def _run_njit(self, image: np.ndarray):
        image_out = _njit_template(image)
        return image_out
