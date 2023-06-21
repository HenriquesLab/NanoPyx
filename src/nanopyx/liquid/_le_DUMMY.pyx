# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
cimport numpy as np

from libc.math cimport cos, sin

from .__liquid_engine__ import LiquidEngine
from .__opencl__ import cl, cl_array


class DUMMY(LiquidEngine):
    """
    Dummy method to use as a test uses the NanoPyx Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "DUMMY"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing,
                        opencl_=True, unthreaded_=True, threaded_=True, threaded_static_=True, 
                        threaded_dynamic_=True, threaded_guided_=True)

    def run(self,):
        return self._run()

    def benchmark(self,):
        return super().benchmark()

    def _run_opencl(self, dict device):
        return 0

    def _run_unthreaded(self,):
        return 0

    def _run_threaded(self,):
        return 0

    def _run_threaded_static(self,) -> np.ndarray:
        return 0

    def _run_threaded_dynamic(self,) -> np.ndarray:
        return 0

    def _run_threaded_guided(self,) -> np.ndarray:
        return 0
  