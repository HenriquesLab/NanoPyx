# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
cimport numpy as np

from time import sleep

from libc.math cimport cos, sin

from .__liquid_engine__ import LiquidEngine
from .__opencl__ import cl, cl_array


class DUMMY(LiquidEngine):
    """
    Dummy method to use as a test uses the NanoPyx Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False, delay=None, delay_amount=1):
        self._designation = "DUMMY"
        self._delay = delay
        self._delay_amount = delay_amount
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing,
                        opencl_=True, unthreaded_=True, threaded_=True, threaded_static_=True, 
                        threaded_dynamic_=True, threaded_guided_=True)

        

    def run(self, run_type=None):
        return self._run(run_type=run_type)

    def benchmark(self,):
        return super().benchmark()

    def _run_opencl(self, dict device):
        # 5 seconds
        msec = 5
        if self._delay=="OpenCL":
            sleep(msec*self._delay_amount)
        else:
            sleep(msec)

        return None

    def _run_unthreaded(self,):
        # 20 seconds
        msec = 20
        if self._delay=="Unthreaded":
            sleep(msec*self._delay_amount)
        else:
            sleep(msec)

        return None

    def _run_threaded(self,):
        # 10 seconds
        msec = 10
        if self._delay=="Threaded":
            sleep(msec*self._delay_amount)
        else:
            sleep(msec)

        return None

    def _run_threaded_static(self,):
        # 10 seconds
        msec = 10
        if self._delay=="Threaded_static":
            sleep(msec*self._delay_amount)
        else:
            sleep(msec)

        return None

    def _run_threaded_dynamic(self,):
        # 10 seconds
        msec = 10
        if self._delay=="Threaded_dynamic":
            sleep(msec*self._delay_amount)
        else:
            sleep(msec)

        return None

    def _run_threaded_guided(self,):
        # 10 seconds
        msec = 10
        if self._delay=="Threaded_guided":
            sleep(msec*self._delay_amount)
        else:
            sleep(msec)

        return None
  