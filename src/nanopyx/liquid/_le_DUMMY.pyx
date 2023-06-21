# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
cimport numpy as np

from libc.math cimport cos, sin
from libc.stdlib cimport usleep

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
        msec = 5000000
        if delay=="OpenCL":
            usleep(msec*delay_amount)
        else:
            usleep(msec)

        return None

    def _run_unthreaded(self,):
        # 20 seconds
        msec = 20000000
        if delay=="Unthreaded":
            usleep(msec*delay_amount)
        else:
            usleep(msec)

        return None

    def _run_threaded(self,):
        # 10 seconds
        msec = 10000000
        if delay=="Threaded":
            usleep(msec*delay_amount)
        else:
            usleep(msec)

        return None

    def _run_threaded_static(self,):
        # 10 seconds
        msec = 10000000
        if delay=="Threaded_static":
            usleep(msec*delay_amount)
        else:
            usleep(msec)

        return None

    def _run_threaded_dynamic(self,):
        # 10 seconds
        msec = 10000000
        if delay=="Threaded_dynamic":
            usleep(msec*delay_amount)
        else:
            usleep(msec)

        return None

    def _run_threaded_guided(self,):
        # 10 seconds
        msec = 10000000
        if delay=="Threaded_guided":
            usleep(msec*delay_amount)
        else:
            usleep(msec)

        return None
  