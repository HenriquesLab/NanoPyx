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

        self._default_benchmarks = {'OpenCL':[1,1,1],'Threaded':[2,2,2],'Threaded_static':[3,3,3],'Threaded_dynamic':[4,4,4],'Threaded_guided':[5,5,5],'Python':[6,6,6],'Numba':[7,7,7],'Unthreaded':[8,8,8]}

    def run(self, run_type=None):
        return self._run(run_type=run_type)

    def benchmark(self,):
        return super().benchmark()

    def _run_opencl(self, dict device):
        # 2 seconds
        msec = 2
        if self._delay=="OpenCL":
            t2run = np.random.normal(msec*self._delay_amount, 0.1*msec*self._delay_amount)
            sleep(t2run)
        else:
            t2run = np.random.normal(msec, 0.1*msec)
            sleep(t2run)

        return None

    def _run_unthreaded(self,):
        # 10 seconds
        msec = 10
        if self._delay=="Unthreaded":
            sleep(msec*self._delay_amount)
        else:
            sleep(msec)

        return None

    def _run_threaded(self,):
        # 5 seconds
        msec = 5
        if self._delay=="Threaded":
            t2run = np.random.normal(msec*self._delay_amount, 0.1*msec*self._delay_amount)
            sleep(t2run)
        else:
            t2run = np.random.normal(msec, 0.1*msec)
            sleep(t2run)

        return None

    def _run_threaded_static(self,):
        # 5 seconds
        msec = 5
        if self._delay=="Threaded_static":
            t2run = np.random.normal(msec*self._delay_amount, 0.1*msec*self._delay_amount)
            sleep(t2run)
        else:
            t2run = np.random.normal(msec, 0.1*msec)
            sleep(t2run)

        return None

    def _run_threaded_dynamic(self,):
        # 5 seconds
        msec = 5
        if self._delay=="Threaded_dynamic":
            t2run = np.random.normal(msec*self._delay_amount, 0.1*msec*self._delay_amount)
            sleep(t2run)
        else:
            t2run = np.random.normal(msec, 0.1*msec)
            sleep(t2run)

        return None

    def _run_threaded_guided(self,):
        # 5 seconds
        msec = 5
        if self._delay=="Threaded_guided":
            t2run = np.random.normal(msec*self._delay_amount, 0.1*msec*self._delay_amount)
            sleep(t2run)
        else:
            t2run = np.random.normal(msec, 0.1*msec)
            sleep(t2run)

        return None