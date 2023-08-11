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

        self._default_benchmarks = {'OpenCL': {'([], {})': [1, 1.8857776250006282, 1.7222011249996285, 1.9640285829991626, 2.4479195829990203, 1.8737244999992981, 1.7329023339989362]}, 'Threaded': {'([], {})': [1, 5.438421417000427, 4.801935208999566, 4.431757958000162, 3.945754375001343, 5.375589833998674, 4.888667750001332]}, 'Threaded_dynamic': {'([], {})': [1, 4.579944375000196, 4.745452291999754, 4.816967166998438, 4.024910833999456, 5.125496500000736, 4.431301374999748]}, 'Threaded_guided': {'([], {})': [1, 4.660112792000291, 4.024248042000181, 5.271654541998942, 4.687210165999204, 4.540968499999508, 4.769786417000432]}, 'Threaded_static': {'([], {})': [1, 5.155823874998532, 5.338639875000808, 4.677265624999563, 5.011717708001015, 5.208994832999451, 4.86386420800045]}, 'Unthreaded': {'([], {})': [1, 10.00231229200108, 10.003463375000138, 10.005027916000472, 10.001701082999716, 10.004999834000046, 10.005007916999602]}}

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