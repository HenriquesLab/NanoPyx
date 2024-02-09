import numpy as np

from time import sleep

from nanopyx.__liquid_engine__ import LiquidEngine

class OverheadTester(LiquidEngine):

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "OverheadTester"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing,
                        opencl_=True, unthreaded_=True, threaded_=True)

        
    def run(self, t, run_type=None):
        return self._run(t, run_type=run_type)

    def benchmark(self,t):
        return super().benchmark(t)

    def _run_opencl(self, t, device):
        sleep(t)
        return None

    def _run_unthreaded(self,t):
        sleep(1)
        return None

    def _run_threaded(self,t):
        sleep(0.5)
        return None
