
import platform

from dataclasses import dataclass

import numpy as np

from .__njit__ import njit_works
from .__opencl__ import opencl_works, devices

@dataclass
class Run:
    method: str = ''
    implementation: str = ''
    parameters: dict = {}
    result: dict = {}
    finished: bool = False
    time2run: float = 0.0


class Agent_:

    """
    Base class for the Agent of the Nanopyx Liquid Engine 
    Pond, James Pond
    """


    def __init__(self,) -> None:
        """
        Initialize the Agent
        The agent is supposed to work as a singleton object, initialized only once in the __init__.py of the LE
        PS: (Is this good enough or is it necessary to implement the singleton design pattern?)

        Agent responsabilities:
            1. Store the current state of the machine (e.g. OS, CPU, RAM, GPU, Python version etc.);
            2. Store the current state of ALL LE objects (e.g. anything that is currently running, anything that is scheduled to run,
              runs previously executed in the current session etc.);
            3. Whenever a LE object wants to run, it must query the Agent on what is the best implementation for it;
            4. Handle all necessary I/O operations related to benchmarks;

        Communication betwen a specific LE and the agent is done through Run dataclasses
        """

        ### MACHINE INFO ###
        self.os_info = {'OS':platform.platform(),'Architecture':platform.machine()}
        self.cpu_info = {'CPU':platform.processor()}
        self.ram_info = {'RAM':'TBD'}
        self.py_info = {'Version':platform.python_version(),'Implementation':platform.python_implementation(),'Compiler':platform.python_compiler()}

        self.numba_info = {'Numba':njit_works()}
        self.pyopencl_info = {'PyOpenCL':opencl_works(),'Devices':devices()}
        self.cuda_info = {'CUDA':'TBD'}
        ### MACHINE INFO ###


        self._current_runs = []
        self._scheduled_runs = []
        self._finished_runs = []



