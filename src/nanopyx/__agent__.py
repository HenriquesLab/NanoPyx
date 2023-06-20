import platform
import random

import numpy as np
from hmmlearn import hmm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .liquid.__njit__ import njit_works
from .liquid.__opencl__ import opencl_works, devices

class Agent_:

    """
    Base class for the Agent of the Nanopyx Liquid Engine 
    Pond, James Pond
    """

    def __init__(self,) -> None:
        """
        Initialize the Agent
        The agent is supposed to work as a singleton object, initialized only once in the __init__.py of nanopyx
        PS: (Is this good enough or is it necessary to implement the singleton design pattern?)

        Agent responsabilities:
            1. Store the current state of the machine (e.g. OS, CPU, RAM, GPU, Python version etc.);
            2. Store the current state of ALL initialized LE objects (e.g. anything that is currently running, anything that is scheduled to run,
                runs previously executed in the current session etc.);
            3. Whenever a LE object wants to run, it must query the Agent on what is the best implementation for it;
            4. Tests whether there was an unexpected delay and adjust following paths based on it;
        """

        ### MACHINE INFO ###
        self.os_info = {'OS':platform.platform(),'Architecture':platform.machine()}
        self.cpu_info = {'CPU':platform.processor()}
        self.ram_info = {'RAM':'TBD'}
        self.py_info = {'Version':platform.python_version(),'Implementation':platform.python_implementation(),'Compiler':platform.python_compiler()}

        self.numba_info = {'Numba':njit_works()}
        self.pyopencl_info = {'PyOpenCL':opencl_works(),'Devices':devices}
        self.cuda_info = {'CUDA':'TBD'}
        ### MACHINE INFO ###

        self._current_runs = []
        self._scheduled_runs = []
        self._finished_runs = []
        
        self.delayed_runtypes = {}  # Store runtypes as keys and their values as (delay_factor, delay_prob)

    def _get_ordered_run_types(self, fn, args, kwargs):
        """
        Retrieves an ordered list of run_types for the given args and kwargs
        """

        # str representation of the arguments and their corresponding 'norm'
        repr_args, repr_norm = fn._get_args_repr_score(*args, **kwargs)
        # dictionary to hold speeds
        avg_speed = {}
        std_speed = {}
        # fn._benchmarks is a dictionary of dictionaries. The first key is the run_type, the second key is the repr_args
        # Check every run_type for the most similar args
        for run_type in fn._run_types:
            if repr_args in fn._benchmarks[run_type]:
                run_info = fn._benchmarks[run_type][repr_args][1:]
            else:
                # if the repr_args are not in the benchmarks, find the most similar repr_args
                best_score = np.inf
                best_repr_args = None
                for repr_args_ in fn._benchmarks[run_type]:
                    score = np.abs(fn._benchmarks[run_type][repr_args_][0] - repr_norm)
                    if score < best_score:
                        best_score = score
                        best_repr_args = repr_args_
                # What happens if there are no benchmarks for this runtype?
                # Make it slow TODO
                if best_repr_args is None:
                    print(f"run_type {run_type} has no benchmarks for the given args and kwargs.")
                    run_info = np.inf 
                else:
                    run_info = fn._benchmarks[run_type][best_repr_args][1:]
            
            avg_speed[run_type] = np.nanmean(run_info)
            std_speed[run_type] = np.nanstd(run_info)

        return avg_speed, std_speed
    
    def _calculate_prob_of_delay(self, runtimes_history, avg, std):
        delays = runtimes_history > avg+2*std
        label_encoder = LabelEncoder()
        integer_delays = label_encoder.fit_transform(delays)
        delay_state_index = label_encoder.transform(True)[0]
        model = hmm.CategoricalHMM(n_components=2)
        model.fit(np.array(integer_delays).reshape(-1, 1))
        return model.predict_proba([[True]])[:, delay_state_index]

    def _check_delay(self, run_type, runtime, runtimes_history):
        avg = np.nanmean(runtimes_history)
        std = np.nanstd(runtimes_history)
        if runtime > avg + 2*std:
            self._store_delay(run_type,
                              delay_factor=runtime/avg,
                              prob=self._calculate_prob_of_delay(runtimes_history, avg, std))
    
    def _store_delay(self, run_type, delay_factor=1, prob=0):
        self.delayed_runtypes[run_type] = (delay_factor, prob)  # TODO change to average values of same delay
        
    def _adjust_times(self, device_times):
        for runtype in self.delayed_runtypes.keys():
            delay_factor, delay_prob = self.delayed_runtypes[runtype]
            device_times[runtype] = device_times[runtype] * (1 - delay_prob) + device_times[runtype] * delay_factor * delay_prob

    def get_run_type(self, fn, args, kwargs, mode='dynamic'):
        """
        Returns the best run_type for the given args and kwargs
        """

        # Start easy

        avg, std = self._get_ordered_run_types(fn, args, kwargs)
        
        if len(self.delayed_runtypes.keys()) > 0:
            self._adjust_times(avg)

        sorted_fastest = sorted(avg, key=avg.get)

        # no match case in python >=3.9 so elifs it is
        if mode == 'fastest':
            return sorted_fastest[0]
        elif mode == 'dynamic':
            weights = [(1/avg[k])**2 for k in avg]
            if sum(weights) == 0:
                weights = [1 for k in avg]
            return random.choices(list(avg.keys()), weights=weights, k=1)[0]
        else:
            return sorted_fastest[0]

Agent = Agent_()