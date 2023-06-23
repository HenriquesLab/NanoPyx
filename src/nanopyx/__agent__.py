import platform
import random

import numpy as np
from hmmlearn import hmm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm

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

    def _gaussian_weighted_average_std(self, run_info, n_points=200, sigma=20):
        """
        Calculates the weighted average and standard deviation of the last n_points from the run_info array.
        
        :param run_info: The array of data to calculate the weighted average and standard deviation.
        :type run_info: numpy.ndarray
        
        :param n_points: The number of points to use in the calculation. Defaults to 200.
        :type n_points: int
        
        :return: A tuple containing the weighted average and standard deviation of the last n_points.
        :rtype: Tuple[float, float]
        """
        # TODO: update sigma to reflect choice of either timestamps vs n_points
        
        data = np.array(run_info)
        # remove nan values TODO: give a penalty to nan aka crash instead of ignoring
        data = data[np.isfinite(data)]
        if data.shape[0] < n_points:
            n_points = data.shape[0]
        lower_limit = data.shape[0] - n_points
        data = data[lower_limit:]
        
        # create trucnated normal distribution
        x = np.linspace(0, len(data)-1, len(data))
        mu = x[-1]
        weights = norm.pdf(x,mu,sigma)
        weights = np.abs(weights)  # take absolute value
        weights /= np.sum(weights)  # normalize to sum 1
        
        # calculate weighted average
        weighted_average = np.sum(data * weights)
        weighted_std = np.sqrt(np.sum(weights * (data - weighted_average)**2))
        
        # return weighted_average, weighted_std # TODO test and uncomment
        return np.nanmean(run_info), np.nanstd(run_info)

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
            
            avg_speed[run_type], std_speed[run_type] = self._gaussian_weighted_average_std(run_info)

        return avg_speed, std_speed
    
    def _calculate_prob_of_delay(self, runtimes_history, avg, std):
        """
        Calculates the probability that the given run_type is still delayed using historical data
        """
        # Boolean array, True if delay, False if not
        delays = runtimes_history > avg+2*std

        model = LogisticRegression()
        model.fit([[state] for state in delays[:-1]], delays[1:])
        
        return model.predict_proba([[True]])[0][model.classes_.tolist().index([True])]

    def _check_delay(self, run_type, runtime, runtimes_history):
        """
        Checks if the given run_type ran delayed in the previous run when compared with historical data
        If delayed:
            1. Calculates a probability that this delay is maintained
            2. Stores the delay factor and the probability
        """
        threaded_runtypes = ["Threaded", "Threaded_static", "Threaded_dynamic", "Threaded_guided"]
        avg = np.nanmean(runtimes_history) # standard average as opposed to weighted as a weighted average would throw false negatives if delays happen consecutively
        std = np.nanstd(runtimes_history)
        if runtime > avg + 2*std:
            runtimes_history.append(runtime)
            delay_factor = runtime / avg
            delay_prob = self._calculate_prob_of_delay(runtimes_history, avg, std)
            if "Threaded" in run_type:
                for threaded_run_type in threaded_runtypes:
                    self.delayed_runtypes[threaded_run_type] = (delay_factor, delay_prob)
            else:
                self.delayed_runtypes[run_type] = (delay_factor, delay_prob)
    
    def _adjust_times(self, device_times):
        """
        Adjusts the historic avg time of a run_type if it was delayed in previous runs
        """
        for runtype in self.delayed_runtypes.keys():
            if runtype in device_times.keys():
                delay_factor, delay_prob = self.delayed_runtypes[runtype]
                # Weighted avg by the probability the run_type is still delayed
                # expected_time * P(~delay) + delayed_tiem * P(delay)
                device_times[runtype] = device_times[runtype] * (1 - delay_prob) + device_times[runtype] * delay_factor * delay_prob

        return device_times

    def get_run_type(self, fn, args, kwargs, mode='fastest'):
        """
        Returns the best run_type for the given args and kwargs
        """

        # Get list of run types
        # Note that the avg and std are weighted to give more importance to the most recent runs
        avg, std = self._get_ordered_run_types(fn, args, kwargs)
        
        # Penalize the average time a run_type had if that run_type was delayed in previous runs
        if len(self.delayed_runtypes.keys()) > 0:
            avg = self._adjust_times(avg)

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

    def _inform(self, fn):
        """
        Informs the Agent that a LE object finished running
        """

        repr_args = fn._last_args
        run_type = fn._last_runtype
        
        historical_data = fn._benchmarks[run_type][repr_args][1:]
        
        assert historical_data[-1] == fn._last_time, "Historical data is not consistent with the last runtime"

        self._check_delay(run_type, historical_data[-1], historical_data[:-1])

Agent = Agent_()