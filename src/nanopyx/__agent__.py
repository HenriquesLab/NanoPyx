import platform
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm

from .__njit__ import njit_works
from .__opencl__ import opencl_works, devices


class Agent_:

    """
    Base class for the Agent of the Nanopyx Liquid Engine
    Pond, James Pond
    """

    def __init__(
        self,
    ) -> None:
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
        self.os_info = {"OS": platform.platform(), "Architecture": platform.machine()}
        self.cpu_info = {"CPU": platform.processor()}
        self.ram_info = {"RAM": "TBD"}
        self.py_info = {
            "Version": platform.python_version(),
            "Implementation": platform.python_implementation(),
            "Compiler": platform.python_compiler(),
        }

        self.numba_info = {"Numba": njit_works()}
        self.pyopencl_info = {"PyOpenCL": opencl_works(), "Devices": devices}
        self.cuda_info = {"CUDA": "TBD"}
        ### MACHINE INFO ###

        self._current_runs = []
        self._scheduled_runs = []
        self._finished_runs = []

        self.delayed_runtypes = {}  # Store runtypes as keys and their values as (delay_factor, delay_prob)

    def _get_ordered_run_types(self, fn, args, kwargs,_possible_runtypes=[]):
        """@public
        Retrieves an ordered list of run_types for the given args and kwargs
        """

        if not _possible_runtypes:
            _possible_runtypes = fn._run_types.keys()

        # str representation of the arguments and their corresponding 'norm'
        repr_args, repr_norm = fn._get_args_repr_score(*args, **kwargs)
        # dictionary to hold speeds
        fast_avg_speed = {}
        fast_std_speed = {}
        slow_avg_speed = {}
        slow_std_speed = {}
        # fn._benchmarks is a dictionary of dictionaries. The first key is the run_type, the second key is the repr_args
        # Check every run_type for the most similar args
        for run_type in fn._run_types:
            if run_type not in _possible_runtypes:
                continue
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
                if best_repr_args is None:
                    run_info = [0]
                else:
                    run_info = fn._benchmarks[run_type][best_repr_args][1:]

            if None in run_info: # yamls null are read into None python objects
                continue

            if len(run_info) < 2:
                # Fall back to default values
                if "opencl" in run_type:
                    rt = "opencl"
                else:
                    rt = run_type

                best_score = np.inf
                best_repr_args = None
                for repr_args_ in fn._default_benchmarks[rt]:
                    score = np.abs(fn._default_benchmarks[rt][repr_args_][0] - repr_norm)
                    if score < best_score:
                        best_score = score
                        best_repr_args = repr_args_
                run_info = fn._default_benchmarks[rt][best_repr_args][1:]

            run_info = np.array(run_info)
            if len(run_info) > 50:
                run_info = run_info[-50:]

            fast_values = np.partition(run_info, len(run_info) // 2)[: len(run_info) // 2]
            slow_values = np.partition(run_info, len(run_info) // 2)[len(run_info) // 2 :]
            fast_avg_speed[run_type] = np.average(fast_values)
            fast_std_speed[run_type] = np.std(fast_values)
            slow_avg_speed[run_type] = np.average(slow_values)
            slow_std_speed[run_type] = np.std(slow_values)

        return fast_avg_speed, fast_std_speed, slow_avg_speed, slow_std_speed

    def _calculate_prob_of_delay(self, runtimes_history, avg, std):
        """@public
        Calculates the probability that the given run_type is still delayed using historical data
        """

        # Boolean array, True if delay, False if not
        delays = runtimes_history > avg + 4 * std

        model = LogisticRegression()
        model.fit([[state] for state in delays[:-1]], delays[1:])

        return model.predict_proba([[True]])[:, model.classes_.tolist().index(True)][0]

    def _check_delay(self, run_type, runtime, runtimes_history, verbose=True):
        """@public
        Checks if the given run_type ran delayed in the previous run when compared with historical data
        If delayed:
            1. Calculates a probability that this delay is maintained
            2. Stores the delay factor and the probability
        """
        # TODO test 
        threaded_runtypes = ["threaded", "threaded_static", "threaded_dynamic", "threaded_guided"]

        runtimes_history = np.array(runtimes_history)
        if len(runtimes_history) > 50:
            runtimes_history = runtimes_history[-50:]
        fast_values = np.partition(runtimes_history, len(runtimes_history) // 2)[: len(runtimes_history) // 2]
        slow_values = np.partition(runtimes_history, len(runtimes_history) // 2)[len(runtimes_history) // 2 :]

        fast_avg_speed = np.average(fast_values)
        fast_std_speed = np.std(fast_values)
        slow_avg_speed = np.average(slow_values)
        slow_std_speed = np.std(slow_values)

        if run_type in self.delayed_runtypes:
            if runtime < (slow_avg_speed - slow_std_speed) or runtime < (fast_avg_speed + fast_std_speed):
                if "threaded" in run_type:
                    for threaded_run_type in threaded_runtypes:
                        self.delayed_runtypes.pop(threaded_run_type, None)
                else:
                    if run_type in self.delayed_runtypes:
                        self.delayed_runtypes.pop(run_type, None)
                return "Delay off"

        if runtime > fast_avg_speed + 4 * fast_std_speed:
            runtimes_history = np.append(runtimes_history, runtime)
            delay_factor = runtime / fast_avg_speed
            try:
                delay_prob = self._calculate_prob_of_delay(runtimes_history, fast_avg_speed, fast_std_speed)
            except ValueError:
                delay_prob = 0.01
            if verbose:
                print(
                    f"Run type {run_type} was delayed in the previous run. Delay factor: {delay_factor}, Delay probability: {delay_prob}"
                )

            if "threaded" in run_type:
                for threaded_run_type in threaded_runtypes:
                    self.delayed_runtypes[threaded_run_type] = (delay_factor, delay_prob)
            else:
                self.delayed_runtypes[run_type] = (delay_factor, delay_prob)

    def _adjust_times(self, fast_device_times, slow_device_times):
        """@public
        Adjusts the historic avg time of a run_type if it was delayed in previous runs
        """
        adjusted_times = fast_device_times.copy()
        for runtype in self.delayed_runtypes.keys():
            if runtype in fast_device_times.keys():
                delay_factor, delay_prob = self.delayed_runtypes[runtype]
                # Weighted avg by the probability the run_type is still delayed
                # expected_time * P(~delay) + delayed_time * P(delay)
                adjusted_times[runtype] = (
                    fast_device_times[runtype] * (1 - delay_prob)
                    + fast_device_times[runtype] * delay_factor * delay_prob
                )

        return adjusted_times

    def get_run_type(self, fn, args, kwargs,_possible_runtypes=[]):
        """
        Returns the best run_type for the given args and kwargs
        """

        # Get list of run types
        try:
            fast_avg, fast_std, slow_avg, slow_std = self._get_ordered_run_types(fn, args, kwargs,_possible_runtypes)
        except TypeError:
            print(f"There seems to be an error regarding your benchmarks. \n\
To give full control to the agent please ensure that one of the following is true: \n\
\t - You have at least 3 benchmarks for all runtypes using any set of args,kwargs \n\
\t - Provide a set of default benchmarks during the Liquid Engine class creation \n\
Otherwise explicity choose one of the following run_types:")
            print('\t-','\n\t- '.join(fn._run_types.keys()))

            print("The agent will choose a random run_type")
            return random.choices(list(fn._run_types.keys()), k=1)[0]

        # Penalize the average time a run_type had if that run_type was delayed in previous runs
        if len(self.delayed_runtypes.keys()) > 0:
            adjusted_avg = self._adjust_times(fast_avg, slow_avg)

            if sorted(fast_avg, key=fast_avg.get)[0] == sorted(adjusted_avg, key=adjusted_avg.get)[0]:
                return sorted(fast_avg, key=fast_avg.get)[0]

            weights = [(1 / adjusted_avg[k]) ** 2 for k in adjusted_avg]
            weights = weights / np.sum(weights)

            # failsafe
            if sum(weights) == 0:
                weights = [1 for k in adjusted_avg]

            return random.choices(list(adjusted_avg.keys()), weights=weights, k=1)[0]
        else:
            return sorted(fast_avg, key=fast_avg.get)[0]

    def _inform(self, fn, verbose=True):
        """@public
        Informs the Agent that a LE object finished running
        """

        repr_args = fn._last_args
        run_type = fn._last_runtype

        historical_data = fn._benchmarks[run_type][repr_args][1:]

        assert historical_data[-1] == fn._last_time, "Historical data is not consistent with the last runtime"

        if verbose:
            print(f"Agent: {fn._designation} using {run_type} ran in {fn._last_time} seconds")

        if len(historical_data) > 19:
            self._check_delay(run_type, historical_data[-1], historical_data[:-1], verbose=verbose)


Agent = Agent_()
