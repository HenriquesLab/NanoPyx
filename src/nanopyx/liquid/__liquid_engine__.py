import os
import time
from pathlib import Path

import numpy as np
import yaml

from .. import __config_folder__
from . import works


class LiquidEngine:
    """
    Base class for parts of the NanoPyx Liquid Engine
    """

    RUN_TYPE_OPENCL: int = 0
    RUN_TYPE_UNTHREADED: int = 1
    RUN_TYPE_THREADED: int = 2
    RUN_TYPE_THREADED_STATIC: int = 3
    RUN_TYPE_THREADED_DYNAMIC: int = 4
    RUN_TYPE_THREADED_GUIDED: int = 5
    RUN_TYPE_PYTHON: int = 6

    # designations are stored in the config files, to associate runtime statistics
    RUN_TYPE_DESIGNATION = {
        RUN_TYPE_OPENCL: "OpenCL",
        RUN_TYPE_UNTHREADED: "Unthreaded",
        RUN_TYPE_THREADED: "Threaded",
        RUN_TYPE_THREADED_STATIC: "Threaded_static",
        RUN_TYPE_THREADED_DYNAMIC: "Threaded_dynamic",
        RUN_TYPE_THREADED_GUIDED: "Threaded_guided",
        RUN_TYPE_PYTHON: "Python",
    }

    _has_opencl: bool = False
    _has_unthreaded: bool = False
    _has_threaded: bool = False
    _has_threaded_static: bool = False
    _has_threaded_dynamic: bool = False
    _has_threaded_guided: bool = False
    _has_python: bool = False

    _default_fastest: int = RUN_TYPE_OPENCL
    _last_run_type: int | None = None
    _last_run_time: float | None = None

    def __init__(self, clear_config=False):
        """
        Initialize the Liquid Engine
        :param clear_config: whether to clear the config file
        """
        # Check if OpenCL is available
        if not works():
            self._has_opencl = False

        # Load the config file
        base_path = os.path.join(__config_folder__, "liquid")
        os.makedirs(base_path, exist_ok=True)

        self._config_file = os.path.join(base_path, self.__class__.__name__ + ".yml")

        if not clear_config and os.path.exists(self._config_file):
            with open(self._config_file, "r") as f:
                self._cfg = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self._cfg = {}

        # Initialize missing dictionaries in cfg
        for run_type_designation in self.RUN_TYPE_DESIGNATION.values():
            if run_type_designation not in self._cfg:
                self._cfg[run_type_designation] = {}

    def run(self, *args, **kwds):
        """
        Runs the function with the given args and kwargs
        Should be overridden by the any class that inherits from this class
        """
        return self._run(*args, **kwds)

    def benchmark(self, *args, **kwargs):
        """
        Benchmark the different run types
        :param args: args for the run method
        :param kwargs: kwargs for the run method
        :return: None
        """
        # Create some lists to store runtimes and return values of run types
        run_times = {}
        returns = {}
        run_types = []

        if self._has_opencl:
            run_types.append(self.RUN_TYPE_OPENCL)
        if self._has_threaded:
            run_types.append(self.RUN_TYPE_THREADED)
        if self._has_threaded_static:
            run_types.append(self.RUN_TYPE_THREADED_STATIC)
        if self._has_threaded_dynamic:
            run_types.append(self.RUN_TYPE_THREADED_DYNAMIC)
        if self._has_threaded_guided:
            run_types.append(self.RUN_TYPE_THREADED_GUIDED)
        if self._has_unthreaded:
            run_types.append(self.RUN_TYPE_UNTHREADED)
        if self._has_python:
            run_types.append(self.RUN_TYPE_PYTHON)

        for run_type in run_types:
            designation = self.RUN_TYPE_DESIGNATION[run_type]
            r = self._run(*args, run_type=run_type, **kwargs)
            print(f"{designation} run time: {format_time(self._last_run_time)}")
            run_times[run_type] = self._last_run_time
            returns[run_type] = r

        # Sort run_times by value
        speed_sort = []
        for run_type in sorted(run_times, key=run_times.get, reverse=False):
            speed_sort.append(
                (
                    run_times[run_type],
                    self.RUN_TYPE_DESIGNATION[run_type],
                    returns[run_type],
                )
            )

        print(f"Fastest run type: {speed_sort[0][1]}")
        print(f"Slowest run type: {speed_sort[-1][1]}")

        # Compare each run type against each other, sorted by speed
        for i in range(len(speed_sort)):
            for j in range(i + 1, len(speed_sort)):
                if run_times[i] is None or run_times[j] is None:
                    continue

                print(
                    f"{speed_sort[i][1]} is {speed_sort[j][0]/speed_sort[i][0]:.2f} faster than {speed_sort[j][1]}"
                )

        print(f"Run-times log: {self.get_run_times_log()}")

        return speed_sort

    def get_run_times_log(self):
        """
        Get the run times log
        :return: the run times log
        """
        return self._cfg

    def _store_run_time(self, run_type, delta, *args, **kwargs):
        """
        Store the run time in the config file
        :param run_type: the type of run
        :param delta: the time it took to run
        :param args: args for the run method
        :param kwargs: kwargs for the run method
        :return: None
        """
        self._last_run_time = delta
        call_args = self._get_args_repr(args, kwargs)
        run_type_designation = self.RUN_TYPE_DESIGNATION[run_type]

        r = self._cfg[run_type_designation]
        if call_args not in r:
            r[call_args] = [0, 0, 0]

        c = r[call_args]
        # add the time it took to run, later used for average
        c[0] = c[0] + delta
        # add the time it took to run squared, later used for standard deviation
        c[1] = c[1] + delta * delta
        # increment the number of times it was run
        c[2] += 1

        with open(self._config_file, "w") as f:
            yaml.dump(self._cfg, f)

    def _get_fastest_run_type(self, *args, **kwargs):
        """
        Retrieves the fastest run type for the given args and kwargs
        """
        call_args = self._get_args_repr(args, kwargs)
        run_times = [v + 99999999999999999990 for v in self.RUN_TYPE_DESIGNATION.keys()]

        for run_type in self.RUN_TYPE_DESIGNATION.keys():
            run_type_designation = self.RUN_TYPE_DESIGNATION[run_type]
            if (
                run_type_designation in self._cfg
                and call_args in self._cfg[run_type_designation]
            ):
                runtime_sum = self._cfg[run_type_designation][call_args][0]
                runtime_count = self._cfg[run_type_designation][call_args][2]
                run_times[run_type] = runtime_sum / runtime_count

        fastest = run_times.index(min(run_times))
        return fastest

    def _get_cl_code(self, file_name):
        """
        Retrieves the OpenCL code from the corresponding .cl file
        """
        cl_file = os.path.splitext(file_name)[0] + ".cl"
        if not os.path.exists(cl_file):
            cl_file = Path(__file__).parent / file_name

        assert os.path.exists(cl_file), "Could not find OpenCL file: " + cl_file
        return open(cl_file, "r").read()

    def _get_args_repr(self, args, kwargs):
        _args = []
        for arg in args[0]:
            if hasattr(arg, "shape"):
                _args.append(arg.shape)
            else:
                _args.append(arg)
        _kwargs = {}
        for k, v in kwargs.items():
            if hasattr(v, "shape"):
                _kwargs[k] = v.shape
            else:
                _kwargs[k] = v
        return repr((_args, _kwargs))

    ###############
    # _run methods #
    ###############

    def _run(self, *args, run_type=None, **kwargs):
        """
        Runs the function with the given args and kwargs
        :param args: args for the function
        :param run_type: the run type to use, if None use the fastest run type
        :param kwargs: kwargs for the function
        :return: the result of the function
        """

        if run_type is None:
            print(
                f"No run type specified, using fastest run type: {self.RUN_TYPE_DESIGNATION[run_type]}"
            )
            run_type = self._get_fastest_run_type(*args, **kwargs)

        t_start = time.time()
        if run_type == self.RUN_TYPE_OPENCL and self._has_opencl:
            r = self._run_opencl(*args, **kwargs)
        elif run_type == self.RUN_TYPE_UNTHREADED and self._has_unthreaded:
            r = self._run_unthreaded(*args, **kwargs)
        elif run_type == self.RUN_TYPE_THREADED and self._has_threaded:
            r = self._run_threaded(*args, **kwargs)
        elif run_type == self.RUN_TYPE_THREADED_STATIC and self._has_threaded_static:
            r = self._run_threaded_static(*args, **kwargs)
        elif run_type == self.RUN_TYPE_THREADED_DYNAMIC and self._has_threaded_dynamic:
            r = self._run_threaded_dynamic(*args, **kwargs)
        elif run_type == self.RUN_TYPE_THREADED_GUIDED and self._has_threaded_guided:
            r = self._run_threaded_guided(*args, **kwargs)
        elif run_type == self.RUN_TYPE_PYTHON and self._has_python:
            r = self._run_python(*args, **kwargs)
        else:
            raise NotImplementedError("No run method defined")

        self._store_run_time(
            run_type,
            time.time() - t_start,
            args,
            kwargs,
        )
        self._last_run_type = run_type
        return r

    def _run_opencl(*args, **kwargs):
        """
        Runs the OpenCL version of the function
        Should be overridden by the any class that inherits from this class
        """
        pass

    def _run_unthreaded(*args, **kwargs):
        """
        Runs the cython unthreaded version of the function
        Should be overridden by the any class that inherits from this class
        """
        pass

    def _run_threaded(*args, **kwargs):
        """
        Runs the cython threaded version of the function
        Should be overridden by the any class that inherits from this class
        """
        pass

    def _run_threaded_static(*args, **kwargs):
        """
        Runs the cython threaded static version of the function
        Should be overridden by the any class that inherits from this class
        """
        pass

    def _run_threaded_dynamic(*args, **kwargs):
        """
        Runs the cython threaded dynamic version of the function
        Should be overridden by the any class that inherits from this class
        """
        pass

    def _run_threaded_guided(*args, **kwargs):
        """
        Runs the cython threaded guided version of the function
        Should be overridden by the any class that inherits from this class
        """
        pass

    def _run_python(*args, **kwargs):
        """
        Runs the python version of the function
        Should be overridden by the any class that inherits from this class
        """
        pass


def format_time(t: float):
    """
    Formats a time in seconds to a human readable string
    :param t: the time in seconds
    :return: a human readable string
    """
    if t < 1e-6:
        return f"{t * 1e9:.2f}ns"
    elif t < 1:
        return f"{t * 1000:.2f}ms"
    else:
        return f"{t:.2f}s"
