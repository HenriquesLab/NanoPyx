import os
import time
import difflib
from pathlib import Path
import inspect
import random

import numpy as np
import yaml

from .. import __config_folder__
from .__njit__ import njit_works
from .__opencl__ import opencl_works, cl_dp

# flake8: noqa: E501


class LiquidEngine:
    """
    Base class for parts of the NanoPyx Liquid Engine
    """

    # the following variables are used to identify if each type of run is available
    _has_opencl: bool = False
    _has_unthreaded: bool = False
    _has_threaded: bool = False
    _has_threaded_static: bool = False
    _has_threaded_dynamic: bool = False
    _has_threaded_guided: bool = False
    _has_python: bool = False
    _has_njit: bool = False

    _random_testing: bool = True  # used to sometimes try different run types when using the run(...) method
    _show_info: bool = False  # print what's going on

    _engine_parts: dict = {}  # the parts of the Liquid Engine
    _default_fastest: str = None  # the default engine part to use when using the run(...) method
    _last_engine_part_designation: str = None  # the last engine part used
    _last_run_time: float = None  # the time the last run took

    def __initialize_parts__(self):
        if self._has_opencl and opencl_works():
            self._engine_parts["OpenCL"] = self._run_opencl
        if self._has_unthreaded:
            self._engine_parts["Unthreaded"] = self._run_unthreaded
        if self._has_threaded:
            self._engine_parts["Threaded"] = self._run_threaded
        if self._has_threaded_static:
            self._engine_parts["Threaded_static"] = self._run_threaded_static
        if self._has_threaded_dynamic:
            self._engine_parts["Threaded_dynamic"] = self._run_threaded_dynamic
        if self._has_threaded_guided:
            self._engine_parts["Threaded_guided"] = self._run_threaded_guided
        if self._has_python:
            self._engine_parts["Python"] = self._run_python
        if self._has_njit and njit_works():
            self._engine_parts["Numba"] = self._run_njit

        # set default fastest run type
        if "OpenCL" in self._engine_parts:
            self._default_fastest = "OpenCL"
        else:
            self._default_fastest = "Threaded"

    def __init__(self, clear_config=False):
        """
        Initialize the Liquid Engine

        The code does the following:
        1. Checks whether OpenCL is available (by running a simple OpenCL kernel)
        2. Checks whether Numba is available (by running the njit decorator)
        3. Creates a path to store the config file (e.g. ~/.nanopyx/liquid/_le_interpolation_nearest_neighbor.cpython-310-darwin/ShiftAndMagnify.yml)
        4. Loads the config file (if it exists)
        5. Creates empty dictionaries for each run type (e.g. 'Threaded', 'OpenCL', 'Numba')

        :param clear_config: whether to clear the config file
        """
        # Initialize the parts of the Liquid Engine
        self.__initialize_parts__()

        # Load the config file
        # e.g.: ~/.nanopyx/liquid/_le_interpolation_nearest_neighbor.cpython-310-darwin/ShiftAndMagnify.yml
        base_path = os.path.join(
            __config_folder__,
            "liquid",
            os.path.split(os.path.splitext(inspect.getfile(self.__class__))[0])[1],
        )
        os.makedirs(base_path, exist_ok=True)

        # set path to config file
        self._config_file = os.path.join(base_path, self.__class__.__name__ + ".yml")

        # Load config file if it exists, otherwise create an empty config
        if not clear_config and os.path.exists(self._config_file):
            with open(self._config_file) as f:
                self._cfg = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self._cfg = {}

        # Initialize missing dictionaries in cfg
        for engine_part_designation in self._engine_parts:
            if engine_part_designation not in self._cfg:
                self._cfg[engine_part_designation] = {}

    def is_opencl_enabled(self):
        """
        Returns whether OpenCL is enabled
        :return: whether OpenCL is enabled
        :rtype: bool
        """
        return self._has_opencl

    def is_njit_enabled(self):
        """
        Returns whether Numba is enabled
        :return: whether Numba is enabled
        :rtype: bool
        """
        return self._has_njit

    def set_opencl_enabled(self, enabled: bool = True):
        """
        Sets whether OpenCL is enabled
        :param enabled: whether OpenCL is enabled
        """
        self._has_opencl = enabled

    def set_opencl_disabled_if_no_double_support(self):
        """
        Sets whether OpenCL is enabled
        :param enabled: whether OpenCL is enabled
        """
        if not cl_dp:
            self._has_opencl = False

    def set_njit_enabled(self, enabled: bool = True):
        """
        Sets whether Numba is enabled
        :param enabled: whether Numba is enabled
        """
        self._has_njit = enabled

    def run(self, *args, **kwds):
        """
        Runs the function with the given args and kwargs
        Should be overridden by the any class that inherits from this class
        """
        return self._run(*args, **kwds)

    def benchmark(self, *args, **kwargs):
        """
        Benchmark the different run types

        The code does the following:
        1. Create a list of run types to benchmark
        2. Run each run type and record the run time and return value
        3. Sort the run times from fastest to slowest
        4. Compare each run type against each other, sorted by speed
        5. Print the results

        :param args: args for the run method
        :param kwargs: kwargs for the run method
        :return:  a list of tuples containing the run time, run type name and return value
        :rtype: [[run_time, engine_part_designation, return_value], ...]
        """
        # Create some lists to store runtimes and return values of run types
        run_times = {}
        returns = {}

        # Create a list of run types to benchmark
        if "Numba" in self._engine_parts:
            # Trigger compilation
            try:
                self._engine_parts["Numba"]()
            except TypeError:
                print("Consider adding default arguments to njit implementation to trigger early compilation")

        # Run each run type and record the run time and return value
        for engine_part_designation in self._engine_parts:
            r = self._run(*args, engine_part_designation=engine_part_designation, **kwargs)
            run_times[engine_part_designation] = self._last_run_time
            returns[engine_part_designation] = r
            mean, std, n = self.get_mean_std_run_time(engine_part_designation, *args, **kwargs)
            self._print(
                f"{engine_part_designation} run time: {format_time(self._last_run_time)}; "
                + f"mean: {format_time(mean)}; std: {format_time(std)}; runs: {n}"
            )

        # Sort run_times by value
        speed_sort = []
        for engine_part_designation in sorted(run_times, key=run_times.get, reverse=False):
            speed_sort.append(
                (
                    run_times[engine_part_designation],
                    engine_part_designation,
                    returns[engine_part_designation],
                )
            )

        print(f"Fastest run type: {speed_sort[0][1]}")
        print(f"Slowest run type: {speed_sort[-1][1]}")

        # Compare each run type against each other, sorted by speed
        for i in range(len(speed_sort)):
            if i not in run_times or run_times[i] is None:
                continue
            for j in range(i + 1, len(speed_sort)):
                if j not in run_times or run_times[j] is None:
                    continue

                print(f"{speed_sort[i][1]} is {speed_sort[j][0]/speed_sort[i][0]:.2f} faster than {speed_sort[j][1]}")

        self._print(f"Run-times log: {self.get_run_times_log()}")
        print(f"Recorded fastest: {self._get_fastest_run(*args, **kwargs)}")

        return speed_sort

    def get_mean_std_run_time(self, engine_part_designation: str, *args, **kwargs):
        """
        Get the mean and standard deviation of the run time for the given run type
        :param run_type: the run type
        :param args: args for the run method
        :param kwargs: kwargs for the run method
        :return: the mean, standard deviation of the run time and the number of runs
        """

        # Get the call args
        call_args = self._get_args_repr(*args, **kwargs)

        # Check if the run type has been run
        r = self._cfg[engine_part_designation]
        # If not, return None
        if call_args not in r:
            return None, None, None

        # Get the run times
        c = r[call_args]
        sum = c[0]  # Sum of run times
        sum_sq = c[1]  # Sum of squared run times (for std)
        n = c[2]  # Number of runs
        mean = sum / n
        if (n - 1) > 0:
            std = np.sqrt((sum_sq - n * mean**2) / (n - 1))
        else:
            std = 0
        return mean, std, n

    def get_run_times_log(self):
        """
        Get the run times log
        :return: the run times log
        """
        return self._cfg

    def set_show_info(self, show_info: bool):
        """
        Set whether to show info
        :param show_info: whether to show info
        :return: None
        """
        self._show_info = show_info

    def _store_run_time(self, engine_part_designation, delta, *args, **kwargs):
        """
        Store the run time in the config file
        :param engine_part_designation: designation of the engine part
        :param delta: the time it took to run
        :param args: args for the run method
        :param kwargs: kwargs for the run method
        :return: None
        """
        self._last_run_time = delta  # Store the last run time
        call_args = self._get_args_repr(*args, **kwargs)  # Get the call args

        # Check if the run type has been run
        r = self._cfg[engine_part_designation]
        if call_args not in r:
            r[call_args] = [0, 0, 0]

        # Get the run times
        c = r[call_args]
        # add the time it took to run, later used for average
        c[0] = c[0] + delta
        # add the time it took to run squared, later used for standard deviation
        c[1] = c[1] + delta * delta
        # increment the number of times it was run
        c[2] += 1

        self._print(
            f"Storing run time: {delta} (m={c[0]/c[2]:.2f},n={c[2]})",
            call_args,
            engine_part_designation,
        )

        with open(self._config_file, "w") as f:
            yaml.dump(self._cfg, f)

    def _get_fastest_run(self, *args, **kwargs) -> int:
        """
        Retrieves the fastest run type for the given args and kwargs

        The code does the following:
        1. Get the fastest run type based on the args and kwargs
        2. If the args and kwargs are not in the config, it will find the most similar args and kwargs
        3. It will also use the runtime of the function to determine the fastest run type

        :return: the fastest engine part designation
        :rtype: str
        """

        fastest = self._default_fastest
        speed_and_type = []

        call_args = self._get_args_repr(*args, **kwargs)
        # print(call_args)

        for engine_part_designation in self._engine_parts:

            if engine_part_designation not in self._cfg:
                self._cfg[engine_part_designation] = {}
                continue

            if call_args not in self._cfg[engine_part_designation] and len(self._cfg[engine_part_designation]) > 0:
                # find the most similar call_args by score
                score_current = self._get_args_score(call_args)
                delta_best = 1e99
                similar_call_args: str = None
                for _call_args in self._cfg[engine_part_designation]:
                    score = self._get_args_score(_call_args)
                    delta = abs(score - score_current)
                    if delta < delta_best:
                        delta_best = delta
                        similar_call_args = _call_args
                if similar_call_args is not None:
                    call_args = similar_call_args
                else:
                    # find the most similar call_args by string similarity
                    similar_args = difflib.get_close_matches(call_args, self._cfg[engine_part_designation].keys())
                    if len(similar_args) > 0:
                        call_args = similar_args[0]

            if call_args in self._cfg[engine_part_designation]:
                run_info = self._cfg[engine_part_designation][call_args]
                runtime_sum = run_info[0]
                runtime_count = run_info[2]
                speed = runtime_count / runtime_sum
                speed_and_type.append((speed, engine_part_designation))
                self._print(f"{engine_part_designation} run time: {speed:.2f} runs/s")

        if len(speed_and_type) == 0:
            return fastest

        elif self._random_testing:
            # randomly choose a run type based on a squared speed weight
            engine_part_designation = [x[1] for x in speed_and_type]
            weights = [x[0] ** 2 for x in speed_and_type]
            return random.choices(engine_part_designation, weights=weights, k=1)[0]

        else:
            # just return the fastest
            return sorted(speed_and_type, key=lambda x: x[0], reverse=True)[0][1]

    def _get_cl_code(self, file_name):
        """
        Retrieves the OpenCL code from the corresponding .cl file
        """
        cl_file = os.path.splitext(file_name)[0] + ".cl"
        if not os.path.exists(cl_file):
            cl_file = Path(__file__).parent / file_name

        assert os.path.exists(cl_file), "Could not find OpenCL file: " + cl_file

        kernel_str = open(cl_file).read()

        if not cl_dp:
            kernel_str = kernel_str.replace("double", "float")

        return kernel_str

    def _get_args_repr(self, *args, **kwargs) -> str:
        """
        Get a string representation of the args and kwargs

        The code does the following:
        1. It uses the "repr" function to get a string representation of the args and kwargs
        2.  It converts any args that are floats or ints to "number()" strings, and any args that are tensors to "shape()" strings
        3.  It converts any kwargs that are floats or ints to "number()" strings, and any kwargs that are tensors to "shape()" strings

        :return: the string representation of the args and kwargs
        :rtype: str
        """
        # print("Args: ", args)
        # print("Kwargs: ", kwargs)
        _args = []
        for arg in args:
            if type(arg) in (float, int):
                _args.append(f"number({arg})")
            elif hasattr(arg, "shape"):
                _args.append(f"shape{arg.shape}")
            else:
                _args.append(arg)
        _kwargs = {}
        for k, v in kwargs.items():
            if type(v) in (float, int):
                _kwargs[k] = f"number({v})"
            if hasattr(v, "shape"):
                _kwargs[k] = f"shape{arg.shape}"
            else:
                _kwargs[k] = v
        return repr((_args, _kwargs))

    def _get_args_shapes_numbers(self, txt: str):
        """
        Get the shapes and numbers from the string representation of the args and kwargs

        The code does the following:
        1. Finds all shape values in the text
        2. Finds all number values in the text
        3. Converts the found values to float

        :param txt: the string representation of the args and kwargs
        :return: a tuple of the shapes and numbers
        """
        shapes = []
        numbers = []

        # example (['shape(3, 64, 32)', 'shape(3,)', 'shape(3,)', 'number(4.0)', 'number(4.0)'], {})

        # find shape values
        _txt = txt
        marker = 0
        while 1:
            start = _txt.find("shape(", marker)
            end = _txt.find(")", start)
            if start == -1 or end == -1:
                break
            elements = _txt[start + 6 : end].split(",")
            for element in elements:
                if element.strip() != "":
                    shapes.append(float(element))
            marker = end

        # find number values
        _txt = txt
        marker = 0
        while 1:
            start = _txt.find("number(", marker)
            end = _txt.find(")", start)
            if start == -1 or end == -1:
                break
            numbers.append(float(_txt[start + 7 : end]))
            marker = end

        return shapes, numbers

    def _get_args_score(self, txt: str) -> float:
        """
        Get the score for the given args and kwargs
        :param txt: the string representation of the args and kwargs
        :return: the score
        """
        shapes, numbers = self._get_args_shapes_numbers(txt)
        score = 1
        if len(shapes) > 0:
            score = score * np.prod(shapes)
        if len(numbers) > 0:
            score = score * np.prod(numbers)
        return score

    def _print(self, *args, **kwargs):
        """
        Prints the args and kwargs
        """
        if self._show_info:
            print(*args, **kwargs)

    ################
    # _run methods #
    ################

    def _run(self, *args, engine_part_designation: str = None, **kwargs):
        """
        Runs the function with the given args and kwargs

        The code above does the following:
        1. Check if you have the specified run_type
            - if you do not, it will raise a NotImplementedError
        2. Check if you have the _run_XXX function
            - if you do not, it will raise a NotImplementedError
        3. It will run the _run_XXX function
            - it will also store the run time
            - it will also store the last run type
        4. It will return the result

        :param args: args for the function
        :param engine_part_designation: designation for the engine part to use, if None use the fastest part
        :param kwargs: kwargs for the function
        :return: the result of the function
        """

        if engine_part_designation is None:
            engine_part_designation = self._get_fastest_run(*args, **kwargs)
            self._print(f"Using run type: {engine_part_designation}")

        t_start = time.time()
        r = self._engine_parts[engine_part_designation](*args, **kwargs)
        t_end = time.time()

        self._store_run_time(
            engine_part_designation,
            t_end - t_start,
            *args,
            **kwargs,
        )
        self._last_engine_part_designation = engine_part_designation
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

    def _run_njit(*args, **kwargs):
        """
        Runs the njit version of the function
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
