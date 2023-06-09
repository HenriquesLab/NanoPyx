import os
import yaml
import timeit
import datetime
import inspect
from functools import partial
from pathlib import Path

import random
import numpy as np

from .__njit__ import njit_works
from .__opencl__ import opencl_works, devices

__home_folder__ = os.path.expanduser("~")
__benchmark_folder__ = os.path.join(__home_folder__, ".nanopyx")
if not os.path.exists(__benchmark_folder__):
    os.makedirs(__benchmark_folder__)

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(f'{__name__}.log')
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(process)d - %(processName)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

logger.addHandler(fh)

class LiquidEngine:

    """
    Base class for parts of the Nanopyx Liquid engine
    """


    def __init__(self, clear_benchmarks:bool=False, testing:bool=False, dynamic_runtypes=True,
                 opencl_:bool = False, unthreaded_:bool = False,
                 threaded_:bool = False, threaded_static_:bool = False,
                 threaded_dynamic_:bool = False, threaded_guided_:bool = False,
                 python_:bool=False, njit_:bool=False) -> None:
        """
        Initialize the Liquid Engine

        1. Checks available implementations:
            Checks whether OpenCL is available 
            Checks whether Numba is available 
        2. Builds an dictionary with strings as keys that point to the corresponding method
            Implementations that are NOT available do not appear in this dictionary
        3. Creates a path to store the benchmark .yml file in the home folder
            (e.g. ~/.nanopyx/liquid/_le_interpolation_nearest_neighbor.cpython-310-darwin/ShiftAndMagnify.yml)
        4. Loads the benchmark file (if it exists) as a dictionary and checks pre existing benchmarks
            The benchmark file is read as dict of dicts. 
            BENCHMARK DICT 
                |- RUN_TYPE #1
                |      |- ARGS_REPR #1
                |      |      |- [sum, sum_squared, arg_norm, [success timestamps], [fail timestamps]]
                |      |- ARGS_REPR #2  
                |      |      |- [sum, sum_squared, arg_norm, [success timestamps], [fail timestamps]]
                |      (...)
                |- RUN_TYPE #2 
                (...)
            If necessary, creates empty dictionaries in the benchmark parent dict for each run type (e.g. 'Threaded', 'OpenCL', 'Numba')
            These per runtype dictionaries are going to be populated by the benchmarks

        :param clear_benchmark: whether to clear the config file of previous data.
        :param testing: whether to run in testing mode. testing mode keeps track of results of each method
        during a benchmark. this has a BIG memory footprint which can lead to unexpected crashes when using big datasets
        :param dynamic_runtypes: whether the runtype is randomly based on average time taken in the past or if the runtype
        is simply chosen as the lowest time taken in the past
        """
        
        # Start by checking available run types
        self._run_types = {}
        if opencl_ and opencl_works():
            for d in devices:
                self._run_types[f"OpenCL_{d['device'].name}"] = partial(self._run_opencl, device=d)
        if threaded_:
            self._run_types["Threaded"] = self._run_threaded
        if unthreaded_:
            self._run_types["Unthreaded"] = self._run_unthreaded
        if threaded_static_:
            self._run_types["Threaded_static"] = self._run_threaded_static
        if threaded_dynamic_:
            self._run_types["Threaded_dynamic"] = self._run_threaded_dynamic
        if threaded_guided_:
            self._run_types["Threaded_guided"] = self._run_threaded_guided
        if python_:
            self._run_types["Python"] = self._run_python
        if njit_ and njit_works():
            self._run_types["Numba"] = self._run_njit
            # Try to trigger early compilation
            try:
                self._run_njit()
            except TypeError:
                print("Consider adding default arguments to the njit implementation to trigger early compilation")

        
        # benchmarks file path
        # e.g.: ~/.nanopyx/liquid/_le_interpolation_nearest_neighbor.cpython-310-darwin/ShiftAndMagnify.yml
        base_path = os.path.join(
            __benchmark_folder__,
            "liquid",
            os.path.split(os.path.splitext(inspect.getfile(self.__class__))[0])[1])
        os.makedirs(base_path, exist_ok=True)
        self._benchmark_filepath = os.path.join(base_path,self.__class__.__name__+".yml")

        # Load config file if it exists, otherwise create an empty config
        if not clear_benchmarks and os.path.exists(self._benchmark_filepath):
            with open(self._benchmark_filepath) as f:
                self._benchmarks = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self._benchmarks = {}

        # check if the cfg dictionary has a key for every run type
        for run_type_designation in self._run_types.keys():
            if run_type_designation not in self._benchmarks:
                self._benchmarks[run_type_designation] = {}

        # Testing mode boolean
        self.testing = testing

        # Are run_types probabilistic?
        self.dynamic_runtypes = dynamic_runtypes    

        # Storage attributes to help benchmarking
        self._last_runtype = None
        self._last_runtime = None

    def _run(self, *args, run_type:str=None, **kwargs):
        """
        Runs the function with the given args and kwargs

        The code above does the following:
        1. Check the specified run_type
            - if None, tries to run the fastest recorded one
            - if str checks if the run type exists otherwise raise a NotImplementedError
        2. It will run the _run_{run_type} function
            - it will also store the run time
            - it will also store the last run type
        4. It will return the result

        :param args: args for the function
        :param run_type: the run type to use, if None use the fastest run type
        :param kwargs: kwargs for the function
        :return: the result of the function
        """
        
        if run_type is None:
            run_type = self._get_fastest_run_type(*args, **kwargs)

        if run_type not in self._benchmarks:
            print(f"Unexpected run type {run_type}")
            raise NotImplementedError
        
        # try to run
        try:
            t_start = timeit.default_timer()
            result = self._run_types[run_type](*args, **kwargs)
            t2run = timeit.default_timer()-t_start
        except Exception as e:
            print(f"Unexpected error while trying to run {run_type}")
            print(e)
            print("Please try again with another run type")
            result = None
            t2run = None

        self._store_run_time(run_type, t2run, *args, **kwargs)

        return result
    
    def _get_fastest_run_type(self, *args, **kwargs):
        """
        Retrieves the fastest run type for the given args and kwargs

        1. Get the fastest run type for the specific args and kwargs
        2. If the args and kwargs are not in the benchmarked yet, it will find the most similar args and kwargs
        3. Return the run time with the lowest avg time

        :return: the fastest run type
        :rtype: str (run type designation)
        """

        # Default fastest can be changed in the future
        default_fastest = list(self._run_types.keys())[0]

        # str representation of the arguments and their corresponding 'norm'
        repr_args, repr_norm = self._get_args_repr_norm(*args, **kwargs)

        # dictionary to hold speeds
        speed = {}

        # Check every run_type for similar args
        for run_type in self._run_types:

            if repr_args not in self._benchmarks[run_type]:
                # Never seen these arguments
                # Find closest match
                #[sum, sum_squared, arg_norm, [success timestamps], [fail timestamps]]
                similar_args = None
                best_args_similarity = np.inf
                for call_args in self._benchmarks[run_type]:
                    args_similarity = np.abs(self._benchmarks[run_type][call_args][2]-repr_norm)
                    if args_similarity<best_args_similarity:
                        best_args_similarity = args_similarity
                        similar_args = call_args
                run_info = self._benchmarks[run_type][similar_args]
            else:
                # I have seen these arguments
                run_info = self._benchmarks[run_type][repr_args]

            # [sum, sum_squared, arg_norm, [success timestamps], [fail timestamps]]
            runtime_sum = run_info[0]
            #runtime_sqsum = run_info[1]
            #runtime_norm = run_info[2]
            runtime_count = len(run_info[3])
            #runtime_fails = len(run_info[4])

            runtime_avgspeed = runtime_sum / runtime_count
            speed[run_type] = runtime_avgspeed

        # empty dict?
        if not speed: 
            return default_fastest
        # dict not empty with dynamic runtypes
        elif speed and self.dynamic_runtypes: 
            weights = [(1/speed[k])**2 for k in speed]
            return random.choices(list(speed.keys()), weights=weights, k=1)[0]
        # dict not empty but without dynamic runtypes
        else:
            return sorted(speed, key=speed.get, reverse=True)
            
    def _store_run_time(self, run_type, t2run, *args, **kwargs):
        """
        Store the run time in the config file
        :param run_type: the type of run
        :param t2run: the time it took to run
        :param args: args for the run method
        :param kwargs: kwargs for the run method
        :return: None
        """

        call_args, norm = self._get_args_repr_norm(*args, **kwargs)  # Get the call args

        # Check if the run type has been run, and if not create empty info
        run_type_benchs = self._benchmarks[run_type]
        if call_args not in run_type_benchs:
            # [sum, sum_squared, arg_norm, [success timestamps], [fail timestamps]]
            run_type_benchs[call_args] = [0, 0, norm, [], []]

        # Get the run info
        c = run_type_benchs[call_args]
        # timestamp
        ct = datetime.datetime.now()

        # if run failed, t2run is None
        if t2run is None:
            c[4].append(ct)
        else:
            # add the time it took to run, later used for average
            c[0] = c[0] + t2run
            # add the time it took to run squared, later used for standard deviation
            c[1] = c[1] + t2run * t2run
            # increment the number of times it was run
            c[3].append(ct)

        # Check if the norm if consistent
        assert c[2] == norm

        self._last_runtype = run_type
        self._last_runtime = t2run

        self._dump_run_times()

    def _dump_run_times(self,):
        """We might need to wrap this into a multiprocessing.Queue if we find it blocking"""
        with open(self._benchmark_filepath, "w") as f:
            yaml.dump(self._benchmarks, f)

    def _get_args_repr_norm(*args, **kwargs):
        """
        Get a string representation of the args and kwargs and corresponding 'norm'
        The idea is that similar args have closer 'norms'. Fuzzy logic

        The code does the following:
        1. It converts any args that are floats or ints to "number()" strings, and any args that are tensors to "shape()" strings
        2. It converts any kwargs that are floats or ints to "number()" strings, and any kwargs that are tensors to "shape()" strings
        3. The 'norm' is given by the product of all the floats or ints and all the shape sizes. 

        :return: the string representation of the args and kwargs
        :rtype: str
        """
        _norm = 1
        _args = []
        for arg in args:
            if type(arg) in (float, int):
                _args.append(f"number({arg})")
                _norm *= arg
            elif hasattr(arg, "shape"):
                _args.append(f"shape{arg.shape}")
                _norm *= arg.size
            else:
                _args.append(arg)

        _kwargs = {}
        for k, v in kwargs.items():
            if type(v) in (float, int):
                _kwargs[k] = f"number({v})"
                _norm *= arg
            if hasattr(v, "shape"):
                _kwargs[k] = f"shape{arg.shape}"
                _norm *= arg.size
            else:
                _kwargs[k] = v

        return repr((_args, _kwargs)), _norm

    def benchmark(self,*args, **kwargs):
        """
        1. Run each available run type and record the run time and return value
        2. Sort the run times from fastest to slowest
        3. Store the benchmark results in the benchmark yaml file
        4. Compare each run type against each other, sorted by speed
        5. Print the results

        :param args: args for the run method
        :param kwargs: kwargs for the run method
        :return:  a list of tuples containing the run time, run type name and optionally the return values
        :rtype: [[run_time, run_type_name, return_value], ...]
        """

        # Create some lists to store runtimes and return values of run types
        run_times = {}
        returns = {}

        # Run each run type and record the run time and return value
        for run_type in self._run_types:

            r = self._run(*args, run_type=run_type, **kwargs)

            if r is None:
                run_times[run_type] = np.inf
            else:
                run_times[run_type] = self._last_runtime
            
            if self.testing:
                returns[run_type] = r
            else:
                returns[run_type] = None
        
        # Sort run_times by value
        speed_sort = []
        for run_type in sorted(run_times, key=run_times.get, reverse=False):
            speed_sort.append(
                (
                    run_times[run_type],
                    run_type,
                    returns[run_type],
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

        return speed_sort

    def _get_cl_code(self, file_name, cl_dp):
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

    #####################################################
    #                   RUN METHODS                     #
    # THESE SHOULD ALWAYS BE OVERRIDEN BY CHILD CLASSES #
    #####################################################

    def run(self, *args, **kwargs):
        """
        Runs the function with the given args and kwargs
        Should be overridden by the any class that inherits from this class
        """
        return self._run(*args, **kwargs)

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
            