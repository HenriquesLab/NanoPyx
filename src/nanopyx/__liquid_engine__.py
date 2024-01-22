import os
import timeit
import yaml
import datetime
import inspect
import warnings
from functools import partial, reduce
from itertools import combinations
from pathlib import Path

from importlib_resources import files

import numpy as np

# This will in the future come from the Agent
from .__njit__ import njit_works
from .__dask__ import dask_works
from .__transonic__ import transonic_works
from .__cuda__ import cuda_works
from .__opencl__ import opencl_works, devices, cl

__home_folder__ = os.path.expanduser("~")
__benchmark_folder__ = os.path.join(__home_folder__, ".nanopyx")
if not os.path.exists(__benchmark_folder__):
    os.makedirs(__benchmark_folder__)

from .__agent__ import Agent  # noqa: E402

from .core.analysis.pearson_correlation import pearson_correlation


class LiquidEngine:

    """
    Base class for parts of the Nanopyx Liquid Engine
    Vroom Vroom
    """

    def __init__(
        self,
        testing: bool = False,
        opencl_: bool = False,
        unthreaded_: bool = False,
        threaded_: bool = False,
        threaded_static_: bool = False,
        threaded_dynamic_: bool = False,
        threaded_guided_: bool = False,
        python_: bool = False,
        njit_: bool = False,
        dask_: bool = False,
        transonic_: bool = False,
        cuda_: bool = False,
        clear_benchmarks: bool = False,
    ) -> None:
        """
        Initialize the Liquid Engine
        The Liquid Engine base class is inherited by children classes that implement specific methods

        Engine responsabilities:
        1. Store implemented run types;
        2. Handle previous benchmarks and I/O;
        2. When queried, benchmark all available run types;
        3. Run a specific method using a selected run type;

        Benchmark files have the following format:
        The benchmark file is read as dict of dicts.
            BENCHMARK DICT FOR A SPECIFIC METHOD
                |- RUN_TYPE #1
                |      |- ARGS_REPR #1
                |      |      |- [score, t2run#1, t2run#2, t2run#3, ...] last are newer. nan means fail
                |      |- ARGS_REPR #2
                |      |      |- [score, t2run#1, t2run#2, t2run#3, ...] last are newer. nan means fail
                |      (...)
                |- RUN_TYPE #2
                (...)
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
        if dask_ and dask_works():
            self._run_types["Dask"] = self._run_dask
        if transonic_ and transonic_works():
            self._run_types["Transonic"] = self._run_transonic
        if cuda_ and cuda_works():
            self._run_types["Cuda"] = self._run_cuda

        self.testing = testing
        self.mem_div = 1

        # benchmarks file path
        # e.g.: ~/.nanopyx/liquid/_le_interpolation_nearest_neighbor.cpython-310-darwin/ShiftAndMagnify.yml
        base_path = os.path.join(
            __benchmark_folder__, "liquid", os.path.split(os.path.splitext(inspect.getfile(self.__class__))[0])[1]
        )
        os.makedirs(base_path, exist_ok=True)
        self._benchmark_filepath = os.path.join(base_path, self.__class__.__name__ + ".yml")

        # Load config file if it exists, otherwise create an empty config
        if not clear_benchmarks and os.path.exists(self._benchmark_filepath):
            with open(self._benchmark_filepath) as f:
                self._benchmarks = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self._benchmarks = {}

        # check if the cfg dictionary has a key for every available run type
        for run_type_designation in self._run_types.keys():
            if run_type_designation not in self._benchmarks:
                self._benchmarks[run_type_designation] = {}

        # helper attribute for benchmarking function
        self._last_args = None
        self._last_runtype = None
        self._last_time = None

        self.Agent = Agent

        # load defaults
        try:
            self._default_benchmarks = yaml.safe_load(files(f'liquid_benchmarks.{inspect.getmodule(self.__class__).__name__.split(".")[-1]}').joinpath(self.__class__.__name__ + ".yml").read_text())
        except:
            self._default_benchmarks = []
        

    def _run(self, *args, run_type=None, **kwargs):
        """
        Runs the function with the given args and kwargs

        The code above does the following:
        1. Check the specified run_type
            - if str checks if the run type exists otherwise raise a NotImplementedError
        2. It will run the _run_{run_type} function
        3. It will return the result and the time taken to run

        :param args: args for the function
        :param run_type: the run type to use
        :param kwargs: kwargs for the function
        :return: the result and time taken
        """

        if run_type is None:
            print("Querying the Agent...")
            run_type = self.Agent.get_run_type(self, args, kwargs)
        elif run_type not in self._run_types:
            print(f"Unexpected run type {run_type}")
            print("Querying the Agent...")
            run_type = self.Agent.get_run_type(self, args, kwargs)
            print(f"Agent chose:{run_type}") 

        # try to run
        try:
            if self.mem_div > 999:
                raise ValueError(f"Maxmimum memory division factor achieved, can not try any longer with {run_type}. Use a smaller input or a different run_type")
            t_start = timeit.default_timer()
            result = self._run_types[run_type](*args, **kwargs)
            t2run = timeit.default_timer() - t_start
            arg_repr, arg_score = self._get_args_repr_score(*args, **kwargs)
            self._store_results(arg_repr, arg_score, run_type, t2run)

            self._last_time = t2run
            self._last_args = arg_repr
            self._last_runtype = run_type

            self.Agent._inform(self)

        except (cl.MemoryError, cl.LogicError) as e:
            print("Found: ", e)
            print("Reducing maximum buffer size and trying again...")
            self.mem_div += 1
            kwargs["mem_div"] = self.mem_div
            result = self._run(*args, run_type=run_type, **kwargs)
        except cl.Error as e:
            if e.__str__() == "Buffer size is larger than device maximum memory allocation size":
                print("Found: ", e)
                print("Reducing maximum buffer size and trying again...")
                self.mem_div += 1
                kwargs["mem_div"] = self.mem_div
                result = self._run(*args, run_type=run_type, **kwargs)
            else:
                print(f"Unexpected error while trying to run {run_type}")
                print(e)
                print("Please try again with another run type")
                result = None
        except Exception as e:
            print(f"Unexpected error while trying to run {run_type}")
            print(e)
            print("Please try again with another run type")
            result = None

        self.mem_div = 1
        return result

    def benchmark(self, *args, **kwargs):
        """
        1. Run each available run type and record the run time and return value
        2. Sort the run times from fastest to slowest
        3. Compare each run type against each other, sorted by speed

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

            run_times[run_type] = self._last_time

            if self.testing:  # Store return values if testing
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
        different_runtypes = []
        for pair in combinations(speed_sort, 2):
            print(f"{pair[0][1]} is {pair[1][0]/pair[0][0]:.2f}x faster than {pair[1][1]}")
            if self.testing:
                if self._compare_runs(pair[0][2], pair[1][2]):
                    print(f"{pair[0][1]} and {pair[1][1]} have similar outputs!")
                else:
                    warnings.warn(f"WARNING: outputs of {pair[0][1]} and {pair[1][1]} don't match!")
                    different_runtypes.append(set([pair[0][1],pair[1][1]]))
        if len(different_runtypes)<=len(self._run_types)-1:
            try:
                common_runtype = reduce(lambda a,b: a&b,different_runtypes)
            except(TypeError):
                common_runtype = {} 
            if common_runtype:
                warnings.warn(f"WARNING: disabling {list(common_runtype)[0]} for this set of arguments!")
                arg_repr, arg_score = self._get_args_repr_score(*args, **kwargs)
                self._store_results(arg_repr, arg_score, list(common_runtype)[0], None) # None saves to null in yamls

        return speed_sort

    def _compare_runs(self, output_1, output_2):
        if output_1.ndim > 2:
            pcc = 0
            for i in range(output_1.shape[0]):
                pcc += pearson_correlation(output_1[i, :, :], output_2[i, :, :])
            pcc /= output_1.shape[0]
        else:
            pcc = pearson_correlation(output_1, output_2)

        if pcc > 0.8:
            return True
        else:
            return False

    def _get_cl_code(self, file_name, cl_dp):
        """
        Retrieves the OpenCL code from the corresponding .cl file
        """
        cl_file = os.path.splitext(file_name)[0] + ".cl"

        if not os.path.exists(cl_file):
            cl_file = Path(os.path.abspath(inspect.getfile(self.__class__))).parent / file_name

        assert os.path.exists(cl_file), "Could not find OpenCL file: " + str(cl_file)

        kernel_str = open(cl_file).read()

        if not cl_dp:
            kernel_str = kernel_str.replace("double", "float")

        return kernel_str

    def _store_results(self, arg_repr, arg_score, run_type, t2run):
        """
        Stores the results of a run
        """

        # Check if the run type has been run, and if not create empty info
        run_type_benchs = self._benchmarks[run_type]
        if arg_repr not in run_type_benchs:
            run_type_benchs[arg_repr] = [arg_score]

        # Get the run info
        c = run_type_benchs[arg_repr]

        assert c[0] == arg_score, "arg_score mismatch"
        
        c.append(t2run)

        self._dump_run_times()

    def _dump_run_times(
        self,
    ):
        # TODO We might need to wrap this into a multiprocessing.Queue if we find it blocking 
        with open(self._benchmark_filepath, "w") as f:
            yaml.dump(self._benchmarks, f)

    def _get_args_repr_score(self, *args, **kwargs):
        """
        Get a string representation of the args and kwargs and corresponding 'score' / 'norm'
        The idea is that similar args have closer 'score'. Fuzzy logic

        The code does the following:
        1. It converts any args that are floats or ints to "number()" strings, and any args that are tensors to "shape()" strings
        2. It converts any kwargs that are floats or ints to "number()" strings, and any kwargs that are tensors to "shape()" strings
        3. The 'score' is given by the product of all the floats or ints and all the shape sizes.

        :return: the string representation of the args and kwargs
        :rtype: str
        """
        _norm = 1
        _args = []
        for arg in args:
            if type(arg) in (float, int):
                _args.append(f"number({arg})")
                if arg == 0:
                    arg = 1
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
                if v == 0:
                    v = 1
                _norm *= v
            if hasattr(v, "shape"):
                _kwargs[k] = f"shape{arg.shape}"
                _norm *= v.size
            else:
                _kwargs[k] = v

        return repr((_args, _kwargs)), _norm

    def get_highest_divisor(self, size_, max_):
        """
        Returns the highest divisor of size_ that is still lower than max_
        """
        value = 1
        for i in range(1, int(np.sqrt(size_) + 1)):
            if size_ % i == 0:
                if i * i != size_:
                    div2 = size_ / i

                    if i < max_:
                        value = max(value, i)
                    if div2 < max_:
                        value = max(value, div2)
        return int(value)

    def get_work_group(self, device, shape):
        """
        Calculates work group size for a given device and shape of global work space
        """

        max_wg_dims = device.max_work_item_sizes[0:3]
        max_glo_dims = device.max_work_group_size

        three = self.get_highest_divisor(shape[2], max_wg_dims[2])
        max_two = max_glo_dims / three
        two = self.get_highest_divisor(shape[1], max_two)
        one = 1
        return (one, two, three)
    
    def _check_max_slices(self, input, number_of_max_slices):
        """
        Checks if number of maximum slices is greater than 0
        """
        if number_of_max_slices < 1:
            raise ValueError("This device doesn't have enough memory to run this function with this input")
        elif input.shape[0] < number_of_max_slices:
            return input.shape[0]
        else:
            return number_of_max_slices

    def _check_max_buffer_size(self, size, device, n_slices):
        """
        Checks if buffer size is larger than device maximum memory allocation size and n_slices is 1 and raises appropriate errors that are handled in the _run function.
        """
        if size > device.max_mem_alloc_size and n_slices == 1:
            raise ValueError("This device cannot handle this input size with these parameters, try using a smaller input or other parameters")

        if size > device.max_mem_alloc_size:
            raise cl.Error("Buffer size is larger than device maximum memory allocation size")

        return size

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

    def _run_dask(*args, **kwargs):
        """
        Runs the dask version of the function
        Should be overridden by the any class that inherits from this class
        """
        pass

    def _run_transonic(*args, **kwargs):
        """
        Runs the transonic version of the function
        Should be overridden by the any class that inherits from this class
        """
        pass

    def _run_cuda(*args, **kwargs):
        """
        Runs the cuda version of the function
        Should be overridden by the any class that inherits from this class
        """
        pass
