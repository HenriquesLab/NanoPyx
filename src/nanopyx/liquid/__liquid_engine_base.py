import os
import timeit
from functools import partial
from pathlib import Path

import numpy as np

# This will in the future come from the Agent
from .__njit__ import njit_works
from .__opencl__ import opencl_works, devices

class LiquidEngine_:

    """
    Base class for parts of the Nanopyx Liquid Engine
    Vroom Vroom 
    """

    def __init__(self, testing:bool=False,
                 opencl_:bool = False, unthreaded_:bool = False,
                 threaded_:bool = False, threaded_static_:bool = False,
                 threaded_dynamic_:bool = False, threaded_guided_:bool = False,
                 python_:bool=False, njit_:bool=False) -> None:
        """
        Initialize the Liquid Engine
        The Liquid Engine base class is inherited by children classes that implement specific methods

        Engine responsabilities:
        1. Store implemented run types;
        2. Benchmark all available run types;
        3. Run the method using a selected run type;

        Communication betwen a specific LE and the agent is done through Run dataclasses
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

        self.testing = testing

    def _run(self, *args, run_type:str, **kwargs):
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

        if run_type not in self._run_types:
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

        return result, t2run
    

    def benchmark(self,*args, **kwargs):
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

            r, t = self._run(*args, run_type=run_type, **kwargs)

            if r is None:
                run_times[run_type] = np.inf
            else:
                run_times[run_type] = t
            
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
            