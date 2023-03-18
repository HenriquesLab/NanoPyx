import os
import time

import yaml

from .. import __config_folder__
from . import works


class LiquidEngine:
    """
    Base class for parts of the NanoPyx Liquid Engine
    """

    RUN_TYPE_OPENCL = 0
    RUN_TYPE_THREAD = 1
    RUN_TYPE_UNTHREAD = 2
    RUN_TYPE_DESIGNATION = {
        RUN_TYPE_OPENCL: "OpenCL",
        RUN_TYPE_THREAD: "Threaded",
        RUN_TYPE_UNTHREAD: "Unthreaded",
    }

    _has_opencl = False
    _has_threaded = False
    _has_unthreaded = False
    _default_fastest = RUN_TYPE_OPENCL
    _last_run_type = None
    _last_run_time = None

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

        if self.RUN_TYPE_OPENCL not in self._cfg:
            self._cfg[self.RUN_TYPE_OPENCL] = {}
        if self.RUN_TYPE_THREAD not in self._cfg:
            self._cfg[self.RUN_TYPE_THREAD] = {}
        if self.RUN_TYPE_UNTHREAD not in self._cfg:
            self._cfg[self.RUN_TYPE_UNTHREAD] = {}

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
        call_args = self._get_args_repr(*args, **kwargs)
        if run_type not in self._cfg:
            self._cfg[run_type] = {}

        r = self._cfg[run_type]
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

    def benchmark(self, *args, **kwargs):
        """
        Benchmark the different run types
        :param args: args for the run method
        :param kwargs: kwargs for the run method
        :return: None
        """
        run_times = [None, None, None]
        returns = [None, None, None]

        if self._has_opencl:
            r = self._run(*args, run_type=self.RUN_TYPE_OPENCL, **kwargs)
            print(f"OpenCL run time: {self._last_run_time/1000} ms")
            run_times[self.RUN_TYPE_OPENCL] = self._last_run_time
            returns[self.RUN_TYPE_OPENCL] = r
        if self._has_threaded:
            r = self._run(*args, run_type=self.RUN_TYPE_THREAD, **kwargs)
            print(f"Threaded run time: {self._last_run_time/1000} ms")
            run_times[self.RUN_TYPE_THREAD] = self._last_run_time
            returns[self.RUN_TYPE_THREAD] = r
        if self._has_unthreaded:
            r = self._run(*args, run_type=self.RUN_TYPE_UNTHREAD, **kwargs)
            print(f"Unthreaded run time: {self._last_run_time/1000} ms")
            run_times[self.RUN_TYPE_UNTHREAD] = self._last_run_time
            returns[self.RUN_TYPE_UNTHREAD] = r

        if (
            run_times[self.RUN_TYPE_OPENCL] is not None
            and run_times[self.RUN_TYPE_THREAD] is not None
        ):
            print(
                f"OpenCL is {round(run_times[self.RUN_TYPE_THREAD] / run_times[self.RUN_TYPE_OPENCL], 2)}"
                + " times faster than threaded"
            )
        if (
            run_times[self.RUN_TYPE_OPENCL] is not None
            and run_times[self.RUN_TYPE_UNTHREAD] is not None
        ):
            print(
                f"OpenCL is {round(run_times[self.RUN_TYPE_UNTHREAD] / run_times[self.RUN_TYPE_OPENCL], 2)}"
                + " times faster than unthreaded"
            )
        if (
            run_times[self.RUN_TYPE_THREAD] is not None
            and run_times[self.RUN_TYPE_UNTHREAD] is not None
        ):
            print(
                f"Threaded is {round(run_times[self.RUN_TYPE_UNTHREAD] / run_times[self.RUN_TYPE_THREAD], 2)}"
                + " times faster than unthreaded"
            )

        call_args = self._get_args_repr(*args, **kwargs)
        fastest = run_times.index(min(run_times))
        print(
            f"{self.RUN_TYPE_DESIGNATION[fastest]} is the fastest for the call {call_args}"
        )

        return returns

    def _get_fastest_run_type(self, *args, **kwargs):
        """
        Retrieves the fastest run type for the given args and kwargs
        """
        call_args = self._get_args_repr(*args, **kwargs)
        run_times = [
            99999999999999999991,
            99999999999999999992,
            99999999999999999993,
        ]
        for n in range(3):
            if n in self._cfg and call_args in self._cfg[n]:
                runtime_sum = self._cfg[n][call_args][0]
                runtime_count = self._cfg[n][call_args][2]
                run_times[n] = runtime_sum / runtime_count

        fastest = run_times.index(min(run_times))
        return fastest

    def _get_cl_code(self, file_name):
        """
        Retrieves the OpenCL code from the corresponding .cl file
        """
        cl_file = os.path.splitext(os.path.abspath(file_name))[0] + ".cl"
        assert os.path.exists(cl_file), "Could not find OpenCL file: " + cl_file
        return open(cl_file, "r").read()

    def _get_args_repr(self, *args, **kwargs):
        return repr((args, kwargs))

    ###############
    # run methods #
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
            run_type = self._get_fastest_run_type(*args, **kwargs)
            print(
                f"No run type specified, using fastest run type: {self.RUN_TYPE_DESIGNATION[run_type]}"
            )

        if run_type == self.RUN_TYPE_OPENCL and self._has_opencl:
            self._last_run_type = self.RUN_TYPE_OPENCL
            t_start = time.time()
            r = self._run_opencl(*args, **kwargs)
            self._store_run_time(
                self.RUN_TYPE_OPENCL, time.time() - t_start, args, kwargs
            )
            return r
        elif run_type == self.RUN_TYPE_THREAD and self._has_threaded:
            self._last_run_type = self.RUN_TYPE_THREAD
            t_start = time.time()
            r = self._run_threaded(*args, **kwargs)
            self._store_run_time(
                self.RUN_TYPE_THREAD, time.time() - t_start, args, kwargs
            )
            return r
        elif self._has_unthreaded:
            self._last_run_type = self.RUN_TYPE_UNTHREAD
            t_start = time.time()
            r = self._run_unthreaded(*args, **kwargs)
            self._store_run_time(
                self.RUN_TYPE_UNTHREAD, time.time() - t_start, args, kwargs
            )
            return r
        else:
            raise NotImplementedError("No run method defined")

    def _run_opencl(*args, **kwargs):
        pass

    def _run_threaded(*args, **kwargs):
        pass

    def _run_unthreaded(*args, **kwargs):
        pass
