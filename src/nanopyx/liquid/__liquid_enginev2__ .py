
# IDEALIZING A NEW LIQUID ENGINE CLASS

class LiquidEngine:

    """
    Base class for parts of the Nanopyx Liquid engine
    """

    # CLASS VARIABLES (aka present in every instance of the class)
    # (...)


    def __init__(self, clear_config:bool=False, testing:bool=False,
                 opencl:bool = False, unthreaded:bool = False,
                 threaded:bool = False, threaded_static:bool = False,
                 threaded_dynamic:bool = False, threaded_guided:bool = False,
                 python_:bool=False, njit_:bool=False) -> None:
        """
        Initialize the Liquid Engine

        The code does the following:
        1. Checks whether OpenCL is available (by running a simple OpenCL kernel)
        2. Checks whether Numba is available (by running the njit decorator)
        3. Creates a path to store the config file (e.g. ~/.nanopyx/liquid/_le_interpolation_nearest_neighbor.cpython-310-darwin/ShiftAndMagnify.yml)
        4. Loads the config file (if it exists)
        5. Creates empty dictionaries for each run type (e.g. 'Threaded', 'OpenCL', 'Numba')

        :param clear_config: whether to clear the config file
        :param testing: whether to run all the methods in testing mode
        """
        
        pass


    ################
    # _run methods #
    ################

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
        pass

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
            