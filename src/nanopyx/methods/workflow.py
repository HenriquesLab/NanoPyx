from ..__agent__ import Agent

class Workflow:
    
    """ 
    Workflow class that aggregates all the steps of the analysis and the corresponding functions.
    """

    def __init__(self,*args) -> None:
        """
        Each arg is a tuple of 3 items (fn, args, kwargs)
        """

        self._methods = []
        self._return_values = []

        for arg in args:
            if isinstance(arg, tuple) and len(arg) == 3:
                self._methods.append((arg[0](), arg[1], arg[2])) # Note the parenthesis! We want to instantiate the class here!
            else:
                raise TypeError("Each arg must be a tuple of 3 items (fn, args, kwargs)")

    def run(self,):
        """
        Run the workflow sequentially
        """

        for method in self._methods:
            fn, args, kwargs = method

            # in the list args, substitute 'PREV_RETURN_VALUE' with the return value of the previous method
            sane_args = []
            for arg in args:
                if isinstance(arg, str) and arg == 'PREV_RETURN_VALUE':
                    sane_args.append(self._return_values[-1])
                else:
                    sane_args.append(arg)

            # in the dict kwargs, substitute 'PREV_RETURN_VALUE' with the return value of the previous method
            for key, value in kwargs.items():
                if isinstance(value, str) and value == 'PREV_RETURN_VALUE':
                    kwargs[key] = self._return_values[-1]

            # Get run type from the Agent
            run_type = Agent.get_run_type(fn, sane_args, kwargs)
            kwargs['run_type'] = run_type

            # TODO maybe we need to warn the agent its running and when it finishes
            return_value = fn.run(*sane_args, **kwargs)
            self._return_values.append(return_value)

        return self._return_values[-1]
    

    
        