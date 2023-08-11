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
                self._methods.append((arg[0], arg[1], arg[2]))
            else:
                raise TypeError("Each arg must be a tuple of 3 items (fn, args, kwargs)")

    def run(self,_force_run_type=None):
        """
        Run the workflow sequentially
        """

        for method in self._methods:
            fn, args, kwargs = method

            # in the list args, substitute 'PREV_RETURN_VALUE' with the return value of the previous method
            sane_args = []
            for arg in args:
                if isinstance(arg, str) and 'PREV_RETURN_VALUE' in arg:
                    indices = [int(i) for i in arg.split('_') if i.isdigit()]
                    return_value = self._return_values[indices[0]][indices[1]]
                    sane_args.append(return_value)
                else:
                    sane_args.append(arg)

            # in the dict kwargs, substitute 'PREV_RETURN_VALUE' with the return value of the previous method
            for key, value in kwargs.items():
                if isinstance(value, str) and 'PREV_RETURN_VALUE' in value:
                    indices = [int(i) for i in value.split('_') if i.isdigit()]
                    return_value = self._return_values[indices[0]][indices[1]]
                    kwargs[key] = return_value


            # Get run type from the Agent
            run_type = Agent.get_run_type(fn, sane_args, kwargs)
            kwargs['run_type'] = run_type

            if _force_run_type:
                kwargs['run_type'] = _force_run_type

            # TODO maybe we need to warn the agent its running and when it finishes
            return_value = fn.run(*sane_args, **kwargs)

            if isinstance(return_value, tuple):
                self._return_values.append(return_value)
            else:
                self._return_values.append((return_value,))

            Agent._inform(fn)

        return self._return_values[-1], fn._last_runtype, fn._last_time
    
    def calculate(self, _force_run_type=None):
        output, tmp, tmp = self.run(_force_run_type=_force_run_type)
        return output
