from ..__agent__ import Agent


class Workflow:
    """
    Workflow class that aggregates all the steps of the analysis and the corresponding functions.

    The Workflow class is designed to organize and execute a sequence of analysis steps in a workflow.
    It allows you to define a series of functions, their arguments, and their dependencies on the previous step's output.
    You can run the workflow sequentially and obtain the final result.

    Args:
        *args: Variable-length arguments. Each argument is expected to be a tuple of three items (fn, args, kwargs),
               where:
               - fn (callable): The function to be executed in this step.
               - args (tuple): The arguments to be passed to the function.
               - kwargs (dict): The keyword arguments to be passed to the function.

    Methods:
        __init__(*args): Initialize the Workflow object with a list of analysis steps.
            - Each arg must be a tuple of three items (fn, args, kwargs).

        run(_force_run_type=None): Run the workflow sequentially.
            - _force_run_type (str, optional): Force a specific run type for all steps in the workflow.

        calculate(_force_run_type=None): Calculate the final result of the workflow.
            - _force_run_type (str, optional): Force a specific run type for all steps in the workflow.

    Example:
        # Define a workflow with three steps
        workflow = Workflow(
            (step1_function, (arg1,), {"kwarg1": value1}),
            (step2_function, ("PREV_RETURN_VALUE_0", arg2), {"kwarg2": value2}),
            (step3_function, ("PREV_RETURN_VALUE_1",), {})
        )

        # Run the workflow and get the final result
        result = workflow.calculate()

    Note:
        - The Workflow class allows you to specify dependencies between steps using "PREV_RETURN_VALUE" placeholders.
        - The result of each step is stored and can be accessed later.
    """

    def __init__(self, *args) -> None:
        """
        Initialize the Workflow object.

        Args:
            *args: Variable-length arguments. Each argument is expected to be a tuple of three items (fn, args, kwargs).

        Returns:
            None
        """

        self._methods = []
        self._return_values = []

        for arg in args:
            if isinstance(arg, tuple) and len(arg) == 3:
                self._methods.append((arg[0], arg[1], arg[2]))
            else:
                raise TypeError("Each arg must be a tuple of 3 items (fn, args, kwargs)")

    def run(self, _force_run_type=None):
        """
        Run the workflow sequentially.

        Args:
            _force_run_type (str, optional): Force a specific run type for all steps in the workflow.

        Returns:
            Tuple: A tuple containing the final result, the run type of the last step, and the execution time.

        Example:
            output, run_type, execution_time = workflow.run()

        Note:
            - The result of each step is stored in self._return_values and can be accessed later.
            - The run type of each step is determined using Agent.get_run_type() and can be overridden with _force_run_type.
        """

        for method in self._methods:
            fn, args, kwargs = method

            # in the list args, substitute 'PREV_RETURN_VALUE' with the return value of the previous method
            sane_args = []
            for arg in args:
                if isinstance(arg, str) and "PREV_RETURN_VALUE" in arg:
                    indices = [int(i) for i in arg.split("_") if i.isdigit()]
                    return_value = self._return_values[indices[0]][indices[1]]
                    sane_args.append(return_value)
                else:
                    sane_args.append(arg)

            # in the dict kwargs, substitute 'PREV_RETURN_VALUE' with the return value of the previous method
            for key, value in kwargs.items():
                if isinstance(value, str) and "PREV_RETURN_VALUE" in value:
                    indices = [int(i) for i in value.split("_") if i.isdigit()]
                    return_value = self._return_values[indices[0]][indices[1]]
                    kwargs[key] = return_value

            # Get run type from the Agent
            run_type = Agent.get_run_type(fn, sane_args, kwargs)
            kwargs["run_type"] = run_type

            if _force_run_type:
                kwargs["run_type"] = _force_run_type

            # TODO maybe we need to warn the agent its running and when it finishes
            return_value = fn.run(*sane_args, **kwargs)

            if isinstance(return_value, tuple):
                self._return_values.append(return_value)
            else:
                self._return_values.append((return_value,))

            Agent._inform(fn)

        return self._return_values[-1], fn._last_runtype, fn._last_time

    def calculate(self, _force_run_type=None):
        """
        Calculate the final result of the workflow.

        Args:
            _force_run_type (str, optional): Force a specific run type for all steps in the workflow.

        Returns:
            Any: The final result of the workflow.

        Example:
            result = workflow.calculate()

        Note:
            This method is a convenient way to obtain the final result without detailed information about each step.
        """
        output, tmp, tmp = self.run(_force_run_type=_force_run_type)
        return output
