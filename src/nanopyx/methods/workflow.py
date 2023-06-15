
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
            if isinstance(arg, tuple) and len(item) == 3:
                self._methods.append(arg)
            else:
                raise TypeError("Each arg must be a tuple of 3 items (fn, args, kwargs)")

    def run(self,):
        """
        Run the workflow sequentially
        """

        
        for method in self._methods:
            fn, args, kwargs = method
            # in the list args, substitute 'PREV_RETURN_VALUE' with the return value of the previous method
            args = [arg if arg != 'PREV_RETURN_VALUE' else self._return_values[-1] for arg in args]
            # in the dict kwargs, substitute 'PREV_RETURN_VALUE' with the return value of the previous method
            kwargs = {key: value if value != 'PREV_RETURN_VALUE' else self._return_values[-1] for key, value in kwargs.items()}
                
            return_value = fn().run(*args, **kwargs)
            self._return_values.append(return_value)

        return self._return_values[-1]
    
        