import time
import types


def timeit(func: types.FunctionType):
    """
    Decorator to measure the execution time of a function and print the result in seconds.

    Args:
        func (types.FunctionType): The function to be decorated.

    Returns:
        Callable: The decorated function.

    Example:
        @timeit
        def my_function():
            # Your code here

        When my_function is called, it will print the execution time in seconds.

    """

    def wrapper(*args, **kwargs):
        t = time.time()
        retval = func(*args, **kwargs)
        print(f"{func.__name__} took {round(time.time()-t,3)} seconds")
        return retval

    return wrapper


def timeit2(func):
    """
    Decorator to measure the execution time of a function and print the result in seconds, milliseconds, or nanoseconds.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.

    Example:
        @timeit2
        def my_function():
            # Your code here

        When my_function is called, it will print the execution time in an appropriate time unit (s, ms, or ns).

    """

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        if delta < 10**-4:
            msg = f"{func.__name__} took {delta/1e-6:.6f} nseconds"
        elif delta < 10**-1:
            msg = f"{func.__name__} took {delta/1e-3:.6f} mseconds"
        else:
            msg = f"{func.__name__} took {delta:.6f} seconds"
        print(msg)
        return result

    return wrapper
