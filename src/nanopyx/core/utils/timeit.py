import time
import types


def timeit(func: types.FunctionType):

    def wrapper(*args, **kwargs):
        t = time.time()
        retval = func(*args, **kwargs)
        print(f"{func.__name__} took {round(time.time()-t,3)} seconds")
        return retval

    return wrapper


def timeit2(func):
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
