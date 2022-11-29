import time
import types

def timeit(func: types.FunctionType):

    def wrapper(*args, **kwargs):
        t = time.time()
        retval = func(*args, **kwargs)
        print(f"{func.__name__} took {round(time.time()-t,3)} seconds")
        return retval

    return wrapper

