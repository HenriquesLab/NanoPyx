import warnings

try:
    from numba import njit, prange

except ImportError:
    # raise a warning that numba is not installed
    # and that the njit functions will not be used
    # and that the pure python functions will be used instead

    prange = range

    def njit(*args, **kwargs):
        def wrapper(func):
            warnings.warn(f"Numba is not installed. Using pure python for {func.__name__}")
            return func

        return wrapper


def njit_works():
    """
    Checks if the system has Numba compatibility
    :return: True if the system has Numba compatibility, False otherwise
    """
    try:
        from numba import njit, prange

        return True

    except ImportError:
        return False
