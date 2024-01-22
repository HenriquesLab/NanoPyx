import warnings

try:
    from transonic import jit

except ImportError:
    # raise a warning that transonic is not installed
    # and that the jit functions will not be used
    # and that the pure python functions will be used instead
    def jit(*args, **kwargs):
        def wrapper(func):
            warnings.warn(f"Transonic is not installed. Using pure python for {func.__name__}")
            return func

        return wrapper


def transonic_works():
    """
    Checks if the system has Transonic compatibility
    :return: True if the system has Transonic compatibility, False otherwise
    """
    try:
        from transonic import jit

        return True

    except ImportError:
        return False
