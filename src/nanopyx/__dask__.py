def dask_works():
    """
    Checks if the system has dask_image compatibility
    :return: True if the system has dask_image compatibility, False otherwise
    """
    try:
        from dask_image import ndfilters

        return True

    except ImportError:
        return False
