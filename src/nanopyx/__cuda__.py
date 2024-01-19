def cuda_works():
    """
    Checks if the system has cupyx compatibility
    :return: True if the system has cupyx compatibility, False otherwise
    """
    try:
        import cupyx

        return True

    except ImportError:
        return False
