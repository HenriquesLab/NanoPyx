import os
import warnings

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

try:
    import pyopencl as cl
    import pyopencl.array as cl_array

    fastest_device = None
    fastest_device_speed = 0
    ctx = None
    queue = None

    for platform in cl.get_platforms():
        for device in platform.get_devices():
            speed = device.max_clock_frequency * device.max_compute_units
            # If no fastest device has been found, set it to the current device
            if fastest_device is None:
                fastest_device = device
                fastest_device_speed = speed

            # Check if the current device is faster than the fastest device
            elif speed > fastest_device_speed:
                fastest_device = device
                fastest_device_speed = speed
    print(cl.device_type.to_string(fastest_device.type))

    ctx = cl.Context([fastest_device])
    queue = cl.CommandQueue(ctx)

except (ImportError, OSError, AttributeError):
    cl = None
    ctx = None
    queue = None


def works():
    """
    Checks if the system has OpenCL compatibility
    :return: True if the system has OpenCL compatibility, False otherwise
    """
    disabled = os.environ.get("NANOPYX_DISABLE_OPENCL", "0") == "1"
    enabled = os.environ.get("NANOPYX_ENABLE_OPENCL", "1") == "1"

    if disabled or not enabled:
        warnings.warn(
            "OpenCL is disabled. To enable it, set the environment variable NANOPYX_ENABLE_OPENCL=1"
        )
        return False

    elif enabled:
        if cl is None:
            warnings.warn("tap... tap... tap... COMPUTER SAYS NO (OpenCL)!")
            os.environ["NANOPYX_DISABLE_OPENCL"] = "1"
            return False

    return True


def find_kernel_path(cl_file_name: str):
    """
    Find the path to a kernel file
    :param cl_file_name: name of the kernel file
    :return: path to the kernel file
    """
    if not cl_file_name.endswith(".cl"):
        cl_file_name = os.path.splitext(cl_file_name)[0] + ".cl"

    # Find if its an absolute path
    if os.path.exists(cl_file_name):
        file_path = cl_file_name
        return file_path

    # Find if its in the current directory
    path = os.path.split(__file__)[0]
    file_path = os.path.join(path, cl_file_name)
    if os.path.exists(file_path):
        return file_path

    # Find if its in the current directory tree
    for root, dirs, files in os.walk(path):
        # Check if the current directory contains the target file
        if cl_file_name in files:
            # If found, print the full path to the file
            cl_file_name = os.path.join(root, cl_file_name)
            return cl_file_name

    raise RuntimeError(f"Could not find kernel {cl_file_name}")


def get_kernel_txt(cl_file_name: str):
    """
    Finds the path to an opencl kernel file
    :param file_name: Name of the kernel file
    :return: Path to the kernel file
    """

    cl_file_path = find_kernel_path(cl_file_name)

    # Read the kernel file
    with open(cl_file_path, "r") as f:
        kernel_txt = f.read()

    return kernel_txt
