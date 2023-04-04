import os
import warnings

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

try:
    import pyopencl as cl
    import pyopencl.array as cl_array

    fastest_device = None
    fastest_device_speed = 0
    cl_ctx = None
    cl_queue = None

    for platform in cl.get_platforms():
        for device in platform.get_devices():
            # check if the device is a GPU
            if "GPU" not in cl.device_type.to_string(device.type):
                continue

            speed = (
                device.max_clock_frequency
                * device.max_compute_units
                * device.global_mem_size
            )
            # If no fastest device has been found, set it to the current device
            if fastest_device is None:
                fastest_device = device
                fastest_device_speed = speed

            # Check if the current device is faster than the fastest device
            elif speed > fastest_device_speed:
                fastest_device = device
                fastest_device_speed = speed

    print("Selecting OpenCL device: " + fastest_device.name)

    if 'cl_khr_fp64' in fastest_device.extensions.strip().split(' '):
        cl_dp = True
    else:
        cl_dp = False

    cl_ctx = cl.Context([fastest_device])
    cl_queue = cl.CommandQueue(cl_ctx)

except (ImportError, OSError, Exception):
    cl = None
    cl_array = None
    cl_ctx = None
    cl_queue = None
    cl_dp = False


def print_opencl_info():
    """
    Prints information about the OpenCL devices on the system
    """
    # REF: https://github.com/benshope/PyOpenCL-Tutorial

    print("\n" + "=" * 60 + "\nOpenCL Platforms and Devices")
    # Print each platform on this computer
    for platform in cl.get_platforms():
        print("=" * 60)
        print("Platform - Name:  " + platform.name)
        print("Platform - Vendor:  " + platform.vendor)
        print("Platform - Version:  " + platform.version)
        print("Platform - Profile:  " + platform.profile)
        # Print each device per-platform
        for device in platform.get_devices():
            print("\t" + "-" * 56)
            print("\tDevice - Name: " + device.name)
            print("\tDevice - Type: " + cl.device_type.to_string(device.type))
            print(
                f"\tDevice - Max Clock Speed:  {device.max_clock_frequency} Mhz"
            )

            print(f"\tDevice - Compute Units:  {device.max_compute_units}")
            print(
                f"\tDevice - Local Memory:  {device.local_mem_size / 1024.0:.0f} KB"
            )
            print(
                f"\tDevice - Constant Memory:  {device.max_constant_buffer_size / 1024.0:.0f} KB"
            )
            print(
                f"\tDevice - Global Memory: {device.global_mem_size / 1073741824.0:.0f} GB"
            )
            print(
                f"\tDevice - Max Buffer/Image Size: {device.max_mem_alloc_size / 1048576.0:.0f} MB"
            )
            print(
                f"\tDevice - Max Work Group Size: {device.max_work_group_size:.0f}"
            )
    print("\n")


def opencl_works():
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
