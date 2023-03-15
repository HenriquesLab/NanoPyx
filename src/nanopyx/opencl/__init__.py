import os

import pyopencl as cl

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"


def find_kernel_path(cl_file_name: str):
    """
    Find the path to a kernel file
    :param cl_file_name: name of the kernel file
    :return: path to the kernel file
    """
    if not cl_file_name.endswith(".cl"):
        if cl_file_name.endswith(".py"):
            cl_file_name = cl_file_name[:-3] + ".cl"
        elif cl_file_name.endswith(".pyx"):
            cl_file_name = cl_file_name[:-4] + ".cl"

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
            print(f"\tDevice - Max Clock Speed:  {device.max_clock_frequency} Mhz")

            print(f"\tDevice - Compute Units:  {device.max_compute_units}")
            print(f"\tDevice - Local Memory:  {device.local_mem_size / 1024.0:.0f} KB")
            print(
                f"\tDevice - Constant Memory:  {device.max_constant_buffer_size / 1024.0:.0f} KB"
            )
            print(
                f"\tDevice - Global Memory: {device.global_mem_size / 1073741824.0:.0f} GB"
            )
            print(
                f"\tDevice - Max Buffer/Image Size: {device.max_mem_alloc_size / 1048576.0:.0f} MB"
            )
            print(f"\tDevice - Max Work Group Size: {device.max_work_group_size:.0f}")
    print("\n")


def get_fastest_device(benchmark="clock_frequency", type="gpu"):
    """
    Returns the fastest OpenCL device on the system
    :param benchmark: The benchmark to use to determine the fastest device (clock_frequency or compute_units)
    :return: The fastest OpenCL device
    """

    fastest_device = None
    fastest_device_speed = 0

    for platform in cl.get_platforms():
        for device in platform.get_devices():
            # Ignore devices that are not of the specified type
            if type is not None:
                if type != cl.device_type.to_string(device.type):
                    continue

            speed = device.max_clock_frequency * device.max_compute_units
            # If no fastest device has been found, set it to the current device
            if fastest_device is None:
                fastest_device = device
                fastest_device_speed = speed

            # Check if the current device is faster than the fastest device
            elif speed > fastest_device_speed:
                fastest_device = device
                fastest_device_speed = speed

    return device


def works():
    """
    Checks if the system has OpenCL compatibility
    :return: True if the system has OpenCL compatibility, False otherwise
    """
    disabled = os.environ.get("NANOPYX_DISABLE_OPENCL", "0") == "1"
    enabled = os.environ.get("NANOPYX_ENABLE_OPENCL", "1") == "1"

    if disabled or not enabled:
        return False
    elif not disabled or enabled:
        return True
    try:
        get_fastest_device()
        os.environ["NANOPYX_ENABLE_OPENCL"] = "1"
        return True
    except Exception:
        os.environ["NANOPYX_DISABLE_OPENCL"] = "1"
        return False
