import pyopencl as cl


def get_fastest_device_name():
    platforms = cl.get_platforms()

    max_compute_units = 0
    runtype_name = ""

    for platform in platforms:
        devices = platform.get_devices()

        for device in devices:
            if device.max_compute_units > max_compute_units:
                max_compute_units = device.max_compute_units
                runtype_name = "OpenCL_" + device.name

    return runtype_name