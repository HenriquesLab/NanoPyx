import pyopencl as cl
from nanopyx.core.utils.cl_device import get_fastest_device_name


def test_get_fastest_device_name():
    try:
        get_fastest_device_name()
    except cl.Error:
        assert True
