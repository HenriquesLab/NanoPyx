from . import MandelbrotBenchmark


def check_acceleration(size: int = 128):
    """
    Check the acceleration of the opencl vs cython version
    :param size: size of the image
    :return: tuple of images
    """
    bench = MandelbrotBenchmark()
    images = bench.benchmark(size=size)
    return images
