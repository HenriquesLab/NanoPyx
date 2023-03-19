# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True



from ...liquid import MandelbrotBenchmark


def check_acceleration(size: int = 1000):
    """
    Check the acceleration of the opencl vs cython version
    :param size: size of the image
    :return: tuple of images
    """
    bench = MandelbrotBenchmark()
    images = bench.benchmark(size=size)
    return images
