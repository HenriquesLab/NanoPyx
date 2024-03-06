import numpy as np

from nanopyx.core.transform._le_convolution import Convolution


def test_conv_small():
    small = np.random.random((100, 100)).astype(np.float32)
    kernel = np.ones((23, 23)).astype(np.float32)
    conv = Convolution()
    conv.benchmark(small, kernel)
