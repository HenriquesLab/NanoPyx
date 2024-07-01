import numpy as np

from nanopyx.core.transform._le_convolution import Convolution
from nanopyx.core.transform.convolution import convolution2D_numba


def test_conv_small():
    small = np.random.random((100, 100)).astype(np.float32)
    kernel = np.ones((23, 23)).astype(np.float32)
    conv = Convolution()
    conv.benchmark(small, kernel)


def test_convolution2D_numba():
    img = np.random.random((100, 100)).astype(np.float32)
    kernel = np.random.random((3, 3)).astype(np.float32)

    convolution2D_numba(img, kernel)
