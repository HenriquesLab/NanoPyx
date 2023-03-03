import numpy as np
from nanopyx.core.utils import open_simplex_2F


def test_add_simplex_noise(plt):
    im = np.zeros((100, 100), dtype=np.float32)
    open_simplex_2F.add_simplex_noise(im, 1234)
    plt.imshow(im)
