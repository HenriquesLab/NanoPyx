import numpy as np
from nanopyx.core.utils import open_simplex_2F


def test_add_simplex_noise(plt):
    im = np.zeros((100, 100), dtype=np.float32)
    open_simplex_2F.add_simplex_noise(
        im, frequency=0.1, octaves=3, persistence=0.1, seed=0
    )
    plt.imshow(im)
