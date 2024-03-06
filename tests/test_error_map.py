import numpy as np
from scipy.ndimage import gaussian_filter

from nanopyx.core.transform.error_map import ErrorMap
from nanopyx.core.transform.binning import rebin_2d
from nanopyx.core.generate.noise_add_mixed_noise import (
    add_mixed_gaussian_poisson_noise,
)
from nanopyx.core.generate.noise_add_simplex import add_simplex_noise


def test_error_map():
    # generate some random ground truth
    rows = 1000
    cols = 1000
    alpha = 5
    image_gt = np.ones((rows, cols), dtype="float32") * 1000
    add_simplex_noise(image_gt, amplitude=1000, frequency=0.01)

    image_ref = gaussian_filter(image_gt * alpha, 15)
    image_ref = rebin_2d(image_ref, 10, mode="mean")
    add_mixed_gaussian_poisson_noise(image_ref, 10, 10)

    image_sr = image_gt.copy()
    image_sr = gaussian_filter(image_sr, 3)
    # image_sr = rebin2d(image_sr, 10, mode="mean")

    squirrel_error_map = ErrorMap()
    squirrel_error_map.optimise(image_ref, image_sr)

    # assert squirrel_error_map.getRSP() > 0.95
    # assert abs(squirrel_error_map._alpha - alpha) < 1e3
    # assert squirrel_error_map._sigma - 15 < 1e3
