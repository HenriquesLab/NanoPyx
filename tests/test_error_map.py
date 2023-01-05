import numpy as np
from scipy.ndimage import gaussian_filter

from nanopyx.core.sr.error_map import ErrorMap
from nanopyx.core.transform.binning import rebin2d
from nanopyx.core.transform.image_add_random_noise import (
    addMixedGaussianPoissonNoise,
    addPerlinNoise,
)


def test_error_map():
    # generate some random ground truth
    w = 1000
    h = 1000
    alpha = 5
    image_gt = np.ones((w, h), dtype="float32") * 1000
    addPerlinNoise(image_gt, amp=1000, f=10, octaves=3)

    image_ref = gaussian_filter(image_gt * alpha, 15)
    image_ref = rebin2d(image_ref, 10, mode="mean")
    addMixedGaussianPoissonNoise(image_ref, 10, 10)

    image_sr = image_gt.copy()
    image_sr = gaussian_filter(image_sr, 3)
    # image_sr = rebin2d(image_sr, 10, mode="mean")

    squirrel_error_map = ErrorMap()
    squirrel_error_map.optimise(image_ref, image_sr)

    assert squirrel_error_map.getRSP() > 0.95
    assert abs(squirrel_error_map._alpha - alpha) < 1e3
    assert squirrel_error_map._sigma - 15 < 1e3
