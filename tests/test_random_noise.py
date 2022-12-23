from nanopyx.core.transform.random_noise import addMixedGaussianPoissonNoise
from nanopyx.core.transform import random_noise


def test_logFactorial():
    random_noise.test_logFactorial()


def test_random_noise(random_image_with_ramp_squares):
    random_image = random_image_with_ramp_squares

    addMixedGaussianPoissonNoise(random_image, gaussSigma=100, gaussMean=100)
    assert random_image.mean() > 100

