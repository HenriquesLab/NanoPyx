import numpy as np

from nanopyx.core.transform import image_add_random_noise
from nanopyx.core.transform.image_add_random_noise import (
    addMixedGaussianPoissonNoise, getPerlinNoise, getRamp, getSimplexNoise)


def test_logFactorial():
    image_add_random_noise.test_logFactorial()


def test_random_noise(random_image_with_ramp_squares):
    random_image = random_image_with_ramp_squares
    addMixedGaussianPoissonNoise(random_image, gaussSigma=100, gaussMean=100)
    assert random_image.mean() > 100


def test_getRamp(plt):
    ramp = getRamp(100, 400)
    assert ramp.shape == (100, 400)
    plt.imshow(ramp)


def test_getPerlinNoise(plt):
    noise = getPerlinNoise(100, 400, f=2)
    assert np.all(noise >= 0)
    plt.imshow(noise)


def test_getSimplexNoise(plt):
    noise = getSimplexNoise(100, 400, f=10)
    # assert np.all(noise >= 0)
    plt.imshow(noise)


def test_getSquares(plt):
    squares = image_add_random_noise.getSquares(100, 400, nSquares=10)
    assert np.all(squares >= 0)
    plt.imshow(squares)


def test_logFactorial():
    image_add_random_noise.test_logFactorial()
