import numpy as np

from nanopyx.core.generate.noise_add_random_noise import add_mixed_gaussian_poisson_noise, get_perlin_noise, get_simplex_noise
from nanopyx.core.generate.noise_add_squares import get_squares
from nanopyx.core.generate.noise_add_ramp import get_ramp


def test_random_noise(random_image_with_ramp_squares):
    random_image = random_image_with_ramp_squares
    add_mixed_gaussian_poisson_noise(random_image, gauss_sigma=100, gauss_mean=100)
    assert random_image.mean() > 100


def test_get_ramp(plt):
    ramp = get_ramp(100, 400)
    assert ramp.shape == (400, 100)
    plt.imshow(ramp)


def test_get_perlin_noise(plt):
    noise = get_perlin_noise(100, 400, f=2)
    assert np.all(noise >= 0)
    plt.imshow(noise)


def test_get_simplex_noise(plt):
    noise = get_simplex_noise(100, 400, f=10)
    # assert np.all(noise >= 0)
    plt.imshow(noise)


def test_get_squares(plt):
    squares = get_squares(100, 400, nSquares=10)
    assert np.all(squares >= 0)
    plt.imshow(squares)
