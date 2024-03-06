import numpy as np

from nanopyx.core.generate.noise_add_gaussians import render_random_gaussians
from nanopyx.core.generate.noise_add_mixed_noise import (
    add_mixed_gaussian_poisson_noise,
)
from nanopyx.core.generate.noise_add_ramp import get_ramp
from nanopyx.core.generate.noise_add_simplex import add_simplex_noise
from nanopyx.core.generate.noise_add_squares import get_squares


def test_mixed_noise(random_image_with_ramp_squares):
    random_image = random_image_with_ramp_squares
    add_mixed_gaussian_poisson_noise(random_image, gauss_sigma=100, gauss_mean=100)
    assert random_image.mean() > 100


def test_get_ramp(): #(plt):
    ramp = get_ramp(100, 400)
    assert ramp.shape == (400, 100)
    #plt.imshow(ramp)


def test_get_squares(): #(plt):
    squares = get_squares(100, 400, nSquares=10)
    assert np.all(squares >= 0)
    #plt.imshow(squares)


def test_add_simplex_noise(): #(plt):
    im = np.zeros((100, 100), dtype=np.float32)
    add_simplex_noise(im)
    #plt.imshow(im)


def test_random_gaussians(): #(plt):
    n_frames = 25
    gaussians = render_random_gaussians(150, 100, n_frames, particles_per_slice=1000)

    # plt.figure()
    # f, axarr = plt.subplots(5, 5)
    # for nf in range(n_frames):
    #     axarr[nf // 5, nf % 5].imshow(gaussians[nf], interpolation="none")
    #     # hide the axes
    #     axarr[nf // 5, nf % 5].axis("off")


def test_random_gaussians_with_noise(): #(plt):
    n_frames = 25
    gaussians = render_random_gaussians(
        150, 100, n_frames, particles_per_slice=1000, amplitude=1000
    )
    add_mixed_gaussian_poisson_noise(gaussians, gauss_sigma=10, gauss_mean=100)

    # plt.figure()
    # f, axarr = plt.subplots(5, 5)
    # for nf in range(n_frames):
    #     axarr[nf // 5, nf % 5].imshow(gaussians[nf], interpolation="none")
    #     # hide the axes
    #     axarr[nf // 5, nf % 5].axis("off")
