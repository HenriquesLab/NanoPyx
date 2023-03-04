from nanopyx.core.generate.noise_add_gaussians import render_random_gaussians
from nanopyx.core.generate.noise_add_mixed_noise import (
    add_mixed_gaussian_poisson_noise,
)


def test_random_gaussians(plt):
    n_frames = 25
    gaussians = render_random_gaussians(150, 100, n_frames, particles_per_slice=1000)

    plt.figure()
    f, axarr = plt.subplots(5, 5)
    for nf in range(n_frames):
        axarr[nf // 5, nf % 5].imshow(gaussians[nf], interpolation="none")
        # hide the axes
        axarr[nf // 5, nf % 5].axis("off")


def test_random_gaussians_with_noise(plt):
    n_frames = 25
    gaussians = render_random_gaussians(
        150, 100, n_frames, particles_per_slice=1000, amplitude=1000
    )
    add_mixed_gaussian_poisson_noise(gaussians, gauss_sigma=10, gauss_mean=100)

    plt.figure()
    f, axarr = plt.subplots(5, 5)
    for nf in range(n_frames):
        axarr[nf // 5, nf % 5].imshow(gaussians[nf], interpolation="none")
        # hide the axes
        axarr[nf // 5, nf % 5].axis("off")
