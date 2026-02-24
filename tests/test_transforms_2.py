import numpy as np

from nanopyx.core.generate.noise_add_simplex import get_simplex_noise
from nanopyx.core.transform import NNPolarTransform
from nanopyx.core.analysis.pearson_correlation import pearson_correlation

from skimage.transform import warp_polar


def _compare(output_1, output_2):
    if output_1.ndim > 2:
        pcc = 0
        for i in range(output_1.shape[0]):
            pcc += pearson_correlation(output_1[i, :, :], output_2[i, :, :])
        pcc /= output_1.shape[0]
    else:
        pcc = pearson_correlation(output_1, output_2)
    return pcc > 0.8


def test_interpolation_nearest_neighbor_PolarTransform_linear():
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 64, frames=nFrames, amplitude=1000)
    SM = NNPolarTransform(testing=True, clear_benchmarks=True)
    bench_values = SM.benchmark(image, (100, 100), "linear")

    skimage_linear = warp_polar(
        image, output_shape=(100, 100), channel_axis=0, order=0
    )

    images = [skimage_linear]
    titles = ["Skimage"]
    run_times = ["nan"]

    # unzip the values
    for run_time, title, image in bench_values:
        run_times.append(run_time)
        titles.append(title)
        images.append(image)

    # ensure images are similar
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            assert _compare(images[i], images[j])


def test_interpolation_nearest_neighbor_PolarTransform_log():
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 64, frames=nFrames, amplitude=1000)
    SM = NNPolarTransform(testing=True, clear_benchmarks=True)
    bench_values = SM.benchmark(image, (100, 100), "log")

    skimage_linear = warp_polar(
        image, output_shape=(100, 100), channel_axis=0, scaling="log", order=0
    )

    images = [skimage_linear]
    titles = ["Skimage"]
    run_times = ["nan"]

    # unzip the values
    for run_time, title, image in bench_values:
        run_times.append(run_time)
        titles.append(title)
        images.append(image)

    # ensure images are similar
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            assert _compare(images[i], images[j])
