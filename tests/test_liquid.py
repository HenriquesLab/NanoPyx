import numpy as np

from nanopyx.core.generate.noise_add_simplex import get_simplex_noise
from nanopyx.liquid._le_interpolation_catmull_rom import \
    ShiftAndMagnify as CRShiftAndMagnify
from nanopyx.liquid._le_interpolation_nearest_neighbor import \
    ShiftAndMagnify as NNShiftAndMagnify
from nanopyx.liquid._le_mandelbrot_benchmark import MandelbrotBenchmark


def test_mandelbrot_benchmark(plt):
    mb = MandelbrotBenchmark()
    values = mb.benchmark(128)

    images = []
    titles = []
    run_times = []
    # unzip the values
    for run_time, title, image in values:
        images.append(image)
        titles.append(title)
        run_times.append(run_time)

    # # check that all the images have the same value
    # for i in range(len(images)):
    #     for j in range(i + 1, len(images) - 1):
    #         if images[i] is not None and images[j] is not None:
    #             np.testing.assert_almost_equal(images[i], images[j], decimal=0)

    # check that the run times are in the correct order
    for i in range(len(run_times)):
        for j in range(i + 1, len(run_times)):
            assert run_times[i] <= run_times[j]

    # plot the images in subplots
    fig, axs = plt.subplots(1, len(images), figsize=(20, 20))
    for i in range(len(images)):
        axs[i].imshow(images[i], cmap="hot")
        axs[i].set_title(titles[i])
        axs[i].axis("off")


def test_interpolation_nearest_neighbor_ShiftAndMagnify(plt):
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 32, frames=nFrames, amplitude=1000)
    shift_row = np.arange(nFrames, dtype=np.float32) * 0.5
    shift_col = np.arange(nFrames, dtype=np.float32) * -0.5
    SM = NNShiftAndMagnify()
    bench_values = SM.benchmark(image, shift_row, shift_col, M, M)

    images = []
    titles = []
    run_times = []

    # unzip the values
    for run_time, title, image in bench_values:
        run_times.append(run_time)
        titles.append(title)
        images.append(image)

    # show images
    fig, axes = plt.subplots(nFrames, len(images), figsize=(20, 10))
    for i in range(nFrames):
        for j in range(len(images)):
            if i == 0:
                axes[i, j].set_title(titles[j])
            axes[i, j].imshow(images[j][i], cmap="hot")
            axes[i, j].axis("off")


def test_interpolation_catmull_rom_ShiftAndMagnify(plt):
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 32, frames=nFrames, amplitude=1000)
    shift_row = np.arange(nFrames, dtype=np.float32) * 0.5
    shift_col = np.arange(nFrames, dtype=np.float32) * -0.5
    CR = CRShiftAndMagnify()
    bench_values = CR.benchmark(image, shift_row, shift_col, M, M)

    images = []
    titles = []
    run_times = []

    # unzip the values
    for run_time, title, image in bench_values:
        run_times.append(run_time)
        titles.append(title)
        images.append(image)

    # show images
    fig, axes = plt.subplots(nFrames, len(images), figsize=(20, 10))
    for i in range(nFrames):
        for j in range(len(images)):
            if i == 0:
                axes[i, j].set_title(titles[j])
            axes[i, j].imshow(images[j][i], cmap="hot")
            axes[i, j].axis("off")
