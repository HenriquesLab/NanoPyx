import numpy as np

from nanopyx.core.generate.noise_add_simplex import get_simplex_noise
from nanopyx.liquid._le_interpolation_bicubic import ShiftAndMagnify as BCShiftAndMagnify
from nanopyx.liquid._le_interpolation_bicubic import ShiftScaleRotate as BCShiftScaleRotate
from nanopyx.liquid._le_interpolation_catmull_rom import ShiftAndMagnify as CRShiftAndMagnify
from nanopyx.liquid._le_interpolation_catmull_rom import ShiftScaleRotate as CRShiftScaleRotate
from nanopyx.liquid._le_interpolation_lanczos import ShiftAndMagnify as LZShiftAndMagnify
from nanopyx.liquid._le_interpolation_lanczos import ShiftScaleRotate as LZShiftScaleRotate
from nanopyx.liquid._le_interpolation_nearest_neighbor import ShiftAndMagnify as NNShiftAndMagnify
from nanopyx.liquid._le_interpolation_nearest_neighbor import ShiftScaleRotate as NNShiftScaleRotate
from nanopyx.liquid._le_mandelbrot_benchmark import MandelbrotBenchmark
from nanopyx.liquid._le_radial_gradient_convergence import RadialGradientConvergence as RGC
from nanopyx.liquid._le_radiality import Radiality

# flake8: noqa: E501


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


# tag-start: test_interpolation_nearest_neighbor_ShiftAndMagnify
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

    # ensure images are similar
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            np.testing.assert_allclose(images[i], images[j], rtol=1e1)

    nFrames = images[0].shape[0]
    # show images
    fig, axes = plt.subplots(nFrames, len(images), figsize=(20, 10))
    for i in range(nFrames):
        for j in range(len(images)):
            if i == 0:
                axes[i, j].set_title(titles[j])
            axes[i, j].imshow(images[j][i], cmap="hot")
            axes[i, j].axis("off")


# tag-end


# tag-copy: test_interpolation_nearest_neighbor_ShiftAndMagnify; replace("nearest_neighbor", "bicubic"); replace("NNShiftAndMagnify", "BCShiftAndMagnify")
def test_interpolation_bicubic_ShiftAndMagnify(plt):
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 32, frames=nFrames, amplitude=1000)
    shift_row = np.arange(nFrames, dtype=np.float32) * 0.5
    shift_col = np.arange(nFrames, dtype=np.float32) * -0.5
    SM = BCShiftAndMagnify()
    bench_values = SM.benchmark(image, shift_row, shift_col, M, M)

    images = []
    titles = []
    run_times = []

    # unzip the values
    for run_time, title, image in bench_values:
        run_times.append(run_time)
        titles.append(title)
        images.append(image)

    # ensure images are similar
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            np.testing.assert_allclose(images[i], images[j], rtol=1e1)

    nFrames = images[0].shape[0]
    # show images
    fig, axes = plt.subplots(nFrames, len(images), figsize=(20, 10))
    for i in range(nFrames):
        for j in range(len(images)):
            if i == 0:
                axes[i, j].set_title(titles[j])
            axes[i, j].imshow(images[j][i], cmap="hot")
            axes[i, j].axis("off")


# tag-end


# tag-copy: test_interpolation_nearest_neighbor_ShiftAndMagnify; replace("nearest_neighbor", "catmull_rom"); replace("NNShiftAndMagnify", "CRShiftAndMagnify")
def test_interpolation_catmull_rom_ShiftAndMagnify(plt):
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 32, frames=nFrames, amplitude=1000)
    shift_row = np.arange(nFrames, dtype=np.float32) * 0.5
    shift_col = np.arange(nFrames, dtype=np.float32) * -0.5
    SM = CRShiftAndMagnify()
    bench_values = SM.benchmark(image, shift_row, shift_col, M, M)

    images = []
    titles = []
    run_times = []

    # unzip the values
    for run_time, title, image in bench_values:
        run_times.append(run_time)
        titles.append(title)
        images.append(image)

    # ensure images are similar
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            np.testing.assert_allclose(images[i], images[j], rtol=1e1)

    nFrames = images[0].shape[0]
    # show images
    fig, axes = plt.subplots(nFrames, len(images), figsize=(20, 10))
    for i in range(nFrames):
        for j in range(len(images)):
            if i == 0:
                axes[i, j].set_title(titles[j])
            axes[i, j].imshow(images[j][i], cmap="hot")
            axes[i, j].axis("off")


# tag-end


# tag-copy: test_interpolation_nearest_neighbor_ShiftAndMagnify; replace("nearest_neighbor", "lanczos"); replace("NNShiftAndMagnify", "LZShiftAndMagnify")
def test_interpolation_lanczos_ShiftAndMagnify(plt):
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 32, frames=nFrames, amplitude=1000)
    shift_row = np.arange(nFrames, dtype=np.float32) * 0.5
    shift_col = np.arange(nFrames, dtype=np.float32) * -0.5
    SM = LZShiftAndMagnify()
    bench_values = SM.benchmark(image, shift_row, shift_col, M, M)

    images = []
    titles = []
    run_times = []

    # unzip the values
    for run_time, title, image in bench_values:
        run_times.append(run_time)
        titles.append(title)
        images.append(image)

    # ensure images are similar
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            np.testing.assert_allclose(images[i], images[j], rtol=1e1)

    nFrames = images[0].shape[0]
    # show images
    fig, axes = plt.subplots(nFrames, len(images), figsize=(20, 10))
    for i in range(nFrames):
        for j in range(len(images)):
            if i == 0:
                axes[i, j].set_title(titles[j])
            axes[i, j].imshow(images[j][i], cmap="hot")
            axes[i, j].axis("off")


# tag-end


# tag-start: test_interpolation_nearest_neighbor_ShiftScaleRotate
def test_interpolation_nearest_neighbor_ShiftScaleRotate(plt):
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 32, frames=nFrames, amplitude=1000)
    shift_row = np.arange(nFrames, dtype=np.float32) * 0.5
    shift_col = np.arange(nFrames, dtype=np.float32) * -0.5
    angle = np.pi / 4
    SM = NNShiftScaleRotate()
    bench_values = SM.benchmark(image, shift_row, shift_col, M, M, angle)

    images = []
    titles = []
    run_times = []

    # unzip the values
    for run_time, title, image in bench_values:
        run_times.append(run_time)
        titles.append(title)
        images.append(image)

    # ensure images are similar
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            np.testing.assert_allclose(images[i], images[j], rtol=1e1)

    # show images
    fig, axes = plt.subplots(nFrames, len(images), figsize=(20, 10))
    for i in range(nFrames):
        for j in range(len(images)):
            if i == 0:
                axes[i, j].set_title(titles[j])
            axes[i, j].imshow(images[j][i], cmap="hot")
            axes[i, j].axis("off")


# tag-end


# tag-copy: test_interpolation_nearest_neighbor_ShiftScaleRotate; replace("nearest_neighbor", "bicubic"); replace("NNShiftScaleRotate", "BCShiftScaleRotate")
def test_interpolation_bicubic_ShiftScaleRotate(plt):
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 32, frames=nFrames, amplitude=1000)
    shift_row = np.arange(nFrames, dtype=np.float32) * 0.5
    shift_col = np.arange(nFrames, dtype=np.float32) * -0.5
    angle = np.pi / 4
    SM = BCShiftScaleRotate()
    bench_values = SM.benchmark(image, shift_row, shift_col, M, M, angle)

    images = []
    titles = []
    run_times = []

    # unzip the values
    for run_time, title, image in bench_values:
        run_times.append(run_time)
        titles.append(title)
        images.append(image)

    # ensure images are similar
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            np.testing.assert_allclose(images[i], images[j], rtol=1e1)

    # show images
    fig, axes = plt.subplots(nFrames, len(images), figsize=(20, 10))
    for i in range(nFrames):
        for j in range(len(images)):
            if i == 0:
                axes[i, j].set_title(titles[j])
            axes[i, j].imshow(images[j][i], cmap="hot")
            axes[i, j].axis("off")


# tag-end


# tag-copy: test_interpolation_nearest_neighbor_ShiftScaleRotate; replace("nearest_neighbor", "catmull_rom"); replace("NNShiftScaleRotate", "CRShiftScaleRotate")
def test_interpolation_catmull_rom_ShiftScaleRotate(plt):
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 32, frames=nFrames, amplitude=1000)
    shift_row = np.arange(nFrames, dtype=np.float32) * 0.5
    shift_col = np.arange(nFrames, dtype=np.float32) * -0.5
    angle = np.pi / 4
    SM = CRShiftScaleRotate()
    bench_values = SM.benchmark(image, shift_row, shift_col, M, M, angle)

    images = []
    titles = []
    run_times = []

    # unzip the values
    for run_time, title, image in bench_values:
        run_times.append(run_time)
        titles.append(title)
        images.append(image)

    # ensure images are similar
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            np.testing.assert_allclose(images[i], images[j], rtol=1e1)

    # show images
    fig, axes = plt.subplots(nFrames, len(images), figsize=(20, 10))
    for i in range(nFrames):
        for j in range(len(images)):
            if i == 0:
                axes[i, j].set_title(titles[j])
            axes[i, j].imshow(images[j][i], cmap="hot")
            axes[i, j].axis("off")


# tag-end


# tag-copy: test_interpolation_nearest_neighbor_ShiftScaleRotate; replace("nearest_neighbor", "lanzcos"); replace("NNShiftScaleRotate", "LZShiftScaleRotate")
def test_interpolation_lanzcos_ShiftScaleRotate(plt):
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 32, frames=nFrames, amplitude=1000)
    shift_row = np.arange(nFrames, dtype=np.float32) * 0.5
    shift_col = np.arange(nFrames, dtype=np.float32) * -0.5
    angle = np.pi / 4
    SM = LZShiftScaleRotate()
    bench_values = SM.benchmark(image, shift_row, shift_col, M, M, angle)

    images = []
    titles = []
    run_times = []

    # unzip the values
    for run_time, title, image in bench_values:
        run_times.append(run_time)
        titles.append(title)
        images.append(image)

    # ensure images are similar
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            np.testing.assert_allclose(images[i], images[j], rtol=1e1)

    # show images
    fig, axes = plt.subplots(nFrames, len(images), figsize=(20, 10))
    for i in range(nFrames):
        for j in range(len(images)):
            if i == 0:
                axes[i, j].set_title(titles[j])
            axes[i, j].imshow(images[j][i], cmap="hot")
            axes[i, j].axis("off")


# tag-end

"""
def test_rgc(downloader):

    dataset = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=True)
    small_dataset = dataset[:10,:20,:20]

    liquid_rgc = RGC()
    imRad = liquid_rgc.run(small_dataset)


def test_radiality(downloader):

    dataset = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=True)
    small_dataset = dataset[:10,:20,:20]

    liquid_rad = Radiality()
    imRad = liquid_rad.run(small_dataset)
"""