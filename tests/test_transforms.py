import numpy as np

from nanopyx.core.generate.noise_add_simplex import get_simplex_noise
from nanopyx.core.transform import BCShiftAndMagnify, BCShiftScaleRotate, CRShiftAndMagnify, CRShiftScaleRotate
from nanopyx.core.transform import NNShiftAndMagnify, NNShiftScaleRotate

from nanopyx.__opencl__ import print_opencl_info

def test_interpolation_nearest_neighbor_ShiftAndMagnify(plt):
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 32, frames=nFrames, amplitude=1000)
    shift_row = 5.0
    shift_col = 5.0
    SM = NNShiftAndMagnify(testing=True, clear_benchmarks=True)
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


def test_interpolation_bicubic_ShiftAndMagnify(plt):
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 32, frames=nFrames, amplitude=1000)
    shift_row = 5.0
    shift_col = 5.0
    SM = BCShiftAndMagnify(testing=True, clear_benchmarks=True)
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


def test_interpolation_catmull_rom_ShiftAndMagnify(plt):
    print(print_opencl_info)
    
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 32, frames=nFrames, amplitude=1000)
    shift_row = 5.0
    shift_col = 5.0
    SM = CRShiftAndMagnify(testing=True, clear_benchmarks=True)
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



# def test_interpolation_lanczos_ShiftAndMagnify():
#     M = 4
#     nFrames = 3
#     image = get_simplex_noise(64, 32, frames=nFrames, amplitude=1000)
#     shift_row = 5.0
#     shift_col = 5.0
#     SM = LZShiftAndMagnify(testing=True, clear_benchmarks=True)
#     bench_values = SM.benchmark(image, shift_row, shift_col, M, M)

#     images = []
#     titles = []
#     run_times = []

#     # unzip the values
#     for run_time, title, image in bench_values:
#         run_times.append(run_time)
#         titles.append(title)
#         images.append(image)

#     # ensure images are similar
#     for i in range(len(images)):
#         for j in range(i + 1, len(images)):
#             np.testing.assert_allclose(images[i], images[j], rtol=1e1)

#     nFrames = images[0].shape[0]
#     # show images
#     fig, axes = plt.subplots(nFrames, len(images), figsize=(20, 10))
#     for i in range(nFrames):
#         for j in range(len(images)):
#             if i == 0:
#                 axes[i, j].set_title(titles[j])
#             axes[i, j].imshow(images[j][i], cmap="hot")
#             axes[i, j].axis("off")



def test_interpolation_nearest_neighbor_ShiftScaleRotate(plt): 
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 32, frames=nFrames, amplitude=1000)
    shift_row = 5.0
    shift_col = 5.0
    angle = np.pi / 4
    SM = NNShiftScaleRotate(testing=True,clear_benchmarks=True)
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


def test_interpolation_bicubic_ShiftScaleRotate(plt): 
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 32, frames=nFrames, amplitude=1000)
    shift_row = 5.0
    shift_col = 5.0
    angle = np.pi / 4
    SM = BCShiftScaleRotate(testing=True,clear_benchmarks=True)
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



def test_interpolation_catmull_rom_ShiftScaleRotate(plt): 
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 32, frames=nFrames, amplitude=1000)
    shift_row = 5.0
    shift_col = 5.0
    angle = np.pi / 4
    SM = CRShiftScaleRotate(testing=True,clear_benchmarks=True)
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


# def test_interpolation_lanzcos_ShiftScaleRotate(): 
#     M = 4
#     nFrames = 3
#     image = get_simplex_noise(64, 32, frames=nFrames, amplitude=1000)
#     shift_row = 5.0
#     shift_col = 5.0
#     angle = np.pi / 4
#     SM = LZShiftScaleRotate(testing=True,clear_benchmarks=True)
#     bench_values = SM.benchmark(image, shift_row, shift_col, M, M, angle)

#     images = []
#     titles = []
#     run_times = []

#     # unzip the values
#     for run_time, title, image in bench_values:
#         run_times.append(run_time)  
#         titles.append(title)
#         images.append(image)

#     # ensure images are similar
#     for i in range(len(images)):
#         for j in range(i + 1, len(images)):
#             np.testing.assert_allclose(images[i], images[j], rtol=1e1)

#     # show images
#     fig, axes = plt.subplots(nFrames, len(images), figsize=(20, 10))
#     for i in range(nFrames):
#         for j in range(len(images)):
#             if i == 0:
#                 axes[i, j].set_title(titles[j])
#             axes[i, j].imshow(images[j][i], cmap="hot")
#             axes[i, j].axis("off")


