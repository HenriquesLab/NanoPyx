import numpy as np

from nanopyx.core.generate.noise_add_simplex import get_simplex_noise
from nanopyx.core.transform import NNPolarTransform

from skimage.transform import warp_polar


def test_interpolation_nearest_neighbor_PolarTransform_linear(plt,compare): 
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 64, frames=nFrames, amplitude=1000)
    SM = NNPolarTransform(testing=True,clear_benchmarks=True)
    bench_values = SM.benchmark(image, (100,100), 'linear')

    skimage_linear = warp_polar(image, output_shape=(100,100), channel_axis=0, order=0)

    images = [skimage_linear]
    titles = ["Skimage"]
    run_times = ['nan']

    # unzip the values
    for run_time, title, image in bench_values:
        run_times.append(run_time)
        titles.append(title)
        images.append(image)

    # ensure images are similar
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            assert compare(images[i], images[j])
            
    nFrames = images[0].shape[0]
    # show images
    fig, axes = plt.subplots(nFrames, len(images), figsize=(20, 15))
    for i in range(nFrames):
        for j in range(len(images)):
            if i == 0:
                axes[i, j].set_title(titles[j])
            axes[i, j].imshow(images[j][i], cmap="hot")
            axes[i, j].axis("off")



def test_interpolation_nearest_neighbor_PolarTransform_log(plt,compare): 
    M = 4
    nFrames = 3
    image = get_simplex_noise(64, 64, frames=nFrames, amplitude=1000)
    SM = NNPolarTransform(testing=True,clear_benchmarks=True)
    bench_values = SM.benchmark(image, (100,100), 'log')

    skimage_linear = warp_polar(image, output_shape=(100,100), channel_axis=0, scaling='log', order=0)

    images = [skimage_linear]
    titles = ["Skimage"]
    run_times = ['nan']

    # unzip the values
    for run_time, title, image in bench_values:
        run_times.append(run_time)
        titles.append(title)
        images.append(image)

    #ensure images are similar
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            assert compare(images[i], images[j])

    nFrames = images[0].shape[0]
    # show images
    fig, axes = plt.subplots(nFrames, len(images), figsize=(20, 15))
    for i in range(nFrames):
        for j in range(len(images)):
            if i == 0:
                axes[i, j].set_title(titles[j])
            axes[i, j].imshow(images[j][i], cmap="hot")
            axes[i, j].axis("off")

