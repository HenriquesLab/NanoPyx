from nanopyx.core.utils import MandelbrotBenchmark

# flake8: noqa: E501
def test_mandelbrot_benchmark(plt):
    mb = MandelbrotBenchmark(testing=True,clear_benchmarks=True)
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

