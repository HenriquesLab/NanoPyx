from nanopyx.liquid import MandelbrotBenchmark


def test_mandelbrot_benchmark(plt):
    mb = MandelbrotBenchmark()
    images = mb.benchmark(128)
    assert len(images) == 3
    for image in images:
        assert image.shape == (128, 128)

    # check that images have same value
    if images[0] is not None and images[1] is not None:
        assert (images[0] == images[1]).all()
    if images[1] is not None and images[2] is not None:
        assert (images[1] == images[2]).all()

    # show the 3 images with titles
    fig, axes = plt.subplots(1, 3)
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.set_title(mb.RUN_TYPE_DESIGNATION[i])
    plt.show()
