from nanopyx.liquid._mandelbrot_benchmark import mandelbrot


def test_mandelbrot_benchmark(plt):
    im = mandelbrot()
    plt.imshow(im)
