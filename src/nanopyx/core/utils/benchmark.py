import math
import nanopyx
import numpy as np


def benchmark_all_le_methods(
    n_benchmark_runs=3, img_dims=100, shift=2, magnification=5, rotation=math.radians(15), conv_kernel_dims=23
):
    """
    Runs benchmark tests for all LE methods.
    Args:
        n_benchmark_runs (int): The number of benchmark runs to perform. Default is 3.
        img_dims (int): The dimensions of the input image. Default is 100.
        shift (int): The amount of shift to apply to the image during benchmarking. Default is 2.
        magnification (int): The magnification factor to apply to the image during benchmarking. Default is 5.
        rotation (float): The rotation angle to apply to the image during benchmarking. Default is 0.2617993877991494 (equal to 15 degrees in radians).
        conv_kernel_dims (int): The dimensions of the convolution kernel to use during benchmarking. Default is 23.
    Returns:
        None
    """

    img = np.random.random((img_dims, img_dims)).astype(np.float32)
    img_int = np.random.random((img_dims * magnification, img_dims * magnification)).astype(np.float32)
    kernel = np.ones((conv_kernel_dims, conv_kernel_dims)).astype(np.float32)

    bicubic_sm = nanopyx.core.transform._le_interpolation_bicubic.ShiftAndMagnify()
    bicubic_ssr = nanopyx.core.transform._le_interpolation_bicubic.ShiftScaleRotate()
    cr_sm = nanopyx.core.transform._le_interpolation_catmull_rom.ShiftAndMagnify()
    cr_ssr = nanopyx.core.transform._le_interpolation_catmull_rom.ShiftScaleRotate()
    l_sm = nanopyx.core.transform._le_interpolation_lanczos.ShiftAndMagnify()
    l_ssr = nanopyx.core.transform._le_interpolation_lanczos.ShiftScaleRotate()
    nn_sm = nanopyx.core.transform._le_interpolation_nearest_neighbor.ShiftAndMagnify()
    nn_ssr = nanopyx.core.transform._le_interpolation_nearest_neighbor.ShiftScaleRotate()
    nn_pt = nanopyx.core.transform._le_interpolation_nearest_neighbor.PolarTransform()

    conv2d = nanopyx.core.transform._le_convolution.Convolution()

    rad = nanopyx.core.transform._le_radiality.Radiality()
    rc = nanopyx.core.transform._le_roberts_cross_gradients.GradientRobertsCross()
    rgc = nanopyx.core.transform._le_radial_gradient_convergence.RadialGradientConvergence()

    esrrf = nanopyx.core.transform._le_esrrf.eSRRF()

    nlm = nanopyx.core.transform._le_nlm_denoising.NLMDenoising()

    for i in range(n_benchmark_runs):
        bicubic_sm.benchmark(img, shift, shift, magnification, magnification)
    for i in range(n_benchmark_runs):
        cr_sm.benchmark(img, shift, shift, magnification, magnification)
    for i in range(n_benchmark_runs):
        l_sm.benchmark(img, shift, shift, magnification, magnification)
    for i in range(n_benchmark_runs):
        nn_sm.benchmark(img, shift, shift, magnification, magnification)

    for i in range(n_benchmark_runs):
        bicubic_ssr.benchmark(img, shift, shift, magnification, magnification, rotation)
    for i in range(n_benchmark_runs):
        cr_ssr.benchmark(img, shift, shift, magnification, magnification, rotation)
    for i in range(n_benchmark_runs):
        l_ssr.benchmark(img, shift, shift, magnification, magnification, rotation)
    for i in range(n_benchmark_runs):
        nn_ssr.benchmark(img, shift, shift, magnification, magnification, rotation)

    for i in range(n_benchmark_runs):
        nn_pt.benchmark(img, (img_dims, img_dims), "log")

    for i in range(n_benchmark_runs):
        conv2d.benchmark(img, kernel)

    for i in range(n_benchmark_runs):
        rad.benchmark(img, img_int)
    for i in range(n_benchmark_runs):
        rc.benchmark(img)
    for i in range(n_benchmark_runs):
        rgc.benchmark(img_int, img_int, img_int)

    for i in range(n_benchmark_runs):
        esrrf.benchmark(img)

    for i in range(n_benchmark_runs):
        nlm.benchmark(img)
