import nanopyx
import numpy as np


def test_benchmark_interpolation_bicubic(random_image_with_ramp):
    bicubic_sm = nanopyx.core.transform._le_interpolation_bicubic.ShiftAndMagnify()
    bicubic_ssr = nanopyx.core.transform._le_interpolation_bicubic.ShiftScaleRotate()

    bicubic_sm.benchmark(random_image_with_ramp, 1, 1, 2, 2)
    bicubic_ssr.benchmark(random_image_with_ramp, 1, 1, 2, 2, 15)


def test_benchmark_interpolation_catmull(random_image_with_ramp):
    cr_sm = nanopyx.core.transform._le_interpolation_catmull_rom.ShiftAndMagnify()
    cr_ssr = nanopyx.core.transform._le_interpolation_catmull_rom.ShiftScaleRotate()

    cr_sm.benchmark(random_image_with_ramp, 1, 1, 2, 2)
    cr_ssr.benchmark(random_image_with_ramp, 1, 1, 2, 2, 15)


def test_benchmark_interpolation_lanczos(random_image_with_ramp):
    l_sm = nanopyx.core.transform._le_interpolation_catmull_rom.ShiftAndMagnify()
    l_ssr = nanopyx.core.transform._le_interpolation_catmull_rom.ShiftScaleRotate()

    l_sm.benchmark(random_image_with_ramp, 1, 1, 2, 2)
    l_ssr.benchmark(random_image_with_ramp, 1, 1, 2, 2, 15)


def test_benchmark_interpolation_nn(random_image_with_ramp):
    nn_sm = nanopyx.core.transform._le_interpolation_nearest_neighbor.ShiftAndMagnify()
    nn_ssr = nanopyx.core.transform._le_interpolation_nearest_neighbor.ShiftScaleRotate()
    nn_pt = nanopyx.core.transform._le_interpolation_nearest_neighbor.PolarTransform()

    nn_sm.benchmark(random_image_with_ramp, 1, 1, 2, 2)
    nn_ssr.benchmark(random_image_with_ramp, 1, 1, 2, 2, 15)
    nn_pt.benchmark(random_image_with_ramp, (random_image_with_ramp.shape[0], random_image_with_ramp.shape[1]), "log")


def test_benchmark_conv2d(random_image_with_ramp):
    conv2d = nanopyx.core.transform._le_convolution.Convolution()
    conv2d.benchmark(random_image_with_ramp, np.ones((3, 3)).astype(np.float32))


def test_benchmark_esrrf(random_image_with_ramp):
    esrrf = nanopyx.core.transform._le_esrrf.eSRRF()
    esrrf.benchmark(random_image_with_ramp)

def test_benchmark_esrrf3d():
    esrrf = nanopyx.core.transform._le_esrrf3d.eSRRF3D()
    esrrf.benchmark(np.random.random((1,10,10,10)).astype(np.float32))

def test_benchmark_nlm(random_image_with_ramp):
    nlm = nanopyx.core.transform._le_nlm_denoising.NLMDenoising()
    nlm.benchmark(random_image_with_ramp)


def test_benchmark_channel_reg(random_channel_misalignment):
    channel_reg = nanopyx.core.analysis._le_channel_registration.ChannelRegistrationEstimator()
    channel_reg.benchmark(random_channel_misalignment, 0, 10, 3, 0.5)


def test_benchmark_drift_correction(random_timelapse_w_drift):
    drift_reg = nanopyx.core.analysis._le_drift_calculator.DriftEstimator()
    drift_reg.benchmark(random_timelapse_w_drift)
