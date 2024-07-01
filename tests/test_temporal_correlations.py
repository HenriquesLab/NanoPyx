import numpy as np
from nanopyx.core.transform.sr_temporal_correlations import *


def test_srrf_tcorr_order_n1():
    img = np.random.random((10, 100, 100)).astype(np.float32)
    calculate_SRRF_temporal_correlations(img, order=-1)


def test_srrf_tcorr_order_0():
    img = np.random.random((10, 100, 100)).astype(np.float32)
    calculate_SRRF_temporal_correlations(img, order=0)


def test_srrf_tcorr_order_1():
    img = np.random.random((10, 100, 100)).astype(np.float32)
    calculate_SRRF_temporal_correlations(img, order=1)


def test_srrf_tcorr_order_2():
    img = np.random.random((10, 100, 100)).astype(np.float32)
    calculate_SRRF_temporal_correlations(img, order=2)


def test_srrf_tcorr_order_3():
    img = np.random.random((10, 100, 100)).astype(np.float32)
    calculate_SRRF_temporal_correlations(img, order=3)


def test_srrf_tcorr_order_4():
    img = np.random.random((10, 100, 100)).astype(np.float32)
    calculate_SRRF_temporal_correlations(img, order=4)


def test_esrrf_tcorr_avg():
    img = np.random.random((10, 100, 100))
    calculate_eSRRF_temporal_correlations(img, "AVG")


def test_esrrf_tcorr_var():
    img = np.random.random((10, 100, 100))
    calculate_eSRRF_temporal_correlations(img, "VAR")


def test_esrrf_tcorr_tac2():
    img = np.random.random((10, 100, 100))
    calculate_eSRRF_temporal_correlations(img, "TAC2")


def test_tac2():
    img = np.random.random((10, 100, 100))
    calculate_tac2(img)


def test_esrrf3d_temporal_correlations():
    img = np.random.random((10, 10, 100, 100))
    calculate_eSRRF3d_temporal_correlations(img)


def test_esrrf3d_temporal_correlations():
    img = np.random.random((10, 10, 100, 100))
    calculate_eSRRF3d_temporal_correlations(img)


def test_esrrf_corr():
    img = np.random.random((10, 100, 100))
    try:
        calculate_eSRRF_temporal_correlations(img, correlation="blabla")
    except ValueError:
        assert True


def test_esrrf3d_temporal_correlations_missing_dims():
    img = np.random.random((1, 10, 100, 100))
    calculate_eSRRF3d_temporal_correlations(img)


def test_acrf():
    img = np.random.random((20, 100, 100))
    calculate_acrf_(img, 2, 1)
    calculate_acrf_(img, 4, 1)
