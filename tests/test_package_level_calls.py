import nanopyx
import numpy as np


def test_package_nlm():
    img = np.random.random((100, 100)).astype(np.float32)
    nanopyx.non_local_means_denoising(img)


def test_package_SRRF():
    img = np.random.random((10, 100, 100)).astype(np.float32)
    nanopyx.SRRF(img)


def test_package_eSRRF():
    img = np.random.random((10, 100, 100)).astype(np.float32)
    nanopyx.eSRRF(img)


def test_package_frc():
    img = np.random.random((2, 100, 100)).astype(np.float32)
    nanopyx.calculate_frc(img[0], img[1])


def test_package_decorr():
    img = np.random.random((100, 100)).astype(np.float32)
    nanopyx.calculate_decorr_analysis(img)


def test_package_error_map():
    img = np.random.random((100, 100)).astype(np.float32)
    img_2 = np.random.random((500, 500)).astype(np.float32)
    nanopyx.calculate_error_map(img, img_2)
