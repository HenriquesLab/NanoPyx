import pytest
import numpy as np
from nanopyx.core.analysis.rcc import calculate_x_corr, get_image_shift, rcc, minimize_shifts

def test_calculate_x_corr():
    im1 = np.random.rand(10, 10)
    im2 = np.random.rand(10, 10)
    ccm = calculate_x_corr(im1, im2)
    assert isinstance(ccm, np.ndarray)
    assert ccm.shape == im1.shape


def test_get_image_shift_no_shift():
    im1 = np.ones((10, 10))
    im2 = np.ones((10, 10))
    shift_x, shift_y = get_image_shift(im1, im2, 5, 10)
    assert shift_x == 0
    assert shift_y == 0


def test_rcc():
    im_frames = np.zeros((5, 10, 10))
    for i in range(im_frames.shape[0]):
        im_frames[i, :, 2+i] = 1
    shifts_x, shifts_y = rcc(im_frames, 10)
    assert isinstance(shifts_x, np.ndarray)
    assert isinstance(shifts_y, np.ndarray)
    assert shifts_x.shape[0] == im_frames.shape[0]
    assert shifts_y.shape[0] == im_frames.shape[0]


def test_minimize_shifts():
    shifts_x = np.array([[0, 1, 2], [0, 0, 1], [0, 0, 0]])
    shifts_y = np.array([[0, -1, -2], [0, 0, -1], [0, 0, 0]])
    shift_x, shift_y = minimize_shifts(shifts_x, shifts_y)
    assert isinstance(shift_x, np.ndarray)
    assert isinstance(shift_y, np.ndarray)
    assert shift_x.shape == (3,)
    assert shift_y.shape == (3,)
