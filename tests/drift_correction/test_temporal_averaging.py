from nanopyx_core.methods.drift_correction.drift_estimator import DriftEstimator

import os
import numpy as np
from skimage.io import imread


def test_no_averaging():
    estimator = DriftEstimator()
    img = imread("src" + os.sep + "enanoscopy" + os.sep + "images" + os.sep + "drift_alignment" + os.sep + "drift_test.tif")
    estimator.estimator_table.params["time_averaging"] = 1
    avg_img = estimator.compute_temporal_averaging(img)

    assert img.shape == avg_img.shape


def test_no_averaging_with_roi():
    estimator = DriftEstimator()
    img = imread("src" + os.sep + "enanoscopy" + os.sep + "images" + os.sep + "drift_alignment" + os.sep + "drift_test.tif")
    estimator.estimator_table.params["time_averaging"] = 1
    estimator.estimator_table.params["use_roi"] = True
    x0, y0, x1, y1 = 10, 10, 100, 100
    estimator.estimator_table.params["roi"] = x0, y0, x1, y1
    img = img[:, y0:y1+1, x0:x1+1]
    avg_img = estimator.compute_temporal_averaging(img)

    assert img.shape == avg_img.shape


def test_averaging():
    estimator = DriftEstimator()
    img = imread("src" + os.sep + "enanoscopy" + os.sep + "images" + os.sep + "drift_alignment" + os.sep + "test_ones_zeros.tif")
    estimator.estimator_table.params["time_averaging"] = 100
    avg_img = estimator.compute_temporal_averaging(img)
    n_blocks = int(img.shape[0] / estimator.estimator_table.params["time_averaging"])
    if (img.shape[0] % estimator.estimator_table.params["time_averaging"]) != 0:
        n_blocks += 1

    assert int(np.sum(avg_img)) == int(np.sum(img)) / (img.shape[0] / n_blocks)


def test_averaging_with_roi():
    estimator = DriftEstimator()
    img = imread("src" + os.sep + "enanoscopy" + os.sep + "images" + os.sep + "drift_alignment" + os.sep + "test_ones_zeros.tif")
    estimator.estimator_table.params["time_averaging"] = 100
    estimator.estimator_table.params["use_roi"] = True
    x0, y0, x1, y1 = 0, 0, 24, 24
    estimator.estimator_table.params["roi"] = x0, y0, x1, y1
    img = img[:, y0:y1+1, x0:x1+1]
    avg_img = estimator.compute_temporal_averaging(img)
    n_blocks = int(img.shape[0] / estimator.estimator_table.params["time_averaging"])

    assert int(np.sum(avg_img)) == int(np.sum(img)) / (img.shape[0] / n_blocks)