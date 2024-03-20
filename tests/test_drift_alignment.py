import os

import numpy as np

from nanopyx.methods.drift_alignment import (apply_drift_alignment,
                                             estimate_drift_alignment)
from nanopyx.methods.drift_alignment.estimator import DriftEstimator


def test_drift_alignment_no_temporal_averaging_previous_frame(random_timelapse_w_drift):
    estimator = DriftEstimator()
    aligned_img = estimator.estimate(random_timelapse_w_drift, ref_option=0, apply=True)
    drift_table = estimator.estimator_table.drift_table

    drift_x = drift_table[:, 1]
    drift_y = drift_table[:, 2]

    pass_test = 1

    for i in range(len(drift_x)):
        if drift_x[i] < ((1 * i) - (0.1 * (1 * i))) or drift_x[i] > ((1 * i) + (0.1 * (1 * i))):
            pass_test = 0
        if drift_y[i] < ((1 * i) - (0.1 * (1 * i))) or drift_y[i] > ((1 * i) + (0.1 * (1 * i))):
            pass_test = 0

    assert pass_test == 1


def test_drift_alignment_w_temporal_averaging_previous_frame(random_timelapse_w_drift):
    estimator = DriftEstimator()
    aligned_img = estimator.estimate(random_timelapse_w_drift, ref_option=0, time_averaging=5, apply=True)
    drift_table = estimator.estimator_table.drift_table

    drift_x = drift_table[:, 1]
    drift_y = drift_table[:, 2]

    pass_test = 1

    for i in range(len(drift_x)):
        if drift_x[i] < ((1 * i) - (0.1 * (1 * i))) or drift_x[i] > ((1 * i) + (0.1 * (1 * i))):
            pass_test = 0
        if drift_y[i] < ((1 * i) - (0.1 * (1 * i))) or drift_y[i] > ((1 * i) + (0.1 * (1 * i))):
            pass_test = 0

    assert pass_test == 1


def test_drift_alignment_no_temporal_averaging_initial_frame(random_timelapse_w_drift):
    estimator = DriftEstimator()
    aligned_img = estimator.estimate(random_timelapse_w_drift, ref_option=1, apply=True)
    drift_table = estimator.estimator_table.drift_table

    drift_x = drift_table[:, 1]
    drift_y = drift_table[:, 2]

    pass_test = 1

    for i in range(len(drift_x)):
        if drift_x[i] < ((1 * i) - (0.1 * (1 * i))) or drift_x[i] > ((1 * i) + (0.1 * (1 * i))):
            pass_test = 0
        if drift_y[i] < ((1 * i) - (0.1 * (1 * i))) or drift_y[i] > ((1 * i) + (0.1 * (1 * i))):
            pass_test = 0

    assert pass_test == 1


def test_drift_alignment_w_temporal_averaging_initial_frame(random_timelapse_w_drift):
    estimator = DriftEstimator()
    aligned_img = estimator.estimate(random_timelapse_w_drift, ref_option=1, time_averaging=5, apply=True)
    drift_table = estimator.estimator_table.drift_table

    drift_x = drift_table[:, 1]
    drift_y = drift_table[:, 2]

    pass_test = 1

    for i in range(len(drift_x)):
        if drift_x[i] < ((1 * i) - (0.1 * (1 * i))) or drift_x[i] > ((1 * i) + (0.1 * (1 * i))):
            pass_test = 0
        if drift_y[i] < ((1 * i) - (0.1 * (1 * i))) or drift_y[i] > ((1 * i) + (0.1 * (1 * i))):
            pass_test = 0

    assert pass_test == 1


def test_drift_alignment_init(random_timelapse_w_drift):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    aligned_image = estimate_drift_alignment(random_timelapse_w_drift, ref_option=0, apply=True,
                                             save_drift_table_path=file_dir + os.sep)
    estimate_drift_alignment(random_timelapse_w_drift, ref_option=0, apply=False,
                             save_drift_table_path=file_dir + os.sep, save_as_npy=False)

    assert random_timelapse_w_drift.shape == aligned_image.shape


def test_apply_drift_alignment_init(random_timelapse_w_drift):
    estimator = DriftEstimator()
    file_dir = os.path.dirname(os.path.realpath(__file__))
    aligned_img = estimator.estimate(random_timelapse_w_drift, ref_option=1, time_averaging=5, apply=True)
    aligned_img_corrector = apply_drift_alignment(random_timelapse_w_drift, drift_table=estimator.estimator_table, path=None)

    assert np.array_equal(aligned_img, aligned_img_corrector)


def test_apply_drift_alignment_init_previous_drift_table(random_timelapse_w_drift):
    estimator = DriftEstimator()
    aligned_img = estimator.estimate(random_timelapse_w_drift, ref_option=1, time_averaging=5, max_expected_drift=100,
                                     apply=True)
    file_dir = os.path.dirname(os.path.realpath(__file__))
    estimator.save_drift_table(save_as_npy=False, path=file_dir + os.sep)
    aligned_img_corrector_2 = apply_drift_alignment(random_timelapse_w_drift, path=os.path.join(file_dir,"_drift_table.csv"))

    assert np.array_equal(aligned_img, aligned_img_corrector_2)


def test_apply_drift_alignment_init_previous_drift_table_npy(random_timelapse_w_drift):
    estimator = DriftEstimator()
    aligned_img = estimator.estimate(random_timelapse_w_drift, ref_option=1, time_averaging=5, apply=True)
    file_dir = os.path.dirname(os.path.realpath(__file__))
    estimator.save_drift_table(save_as_npy=True, path=file_dir + os.sep)
    aligned_img_corrector_2 = apply_drift_alignment(random_timelapse_w_drift, path=os.path.join(file_dir, "_drift_table.npy"))

    assert np.array_equal(aligned_img, aligned_img_corrector_2)

