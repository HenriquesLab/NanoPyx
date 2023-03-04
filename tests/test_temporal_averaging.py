from nanopyx.methods.drift_alignment.estimator import DriftEstimator


def test_no_averaging(random_timelapse_w_drift):
    estimator = DriftEstimator()
    estimator.estimator_table.params["time_averaging"] = 1
    avg_img = estimator.compute_temporal_averaging(random_timelapse_w_drift)

    assert random_timelapse_w_drift.shape == avg_img.shape


def test_no_averaging_with_roi(random_timelapse_w_drift):
    estimator = DriftEstimator()
    estimator.estimator_table.params["time_averaging"] = 1
    estimator.estimator_table.params["use_roi"] = True
    x0, y0, x1, y1 = 10, 10, 100, 100
    estimator.estimator_table.params["roi"] = x0, y0, x1, y1
    img = random_timelapse_w_drift[:, y0 : y1 + 1, x0 : x1 + 1]
    avg_img = estimator.compute_temporal_averaging(img)

    assert img.shape == avg_img.shape
