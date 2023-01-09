from nanopyx.methods.drift_alignment.estimator import DriftEstimator


#TODO finish this
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

