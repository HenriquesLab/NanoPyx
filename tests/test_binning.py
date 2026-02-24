from nanopyx.core.transform.binning import rebin_2d
from nanopyx.core.generate.beads import generate_timelapse_drift


def test_rebin_2d_sum():
    random_timelapse_w_drift = generate_timelapse_drift(
        n_objects=5, shape=(50, 500, 500), drift=1
    )
    binned_arr = rebin_2d(random_timelapse_w_drift, 5)

    assert binned_arr.shape == (
        random_timelapse_w_drift.shape[0],
        int(random_timelapse_w_drift.shape[1] / 5),
        int(random_timelapse_w_drift.shape[2] / 5),
    )


def test_rebin_2d_mean():
    random_timelapse_w_drift = generate_timelapse_drift(
        n_objects=5, shape=(50, 500, 500), drift=1
    )
    binned_arr = rebin_2d(random_timelapse_w_drift, 5, mode="mean")

    assert binned_arr.shape == (
        random_timelapse_w_drift.shape[0],
        int(random_timelapse_w_drift.shape[1] / 5),
        int(random_timelapse_w_drift.shape[2] / 5),
    )


def test_rebin_2d_max():
    random_timelapse_w_drift = generate_timelapse_drift(
        n_objects=5, shape=(50, 500, 500), drift=1
    )
    binned_arr = rebin_2d(random_timelapse_w_drift, 5, mode="max")

    assert binned_arr.shape == (
        random_timelapse_w_drift.shape[0],
        int(random_timelapse_w_drift.shape[1] / 5),
        int(random_timelapse_w_drift.shape[2] / 5),
    )
