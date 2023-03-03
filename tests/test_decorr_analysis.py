import math
from nanopyx.core.analysis.decorr import DecorrAnalysis


def test_decorr_analysis(random_timelapse_w_drift):
    decorr = DecorrAnalysis(random_timelapse_w_drift[0].reshape((1, random_timelapse_w_drift.shape[1], random_timelapse_w_drift.shape[2])), pixel_size=1, units="pixel", do_plot=False)
    decorr.run_analysis()
    assert not math.isinf(decorr.resolution)