import math
from nanopyx.core.analysis.decorr import DecorrAnalysis


def test_decorr_analysis(random_timelapse_w_drift):
    decorr = DecorrAnalysis(pixel_size=1, units="pixel")
    decorr.run_analysis(random_timelapse_w_drift[0])
    assert not math.isinf(decorr.resolution)