import math
from nanopyx.core.analysis.decorr import DecorrAnalysis


def test_decorr_analysis():
    from nanopyx.core.generate.beads import generate_timelapse_drift

    random_timelapse_w_drift = generate_timelapse_drift(
        n_objects=5, shape=(50, 500, 500), drift=1
    )
    decorr = DecorrAnalysis(pixel_size=1, units="pixel")
    decorr.run_analysis(random_timelapse_w_drift[0])
    assert not math.isinf(decorr.resolution)
