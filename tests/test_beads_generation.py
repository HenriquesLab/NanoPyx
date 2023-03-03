from nanopyx.core.generate.beads import generate_timelapse_drift


def test_generate_random_drift():
    img = generate_timelapse_drift(shape=(10, 300, 300), drift=None, drift_mode="random")

    assert img.shape == (10, 300, 300)