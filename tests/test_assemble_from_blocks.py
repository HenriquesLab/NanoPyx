from nanopyx.core.transform.blocks import assemble_frame_from_blocks
from nanopyx.core.generate.beads import generate_timelapse_drift


def test_assemble_blocks(random_timelapse_w_drift):
    w = random_timelapse_w_drift.shape[2]
    h = random_timelapse_w_drift.shape[1]
    print(w, h)
    new_arr = assemble_frame_from_blocks(random_timelapse_w_drift, 10, 5)

    print(new_arr.shape)

    assert new_arr.shape == (h*10, w*5)