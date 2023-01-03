
import pytest
import numpy as np
import random

from nanopyx.core.transform.image_add_random_noise import addSquares, addRamp, getRamp
from nanopyx.core.utils.imagegenerator.beads import generate_timelapse_drift

@pytest.fixture
def random_image_with_ramp_squares():
    w = random.randint(1, 32)
    h = random.randint(1, 32)
    image = np.zeros((w, h), dtype="float32")
    addRamp(image, 1000)
    addSquares(image, 100, nSquares=10)
    return image

@pytest.fixture
def random_image_with_ramp():
    return getRamp(64, 64)

@pytest.fixture
def random_timelapse_w_drift():
    return generate_timelapse_drift(n_objects=5, shape=(50, 150, 150), drift=1)
