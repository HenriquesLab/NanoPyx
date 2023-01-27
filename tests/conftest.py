
import random

import numpy as np
import pytest

from nanopyx.core.image.add_squares import add_squares, get_squares
from nanopyx.core.image.add_ramp import add_ramp, get_ramp
from nanopyx.core.image.generate.beads import generate_timelapse_drift, generate_channel_misalignment
from nanopyx.data.download import ExampleDataManager


@pytest.fixture
def random_image_with_ramp_squares():
    w = random.randint(32, 64)
    h = random.randint(32, 64)
    image = np.zeros((w, h), dtype="float32")
    add_ramp(image, 1000)
    add_squares(image, 100, nSquares=10)
    return image


@pytest.fixture
def random_image_with_ramp():
    return get_ramp(64, 64)


@pytest.fixture
def random_image_with_squares():
    return get_squares(128, 64, nSquares=10)


@pytest.fixture
def random_timelapse_w_drift():
    return generate_timelapse_drift(n_objects=5, shape=(50, 500, 500), drift=1)


@pytest.fixture
def random_channel_misalignment():
    return generate_channel_misalignment()


@pytest.fixture
def downloader():
    return ExampleDataManager()
