
import random

import numpy as np
import pytest

from nanopyx.core.generate.noise_add_squares import add_squares, get_squares
from nanopyx.core.generate.noise_add_ramp import add_ramp, get_ramp
from nanopyx.core.generate.beads import generate_timelapse_drift, generate_channel_misalignment
from nanopyx.core.analysis.pearson_correlation import pearson_correlation
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

@pytest.fixture
def compare():
    def compare_imgs(output_1, output_2):
        if output_1.ndim > 2:
            pcc = 0
            for i in range(output_1.shape[0]):
                pcc += pearson_correlation(output_1[i, :, :], output_2[i, :, :])
            pcc /= output_1.shape[0]
        else:
            pcc = pearson_correlation(output_1, output_2)

        if pcc > 0.8:
            return True
        else:
            return False
    return compare_imgs