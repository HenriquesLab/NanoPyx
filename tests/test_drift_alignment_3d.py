import pytest
import numpy as np
from nanopyx.methods.drift_alignment.estimator_3d import (
    Estimator3D,
)  # Adjust the import according to your module structure


@pytest.fixture
def image_stack():
    """
    Create a sample 3D image stack for testing.
    Shape: (t, z, y, x) = (5, 10, 10, 10)
    """
    return np.random.rand(5, 10, 10, 10)


def test_correct_xy_drift_mean_projection(image_stack):
    estimator = Estimator3D()
    estimator.image_array = image_stack.copy()
    estimator.correct_xy_drift(projection_mode="Mean")
    assert estimator.image_array.shape == image_stack.shape
    # Add more assertions to verify drift correction if applicable


def test_correct_xy_drift_max_projection(image_stack):
    estimator = Estimator3D()
    estimator.image_array = image_stack.copy()
    estimator.correct_xy_drift(projection_mode="Max")
    assert estimator.image_array.shape == image_stack.shape
    # Add more assertions to verify drift correction if applicable


def test_correct_z_drift_top_mean_projection(image_stack):
    estimator = Estimator3D()
    estimator.image_array = image_stack.copy()
    estimator.correct_z_drift(axis_mode="top", projection_mode="Mean")
    assert estimator.image_array.shape == image_stack.shape
    # Add more assertions to verify drift correction if applicable


def test_correct_z_drift_left_max_projection(image_stack):
    estimator = Estimator3D()
    estimator.image_array = image_stack.copy()
    estimator.correct_z_drift(axis_mode="left", projection_mode="Max")
    assert estimator.image_array.shape == image_stack.shape
    # Add more assertions to verify drift correction if applicable


def test_correct_3d_drift(image_stack):
    estimator = Estimator3D()
    corrected_stack = estimator.correct_3d_drift(image_stack, axis_mode="top", projection_mode="Mean")
    assert corrected_stack.shape == image_stack.shape
    # Add more assertions to verify drift correction if applicable
