import numpy as np
from math import sqrt
from scipy.interpolate import interp1d

from .estimator_table import DriftEstimatorTable
from .corrector import DriftCorrector
from .estimator import DriftEstimator
from ...core.analysis.estimate_shift import GetMaxOptimizer
from ...core.utils.timeit import timeit
from ...core.analysis.ccm import calculate_ccm


class Estimator3D(object):
    """
    Main class implementing 3D drift correction.

    This class provides methods for performing 3D drift correction on an image array with shape (t, z, y, x).
    It corrects both XY drift and Z drift using projection methods.

    Args:
        None

    Attributes:
        image_array (numpy.ndarray): The input 3D image stack with shape (t, z, y, x).
        xy_estimator (DriftEstimator): A drift estimator for XY drift correction.
        z_estimator (DriftEstimator): A drift estimator for Z drift correction.

    Methods:
        __init__(): Initialize the `Estimator3D` object.

        correct_xy_drift(projection_mode="Mean", **kwargs): Correct XY drift in the image stack using projection.

        correct_z_drift(axis_mode="top", projection_mode="Mean", **kwargs): Correct Z drift in the image stack using projection.

        correct_3d_drift(image_array, axis_mode="top", projection_mode="Mean", **kwargs): Correct both XY and Z drift in the 3D image stack.

    Example:
        estimator = Estimator3D()
        corrected_image_stack = estimator.correct_3d_drift(image_stack, axis_mode="top", projection_mode="Mean", **drift_params)

    Note:
        The `Estimator3D` class is used for performing 3D drift correction on image stacks.
        It includes methods for correcting XY drift and Z drift separately or together.
    """

    def __init__(self):
        """
        Initialize the `Estimator3D` object.

        Args:
            None

        Returns:
            None

        Example:
            estimator = Estimator3D()
        """
        self.image_array = None
        self.xy_estimator = None
        self.z_estimator = None

    def correct_xy_drift(self, projection_mode="Mean", **kwargs):
        """
        Correct XY drift in the image stack using projection.

        Args:
            projection_mode (str, optional): The projection mode for drift correction. "Mean" or "Max" can be used. Default is "Mean".
            **kwargs: Keyword arguments for drift estimation parameters.

        Returns:
            None

        Example:
            estimator.correct_xy_drift(projection_mode="Mean", **drift_params)

        Note:
            This method corrects XY drift in the 3D image stack using projection-based drift estimation and correction.
        """
        self.xy_estimator = DriftEstimator()

        if projection_mode == "Mean":
            projection = np.mean(self.image_array, axis=1)
        elif projection_mode == "Max":
            projection = np.max(self.image_array, axis=1)
        else:
            print("Not a valid projection mode")
            return None

        self.xy_estimator.estimate(projection, apply=False, **kwargs)

        corrector = DriftCorrector()
        corrector.estimator_table = self.xy_estimator.estimator_table
        for i in range(self.image_array.shape[1]):
            self.image_array[:, i, :, :] = corrector.apply_correction(self.image_array[:, i, :, :])

    def correct_z_drift(self, axis_mode="top", projection_mode="Mean", **kwargs):
        """
        Correct Z drift in the image stack using projection.

        Args:
            axis_mode (str, optional): The axis mode for Z drift correction. "top" or "left" can be used. Default is "top".
            projection_mode (str, optional): The projection mode for drift correction. "Mean" or "Max" can be used. Default is "Mean".
            **kwargs: Keyword arguments for drift estimation parameters.

        Returns:
            None

        Example:
            estimator.correct_z_drift(axis_mode="top", projection_mode="Mean", **drift_params)

        Note:
            This method corrects Z drift in the 3D image stack using projection-based drift estimation and correction.
        """
        if axis_mode == "top":
            axis_idx = 2
        elif axis_mode == "left":
            axis_idx = 3
        else:
            print("Not a valid axis mode")
            return None

        self.z_estimator = DriftEstimator()
        if projection_mode == "Mean":
            projection = np.mean(self.image_array, axis=axis_idx)
        elif projection_mode == "Max":
            projection = np.max(self.image_array, axis=axis_idx)
        else:
            print("Not a valid projection mode")
            return None

        self.z_estimator.estimate(projection, apply=False, **kwargs)

        corrector = DriftCorrector()
        print(self.image_array.shape, projection.shape)
        corrector.estimator_table = self.z_estimator.estimator_table
        if axis_mode == "top":
            corrector.estimator_table.drift_table[:, 1] = 0
            for i in range(self.image_array.shape[axis_idx]):
                self.image_array[:, :, i, :] = corrector.apply_correction(self.image_array[:, :, i, :])
        elif axis_mode == "left":
            corrector.estimator_table.drift_table[:, 2] = 0
            for i in range(self.image_array.shape[axis_idx]):
                self.image_array[:, :, :, i] = corrector.apply_correction(self.image_array[:, :, :, i])

    def correct_3d_drift(self, image_array, axis_mode="top", projection_mode="Mean", **kwargs):
        """
        Correct both XY and Z drift in the 3D image stack.

        Args:
            image_array (numpy.ndarray): The input 3D image stack with shape (t, z, y, x).
            axis_mode (str, optional): The axis mode for Z drift correction. "top" or "left" can be used. Default is "top".
            projection_mode (str, optional): The projection mode for drift correction. "Mean" or "Max" can be used. Default is "Mean".
            **kwargs: Keyword arguments for drift estimation parameters.

        Returns:
            numpy.ndarray: The drift-corrected 3D image stack.

        Example:
            corrected_image_stack = estimator.correct_3d_drift(image_stack, axis_mode="top", projection_mode="Mean", **drift_params)

        Note:
            This method performs both XY and Z drift correction on the 3D image stack and returns the corrected stack.
        """
        self.image_array = image_array
        self.correct_xy_drift(projection_mode=projection_mode, **kwargs)
        self.correct_z_drift(axis_mode=axis_mode, projection_mode=projection_mode, **kwargs)
        return self.image_array
