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
    Main class implementing 3d drift correction.
    Requires an image array with shape (t, z, y, x)
    """

    def __init__(self):
        self.image_array = None
        self.xy_estimator = None
        self.z_estimator = None

    def correct_xy_drift(self, projection_mode="Mean", **kwargs):

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
        
        self.image_array = image_array
        self.correct_xy_drift(projection_mode=projection_mode, **kwargs)
        self.correct_z_drift(axis_mode=axis_mode, projection_mode=projection_mode, **kwargs)
        return self.image_array
    
