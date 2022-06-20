from .methods.drift_correction.drift_corrector import DriftCorrector
from .methods.drift_correction.drift_estimator import DriftEstimator

# this library assumes image_array is an array with shape as defined in numpy arrays (time, y, x) or (time, y, x, z)

def estimate_drift_correction(image_array, apply_correction=True, roi=None, **kwargs):
    estimator = DriftEstimator()
    return estimator.estimate(image_array, roi=roi, **kwargs)
    #estimator.save_drift_table()

    #if apply_correction and estimator.estimator_table.drift_table is not None:
    #   apply_drift_correction(image_array, estimator.estimator_table.drift_table)

def apply_drift_correction(image_array, drift_table):
    corrector = DriftCorrector()
    corrector.apply_correction(image_array, drift_table)