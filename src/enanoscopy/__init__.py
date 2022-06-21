from .methods.drift_correction.drift_corrector import DriftCorrector
from .methods.drift_correction.drift_estimator import DriftEstimator

# this library assumes image_array is an array with shape as defined in numpy arrays (time, y, x) or (time, y, x, z)

def estimate_drift_correction(image_array, apply_correction=True, roi=None, **kwargs):
    estimator = DriftEstimator()
    estimator.estimate(image_array, roi=roi, **kwargs)
    estimator.save_drift_table(save_as_npy=False)

def apply_drift_correction(image_array, drift_table):
    corrector = DriftCorrector()
    corrector.apply_correction(image_array, drift_table)