from .corrector import DriftCorrector
from .estimator import DriftEstimator
from ...core.utils.timeit import timeit


def estimate_drift_alignment(image_array, save_as_npy=True, save_drift_table_path=None, roi=None, **kwargs):
    """
    Function use to estimate the drift in a microscopy image.
    :param image_array: numpy array  with shape (z, y, x)
    :param save_as_npy (optional): bool, whether to save as npy (if true) or csv (if false)
    :param save_drift_table_path (optional): str, path to save drift table
    :param roi (optional): in case of use should have shape (x0, y0, x1, y1)
    :param kwargs: additional keyword arguments
    :return: aligned image as numpy array
    """
    estimator = DriftEstimator()
    corrected_img = estimator.estimate(image_array, roi=roi, **kwargs)
    print(save_drift_table_path)
    estimator.save_drift_table(save_as_npy=save_as_npy, path=save_drift_table_path)
    if corrected_img is not None:
        return corrected_img
    else:
        pass


def apply_drift_alignment(image_array, path=None, drift_table=None):
    """
    Function used to correct the drift in a microscopy image given a previously calculated drift table.
    :param image_array: numpy array  with shape (z, y, x); image to be corrected
    :param path (optional): str; path to previously saved
    :param drift_table (optional): estimator table object; object containing previously calculated drift table
    :return: aligned image as numpy array
    """
    corrector = DriftCorrector()
    if drift_table is None:
        corrector.load_estimator_table(path=path)
    else:
        corrector.estimator_table = drift_table
    corrected_img = corrector.apply_correction(image_array)
    return corrected_img
