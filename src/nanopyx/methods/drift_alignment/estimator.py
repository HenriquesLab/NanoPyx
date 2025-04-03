import numpy as np
from math import sqrt
from scipy.interpolate import interp1d

from .estimator_table import DriftEstimatorTable
from .corrector import DriftCorrector
from ...core.analysis.estimate_shift import GetMaxOptimizer
from ...core.utils.timeit import timeit
from ...core.analysis.ccm import calculate_ccm
from ...core.analysis.rcc import rcc
from ...core.analysis._le_drift_calculator import (
    DriftEstimator as leDriftEstimator,
)


class DriftEstimator(object):
    """
    Drift estimator class for estimating and correcting drift in image stacks.

    This class provides methods for estimating and correcting drift in image stacks using cross-correlation.

    Args:
        None

    Attributes:
        estimator_table (DriftEstimatorTable): A table of parameters for drift estimation and correction.
        cross_correlation_map (numpy.ndarray): The cross-correlation map calculated during drift estimation.
        drift_xy (numpy.ndarray): The drift magnitude at each time point.
        drift_x (numpy.ndarray): The drift in the X direction at each time point.
        drift_y (numpy.ndarray): The drift in the Y direction at each time point.

    Methods:
        __init__(): Initialize the `DriftEstimator` object.

        estimate(image_array, **kwargs): Estimate and correct drift in an image stack.

        compute_temporal_averaging(image_arr): Compute temporal averaging of image frames.

        get_shift_from_ccm_slice(slice_index): Get the drift shift from a slice of the cross-correlation map.

        get_shifts_from_ccm(): Get the drift shifts from the entire cross-correlation map.

        create_drift_table(): Create a table of drift values.

        save_drift_table(save_as_npy=True, path=None): Save the drift table to a file.

        set_estimator_params(**kwargs): Set parameters for drift estimation and correction.

    Example:
        estimator = DriftEstimator()
        drift_params = {
            "time_averaging": 2,
            "max_expected_drift": 5,
            "shift_calc_method": "rcc",
            "ref_option": 0,
            "apply": True,
        }
        drift_corrected_image = estimator.estimate(image_stack, **drift_params)

    Note:
        The `DriftEstimator` class is used for estimating and correcting drift in image stacks.
        It provides methods for estimating drift using cross-correlation and applying drift correction to an image stack.
    """

    def __init__(self, verbose=True):
        """
        Initialize the `DriftEstimator` object.

        Args:
            None

        Returns:
            None

        Example:
            estimator = DriftEstimator()
        """
        self.verbose = verbose
        self.estimator_table = DriftEstimatorTable()
        self.cross_correlation_map = None
        self.drift_xy = None
        self.drift_x = None
        self.drift_y = None

    # @timeit
    def estimate(self, image_array, **kwargs):
        """
        Estimate and correct drift in an image stack.

        Args:
            image_array (numpy.ndarray): The input image stack with shape [n_slices, height, width].
            **kwargs: Keyword arguments for setting drift estimation parameters.

        Returns:
            numpy.ndarray or None: The drift-corrected image stack if `apply` is True, else None.

        Example:
            drift_params = {
                "time_averaging": 2,
                "max_expected_drift": 5,
                "ref_option": 0,
                "apply": True,
            }
            drift_corrected_image = estimator.estimate(image_stack, **drift_params)

        Note:
            This method estimates and corrects drift in an image stack using specified parameters.
        """
        self.set_estimator_params(**kwargs)

        n_slices = image_array.shape[0]

        # x0, y0, x1, y1 correspond to the exact coordinates of the roi to be used or full image dims and should be a tuple
        if (
            self.estimator_table.params["use_roi"]
            and self.estimator_table.params["roi"] is not None
        ):  # crops image to roi
            print(
                self.estimator_table.params["use_roi"],
                self.estimator_table.params["roi"],
            )
            x0, y0, x1, y1 = tuple(self.estimator_table.params["roi"])
            image_arr = image_array[:, y0 : y1 + 1, x0 : x1 + 1]
        else:
            image_arr = image_array

        estimator = leDriftEstimator(verbose=self.verbose)
        self.estimator_table.drift_table = estimator.run(
            np.asarray(image_arr, dtype=np.float32),
            time_averaging=self.estimator_table.params["time_averaging"],
            max_drift=self.estimator_table.params["max_expected_drift"],
            ref_option=self.estimator_table.params["ref_option"],
        )

        if self.estimator_table.params["apply"]:
            drift_corrector = DriftCorrector()
            drift_corrector.estimator_table = self.estimator_table
            tmp = drift_corrector.apply_correction(image_array)
            return tmp
        else:
            return None

    def save_drift_table(self, save_as_npy=True, path=None):
        """
        Save the drift table to a file.

        Args:
            save_as_npy (bool, optional): Whether to save the table as a NumPy binary file. Default is True.
            path (str, optional): The file path to save the table. If not provided, a user input prompt will be used.

        Returns:
            None

        Example:
            self.save_drift_table(save_as_npy=True, path="drift_table.npy")

        Note:
            This method allows saving the drift table to a file in either NumPy binary or CSV format.
        """
        if save_as_npy:
            self.estimator_table.export_npy(path=path)
        else:
            self.estimator_table.export_csv(path=path)

    def set_estimator_params(self, **kwargs):
        """
        Set parameters for drift estimation and correction.

        Args:
            **kwargs: Keyword arguments for setting drift estimation parameters.

        Returns:
            None

        Example:
            params = {
                "time_averaging": 2,
                "max_expected_drift": 5,
                "shift_calc_method": "rcc",
                "ref_option": 0,
                "apply": True,
            }
            self.set_estimator_params(**params)

        Note:
            This method allows setting parameters for drift estimation and correction.
        """
        self.estimator_table.set_params(**kwargs)
