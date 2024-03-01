from .estimator_table import DriftEstimatorTable
from ...core.utils.timeit import timeit
from ...core.transform._le_interpolation_bicubic import ShiftAndMagnify

import cv2
import numpy as np
from skimage.transform import EuclideanTransform, warp

# TODO cv2 to LE


class DriftCorrector(object):
    """
    Main class for aligning timelapse images with drift.

    This class is used for aligning timelapse images with drift correction. It requires a previously calculated drift table.
    The class implements the following methods:
    - apply_correction
    - load_estimator_table
    - _translate_slice

    Args:
        None

    Attributes:
        estimator_table (DriftEstimatorTable): An instance of the DriftEstimatorTable containing the drift table data.
        image_arr (numpy.ndarray): The timelapse image array with shape (n_slices, rows, columns).

    Methods:
        __init__(): Initialize the `DriftCorrector` object.

        _translate_slice(slice_idx): Translate an individual image slice based on the drift table.

        apply_correction(image_array): Apply drift correction to the entire image array.

        load_estimator_table(path=None): Load the drift table from a file.

    Example:
        corrector = DriftCorrector()
        corrected_image = corrector.apply_correction(image_array)
        corrector.load_estimator_table("drift_table.csv")

    Note:
        The `DriftCorrector` class is used for correcting drift in timelapse images using a precomputed drift table.
    """

    def __init__(self):
        """
        Initialize the `DriftCorrector` object.

        Args:
            None

        Returns:
            None

        Example:
            corrector = DriftCorrector()
        """
        self.estimator_table = DriftEstimatorTable()
        self.image_arr = None

    def _translate_slice(self, slice_idx):
        """
        Translate an individual image slice based on the drift table.

        Args:
            slice_idx (int): The index of the slice to be translated.

        Returns:
            numpy.ndarray: The translated image slice.

        Example:
            translated_slice = self._translate_slice(0)

        Note:
            This method is used to translate individual image slices based on the drift information in the drift table.
        """
        drift_x = self.estimator_table.drift_table[slice_idx][1]
        drift_y = self.estimator_table.drift_table[slice_idx][2]

        if drift_x == 0 and drift_y == 0:
            return self.image_arr[slice_idx]
        else:
            return cv2.warpAffine(
                self.image_arr[slice_idx].astype(np.float32),
                np.float32([[1, 0, drift_x], [0, 1, drift_y]]),
                self.image_arr[slice_idx].shape[:2][::-1],
            ).astype(self.image_arr.dtype)

    # @timeit
    def apply_correction(self, image_array):
        """
        Apply drift correction to the entire image array.

        Args:
            image_array (numpy.ndarray): The input image array with shape (n_slices, rows, columns).

        Returns:
            numpy.ndarray: The aligned image array with shape (n_slices, rows, columns).

        Example:
            corrected_image = self.apply_correction(image_array)

        Note:
            This is the main method of the `DriftCorrector` class, which applies drift correction to the entire image array.
        """
        if self.estimator_table.drift_table is not None:
            self.image_arr = image_array
            corrected_image = [self._translate_slice(i).astype(np.float32) for i in range(0, image_array.shape[0])]
            return np.array(corrected_image)
            # return np.array(translation.translate_array(image_array.astype(np.float32),
            #                                             np.array(self.estimator_table.drift_table).astype(np.float32)))

        else:
            print("Missing drift calculation")
            return None

    def load_estimator_table(self, path=None):
        """
        Load the drift table from a file.

        Args:
            path (str, optional): The path to the drift table file (CSV or NPY format). Default is None.

        Returns:
            None

        Example:
            self.load_estimator_table("drift_table.csv")

        Note:
            This method is used to load the drift table data from a file into the `estimator_table` attribute.
        """
        if path is None:
            path = input("Please provide a filepath to the drift table")

        if path.split(".")[-1] == "npy":
            self.estimator_table.import_npy(path)
        elif path.split(".")[-1] == "csv":
            self.estimator_table.import_csv(path)
