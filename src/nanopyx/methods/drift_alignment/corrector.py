from .estimator_table import DriftEstimatorTable
from ...core.utils.timeit import timeit
from ...core.transform import translation
from ...core.transform.image_shift import bicubic_shift

import numpy as np
from skimage.transform import EuclideanTransform, warp


class DriftCorrector(object):
    """
    Main class for aligning timelapse images with drift.
    Required previous calculation of a drift table.
    Implements the following methods:
    - apply_correction
    - load_drift_table
    - _translate_slice
    """
    def __init__(self):
        self.estimator_table = DriftEstimatorTable()
        self.image_arr = None

    def _translate_slice(self, slice_idx):
        """
        Method used to translate individual slices.
        Requires previous loading of the timelapse image array with shape (n_slices, rows, columns) and drift table.
        Takes a single index and calculates the translated slice for that index.
        :param slice_idx: int corresponding to the slice to be translated
        :return: translated image slice
        """
        drift_x = self.estimator_table.drift_table[slice_idx][1]
        drift_y = self.estimator_table.drift_table[slice_idx][2]

        if drift_x == 0 and drift_y == 0:
            return self.image_arr[slice_idx]
        else:
            transformation_matrix = EuclideanTransform(rotation=0, translation=(drift_y, drift_x))
            return warp(self.image_arr[slice_idx], transformation_matrix.inverse, mode="edge", order=3, preserve_range=True)

    # @timeit
    def apply_correction(self, image_array):
        """
        Main method of DriftCorrector class.
        Translates each image slice according to the drift table.
        :param image_array: numpy array with shape (n_slices, rows, columns)
        :return: aligned image array with shape (n_slices, rows, columns)
        """
        if self.estimator_table.drift_table is not None:
            # self.image_arr = image_array
            # corrected_image = [self._translate_slice(i) for i in range(0, image_array.shape[0])]
            # return np.array(corrected_image)
            return np.array(translation.translate_array(image_array.astype(np.float32),
                                                        np.array(self.estimator_table.drift_table).astype(np.float32)))

        else:
            print("Missing drift calculation")
            return None

    def load_estimator_table(self, path=None):
        """
        Method used to load the drift table.
        :param path: path to a .csv or .npy drift table
        :return: None, stores drift table data in self.drift_table
        """
        if path is None:
            path = input("Please provide a filepath to the drift table")

        if path.split(".")[-1] == "npy":
            self.estimator_table.import_npy(path)
        elif path.split(".")[-1] == "csv":
            self.estimator_table.import_csv(path)
