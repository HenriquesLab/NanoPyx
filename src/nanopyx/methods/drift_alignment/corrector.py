from .estimator_table import DriftEstimatorTable

import numpy as np
from skimage.transform import EuclideanTransform, warp

class DriftCorrector(object):
    def __init__(self):
        self.estimator_table = DriftEstimatorTable()
        self.image_arr = None

    def _translate_slice(self, slice_idx):
        drift_x = self.estimator_table.drift_table[slice_idx][1]
        drift_y = self.estimator_table.drift_table[slice_idx][2]
        transformation_matrix = EuclideanTransform(rotation=0, translation=(drift_y, drift_x))
        return warp(self.image_arr[slice_idx], transformation_matrix.inverse, order=3, preserve_range=True)

    def apply_correction(self, image_array):
        if self.estimator_table.drift_table is not None:
            self.image_arr = image_array

            corrected_image = [self._translate_slice(i) for i in range(0, image_array.shape[0])]

            return np.array(corrected_image).reshape((self.image_arr.shape[0], self.image_arr.shape[1], self.image_arr.shape[2]))

        else:
            print("Missing drift calculation")
            return None

    def load_drift_table(self, path=None):
        if path is None:
            path = input("Please provide a filepath to the drift table")

        if path.split(".")[-1] == "npy":
            self.estimator_table.import_npy(path)
        elif path.split(".")[-1] == "csv":
            self.estimator_table.import_csv(path)

