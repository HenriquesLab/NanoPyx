import numpy as np

from .drift_estimator_table import DriftEstimatorTable
from ..image.transform.cross_correlation_map import CrossCorrelationMap


class DriftEstimator(object):
    def __init__(self):
        self.estimator_table = DriftEstimatorTable()
        self.crosscorrelation_map = None
        self.drift_plot_XY = None
        self.drift_plot_X = None
        self.drift_plot_Y = None

    def estimate(self, image_array, **kwargs):
        self.set_estimator_params(**kwargs)

        n_slices = image_array.shape[0]

        # x0, y0, x1, y1 correspond to the exact coordinates of the roi to be used or full image dims
        if not self.estimator_table.params["use_roi"]:
            x0, y0, x1, y1 = 0, 0, image_array.shape[2]-1, image_array.shape[1]-1
            image_arr = image_array
        elif self.estimator_table.params["use_roi"] & self.estimator_table.params["roi"] is not None: # crops image to roi
            x0, y0, x1, y1 = self.estimator_table.params["roi"]
            image_arr = image_array[:, y0:y1+1, x0:x1+1]

        # checks time averaging, in case it's lower than 1 defaults to 1
        # if higher than n_slices/2 defaults to n_slices/2
        if self.estimator_table.params["time_averaging"] < 1:
            self.estimator_table.params["time_averaging"] = 1
        elif self.estimator_table.params["time_averaging"] > int(n_slices/2):
            self.estimator_table.params["time_averaging"] =  int(n_slices/2)

        # case of no temporal averaging
        if self.estimator_table.params["time_averaging"] == 1:
            image_averages = image_arr
        else: # case of temporal averaging
            # calculates number of time blocks for averaging
            image_averages = self.compute_temporal_averaging(image_arr)

        # perform cross correlation map calculation
        if self.estimator_table.params["reference_frame"] == 0:
            img_ref = image_arr[0, y0:y1+1, x0:x1+1]
        else:
            img_ref = None

        image_ccm = self.calculate_cross_correlation_map(img_ref, image_averages, normalize=self.estimator_table.params["normalize"])
        return image_ccm


    def compute_temporal_averaging(self, image_arr):
        n_slices = image_arr.shape[0]

        if self.estimator_table.params["use_roi"]:
            x0, y0, x1, y1 = self.estimator_table.params["roi"]
        else:
            x0, y0, x1, y1 = 0, 0, image_arr.shape[2]-1, image_arr.shape[1]-1

        n_blocks = int(n_slices / self.estimator_table.params["time_averaging"])
        if (n_slices % self.estimator_table.params["time_averaging"]) != 0:
            n_blocks += 1
        image_averages = np.zeros((n_blocks, y1+1-y0, x1+1-x0))            
        for i in range(n_blocks):
            t_start = i * self.estimator_table.params["time_averaging"]
            t_stop = (i + 1) * self.estimator_table.params["time_averaging"]
            image_averages[i] = np.mean(image_arr[t_start:t_stop, :, :], axis=0)
        return image_averages

    def calculate_cross_correlation_map(self, img_ref, img_stack, normalize=True):
        ccm_calculator = CrossCorrelationMap()
        ccm = ccm_calculator.calculate_ccm(img_ref, img_stack, normalize)

        max_drift = self.estimator_table.params["max_expected_drift"]
        if max_drift != 0 and max_drift*2+1 < min(ccm.shape[1], ccm.shape[2]):
            x_start = int(ccm.shape[2]/2 - max_drift)
            y_start = int(ccm.shape[1]/2 - max_drift)
            ccm = ccm[:, y_start:y_start+max_drift*2+1, x_start:x_start+max_drift*2+1]

        return ccm

    def get_shift_from_ccm(self):
        pass

    def create_drift_table(self):
        pass

    def save_drift_table(self):
        pass

    def set_estimator_params(self, **kwargs):
        self.estimator_table.set_params(**kwargs)

