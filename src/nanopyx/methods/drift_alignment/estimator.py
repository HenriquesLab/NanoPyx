import numpy as np
from math import sqrt
from scipy.interpolate import interp1d

from .estimator_table import DriftEstimatorTable
from .corrector import DriftCorrector
from ...core.analysis.estimate_shift import GetMaxOptimizer
from ...core.utils.timeit import timeit
from ...core.analysis.ccm import calculate_ccm
from ...core.analysis.rcc import rcc


class DriftEstimator(object):

    def __init__(self):
        self.estimator_table = DriftEstimatorTable()
        self.cross_correlation_map = None
        self.drift_xy = None
        self.drift_x = None
        self.drift_y = None

    # @timeit
    def estimate(self, image_array, **kwargs):
        self.set_estimator_params(**kwargs)

        n_slices = image_array.shape[0]

        # x0, y0, x1, y1 correspond to the exact coordinates of the roi to be used or full image dims and should be a tuple
        if self.estimator_table.params["use_roi"] and self.estimator_table.params["roi"] is not None:  # crops image to roi
            print(self.estimator_table.params["use_roi"], self.estimator_table.params["roi"])
            x0, y0, x1, y1 = tuple(self.estimator_table.params["roi"])
            image_arr = image_array[:, y0:y1+1, x0:x1+1]
        else:
            image_arr = image_array

        # checks time averaging, in case it's lower than 1 defaults to 1
        # if higher than n_slices/2 defaults to n_slices/2
        if self.estimator_table.params["time_averaging"] < 1:
            self.estimator_table.params["time_averaging"] = 1
        elif self.estimator_table.params["time_averaging"] > int(n_slices/2):
            self.estimator_table.params["time_averaging"] = int(n_slices/2)

        # case of no temporal averaging
        if self.estimator_table.params["time_averaging"] == 1:
            image_averages = image_arr
        else: # case of temporal averaging
            # calculates number of time blocks for averaging
            image_averages = self.compute_temporal_averaging(image_arr)

        method = self.estimator_table.params["shift_calc_method"]

        if method == "rcc":
            shifts = rcc(image_averages, max_shift=self.estimator_table.params["max_expected_drift"])
            self.drift_x = shifts[0]
            self.drift_y = shifts[1]
        else:
            self.cross_correlation_map = np.array(calculate_ccm(np.array(image_averages).astype(np.float32), self.estimator_table.params["ref_option"]))
            max_shift = self.estimator_table.params["max_expected_drift"]
            if max_shift > 0 and max_shift*2+1 < self.cross_correlation_map.shape[1] and max_shift*2+1 < self.cross_correlation_map.shape[2]:
                ccm_x_start = int(self.cross_correlation_map.shape[1]/2 - max_shift)
                ccm_y_start = int(self.cross_correlation_map.shape[0]/2 - max_shift)
                slice_ccm = self.cross_correlation_map[ccm_y_start:ccm_y_start+(max_shift*2), ccm_x_start:ccm_x_start+(max_shift*2)]
            self.get_shifts_from_ccm()

        if self.estimator_table.params["time_averaging"] > 1:

            print("Interpolating time points")
            x_idx = np.linspace(1, image_array.shape[0], num=self.drift_x.shape[0], endpoint=True, dtype=int)
            x_interpolator = interp1d(x_idx, self.drift_x, kind="cubic") # linear seems to work similar as in nanoj-core however its codebase calls setInterpolation("Bicubic")
            self.drift_x = x_interpolator(range(1, image_array.shape[0] + 1))
            y_idx = np.linspace(1, image_array.shape[0], num=self.drift_y.shape[0], endpoint=True, dtype=int)
            y_interpolator = interp1d(y_idx, self.drift_y, kind="cubic") # linear seems to work similar as in nanoj-core however its codebase calls setInterpolation("Bicubic")
            self.drift_y = y_interpolator(range(1, image_array.shape[0] + 1))

        self.drift_xy = []
        for i in range(image_array.shape[0]): 
            self.drift_xy.append(sqrt(pow(self.drift_x[i], 2) + pow(self.drift_y[i], 2)))
        self.drift_xy = np.array(self.drift_xy)

        self.create_drift_table()

        if self.estimator_table.params["apply"]:
            drift_corrector = DriftCorrector()
            drift_corrector.estimator_table = self.estimator_table
            tmp = drift_corrector.apply_correction(image_array)
            return tmp
        else:
            return None

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

    def get_shift_from_ccm_slice(self, slice_index):
        slice_ccm = self.cross_correlation_map[slice_index]

        w = slice_ccm.shape[1]
        h = slice_ccm.shape[0]

        radius_x = w / 2.0
        radius_y = h / 2.0

        method = self.estimator_table.params["shift_calc_method"]

        if method == "Max Fitting":
            optimizer = GetMaxOptimizer(slice_ccm)
            shift_y, shift_x = optimizer.get_max()
        elif method == "Max":
            shift_y, shift_x = np.unravel_index(slice_ccm.argmax(), slice_ccm.shape)

        shift_x = round(radius_x - shift_x - 0.5, 3)
        shift_y = round(radius_y - shift_y - 0.5, 3)

        return (shift_x, shift_y)

    def get_shifts_from_ccm(self):

        drift_x = []
        drift_y = []
        drift = []

        for i in range(self.cross_correlation_map.shape[0]):
            drift.append(self.get_shift_from_ccm_slice(i))
        
        drift = np.array(drift)
        drift_x = drift[:, 0]
        drift_y = drift[:, 1]

        bias_x = drift_x[0]
        bias_y = drift_y[0]

        self.drift_x = np.zeros((drift_x.shape[0]))
        self.drift_y = np.zeros((drift_y.shape[0]))

        for i in range(0, self.cross_correlation_map.shape[0]):
            self.drift_x[i] = drift_x[i] - bias_x
            self.drift_y[i] = drift_y[i] - bias_y
            if self.estimator_table.params["ref_option"] == 1 and i > 0:
                self.drift_x[i] += self.drift_x[i-1]
                self.drift_y[i] += self.drift_y[i-1]

        self.drift_x = np.array(self.drift_x)
        self.drift_y = np.array(self.drift_y)

    def create_drift_table(self):
        table = []
        for i in range(0, self.drift_xy.shape[0]):
            table.append([self.drift_xy[i], self.drift_x[i], self.drift_y[i]])
        table = np.array(table)
        self.estimator_table.drift_table = table

    def save_drift_table(self, save_as_npy=True, path=None):
        if save_as_npy:
            self.estimator_table.export_npy(path=path)
        else:
            self.estimator_table.export_csv(path=path)

    def set_estimator_params(self, **kwargs):
        self.estimator_table.set_params(**kwargs)

