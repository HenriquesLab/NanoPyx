from .drift_estimator_table import DriftEstimatorTable

import numpy as np

class DriftCorrector(object):
    def __init__(self):
        self.estimator_table = DriftEstimatorTable()

    def apply_correction(self, image_array, drift_table):
        pass

    def load_drift_table(self):
        pass

