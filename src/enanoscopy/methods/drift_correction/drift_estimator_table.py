from importlib import metadata
from datetime import datetime


class DriftEstimatorTable(object):

    def __init__(self):
        self.params = {}
        self.params["lib_version"] = metadata.version("enanoscopy")
        self.params["date"] = datetime.today()
        self.params["apply"] = False
        self.params["do_batch"] = False
        self.params["ref_option"] = 0 # 0 if it is to use first frame, 1 if uses the previous frame
        self.params["time_averaging"] = 100
        self.params["max_expected_drift"] = 10
        self.params["reference_frame"] = 0
        self.params["normalize"] = True
        self.params["use_roi"] = False
        self.params["roi"] = None
        self.params["show_ccm"] = True
        self.params["show_drift_plot"] = True
        self.params["show_drift_table"] = True
        self.params["comments"] = None

        self.drift_table = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def set_comments(self, comment_string):
        self.params["comments"] = comment_string

    def export_npy(self):
        pass

    def export_csv(self):
        pass

