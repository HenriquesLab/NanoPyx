import numpy as np
from importlib import metadata
from datetime import datetime
from tkinter import filedialog as fd


class DriftEstimatorTable(object):

    def __init__(self):
        self.params = {}
        self.params["lib_version"] = metadata.version("NanoPyx-Core-0.1")
        self.params["date"] = datetime.today()
        self.params["apply"] = False
        self.params["do_batch"] = False
        self.params["ref_option"] = 0 # 0 if it is to use first frame, 1 if uses the previous frame
        self.params["time_averaging"] = 100
        self.params["max_expected_drift"] = 10
        self.params["normalize"] = True
        self.params["shift_calc_method"] = "Max Fitting"
        self.params["use_roi"] = False
        self.params["roi"] = None
        self.params["show_ccm"] = True # used for napari
        self.params["show_drift_plot"] = True # used for napari
        self.params["show_drift_table"] = True # used for napari
        self.params["comments"] = None

        self.drift_table = None

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def set_comments(self, comment_string):
        self.params["comments"] = comment_string

    def export_npy(self, path=None):
        tmp = []
        for key in self.params.keys():
            tmp.append((key, self.params[key]))
        tmp.append(self.drift_table)
        if path is None:
            filepath = fd.asksaveasfilename(title="Save Drift Table as npy")
            np.save(filepath, np.array(tmp, dtype=object))
        else:
            np.save(path, np.array(tmp, dtype=object))

    def import_npy(self, path=None):
        if path is None:
            filepath = fd.askopenfilename(title="Load npy Drift Table")
        else:
            filepath = path
        tmp = np.load(filepath, allow_pickle=True)

        for i in range(tmp.shape[0]-1):
            key, value = tmp[i]
            self.params[key] = value
        self.drift_table = tmp[tmp.shape[0]-1]

    def export_csv(self, path=None):
        if path is None:
            filepath = fd.asksaveasfilename(title="Save Drift Table as csv")
        else:
            filepath = path

        txt = ""
        for key in self.params.keys():
            txt += key + ";" + str(self.params[key]) + "\n"
        txt += "Drift Table\n"
        txt += "XY;X;Y\n"
        for i in range(self.drift_table.shape[0]):
            txt += str(self.drift_table[i][0]) + ";" + str(self.drift_table[i][1]) + ";" + str(self.drift_table[i][2]) + "\n"

        open(filepath + ".csv", "w").writelines(txt)

    def import_csv(self, path=None):
        if path is None:
            filepath = fd.asksaveasfilename(title="Load Drift Table as csv")
        else:
            filepath = path

        tmp = open(filepath, "r").readlines()

        n_params = len(self.params.keys())

        for i in range(n_params):
            param_split = tmp[i].split(";")
            key = param_split[0]
            value = param_split[1].split("\n")[0]
            if value == "TRUE":
                value = True
            elif value == "FALSE":
                value = False
            elif value == "None":
                value = None
            self.params[key] = value

        if self.params["roi"] is not None:
            roi_str_list = self.params["roi"][1:-1].split(", ")
            self.params["roi"] = tuple([int(coord) for coord in roi_str_list])

        drift_table = []

        for row in tmp[n_params+2:]:
            row_split = row.split(";")
            drift_xy = float(row_split[0])
            drift_x = float(row_split[1])
            drift_y = float(row_split[2])
            drift_table.append([drift_xy, drift_x, drift_y])

        self.drift_table = np.array(drift_table)
