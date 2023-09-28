import os
import numpy as np
from importlib import metadata
from datetime import datetime


class DriftEstimatorTable(object):
    """
    Class used to store DriftAlignment parameters as a dictionary.
    Parameters can be changes individually by setting the corresponding params key value to desired parameter
    """

    def __init__(self):
        self.params = {}
        self.params["lib_version"] = metadata.version("nanopyx")
        self.params["date"] = datetime.today()
        self.params["apply"] = False
        self.params["do_batch"] = False
        self.params["ref_option"] = 1  # 0 if it is to use first frame, 1 if uses the previous frame
        self.params["time_averaging"] = 1
        self.params["max_expected_drift"] = 0
        self.params["normalize"] = True
        self.params["shift_calc_method"] = "Max Fitting"
        self.params["use_roi"] = False
        self.params["roi"] = None
        self.params["show_ccm"] = True  # used for napari
        self.params["show_drift_plot"] = True  # used for napari
        self.params["show_drift_table"] = True  # used for napari
        self.params["comments"] = None

        self.drift_table = None

    def set_params(self, **kwargs):
        """
        Method used to set the parameters of drift alignment using keyword arguments.
        :param kwargs: same as self.params.keys()
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def set_comments(self, comment_string: str):
        """
        Method used to set comments for drift alignment operation
        :param comment_string: str, comment text to be added
        """
        self.params["comments"] = comment_string

    def export_npy(self, path: str = None):
        """
        Method used to export drift table as a npy file.
        :param path: Path to export drift table as npy
        """
        tmp = []
        for key in self.params.keys():
            tmp.append((key, self.params[key]))
        tmp.append(self.drift_table)
        if path is None:
            path = input("Please provide a filepath to export drift table as npy") + "_drift_table.npy"
        else:
            path = os.path.join(path, "_drift_table.npy")

        np.save(path, np.array(tmp, dtype=object))

    def import_npy(self, path: str = None):
        """
        Method used to import drift table as a npy file.
        :param path: str, Path to drift table saved as a npy file
        """
        if path is None:
            path = input("Please provide a filepath to import drift table")

        tmp = np.load(path, allow_pickle=True)

        for i in range(tmp.shape[0] - 1):
            key, value = tmp[i]
            self.params[key] = value
        self.drift_table = tmp[tmp.shape[0] - 1]

    def export_csv(self, path: str = None):
        """
        Method used to export drift table as a csv file.
        :param path: str, Path to export drift table as csv
        """
        if path is None:
            path = input("Please provide a filepath to export drift table as csv") + "_drift_table.csv"
        else:
            path = os.path.join(path, "_drift_table.csv")

        txt = ""
        for key in self.params.keys():
            txt += key + ";" + str(self.params[key]) + "\n"
        txt += "Drift Table\n"
        txt += "XY;X;Y\n"
        for i in range(self.drift_table.shape[0]):
            txt += (
                str(self.drift_table[i][0])
                + ";"
                + str(self.drift_table[i][1])
                + ";"
                + str(self.drift_table[i][2])
                + "\n"
            )

        open(path, "w").writelines(txt)

    def import_csv(self, path: str = None):
        """
        Method used to import drift table from a csv file
        :param path: str, path to import drift table as csv
        """
        if path is None:
            path = input("Please provide a filepath to import drift table")

        tmp = open(path, "r").readlines()

        count = 0
        for line in tmp:
            if line == "Drift Table\n":
                break
            else:
                count += 1
            param_split = line.split(";")
            key = param_split[0]
            value = param_split[1].split("\n")[0]
            if value == "True":
                value = True
            elif value == "False":
                value = False
            elif value == "None":
                value = None
            self.params[key] = value

        if self.params["roi"] is not None:
            roi_str_list = self.params["roi"][1:-1].split(", ")
            self.params["roi"] = tuple([int(coord) for coord in roi_str_list])

        drift_table = []

        for row in tmp[count + 2 :]:
            row_split = row.split(";")
            drift_xy = float(row_split[0])
            drift_x = float(row_split[1])
            drift_y = float(row_split[2])
            drift_table.append([drift_xy, drift_x, drift_y])

        self.drift_table = np.array(drift_table)
