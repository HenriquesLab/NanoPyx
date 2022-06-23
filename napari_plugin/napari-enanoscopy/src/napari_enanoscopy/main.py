import os
import pathlib
from tkinter import Image
from matplotlib import use
import enanoscopy
from magicgui import magic_factory, magicgui
from napari.types import ImageData


@magic_factory(call_button="Estimate",
               img={"label": "Image Stack"},
               ref_option={"widget_type": "RadioButtons",
                           "orientation": "horizontal",
                           "value": 0,
                           "choices": [("First Frame", 0), ("Previous Frame", 1)],
                           "label": "Reference Frame"},
               time_averaging={"value": 100,
                               "label": "Time Averaging"},
               max_expected_drift={"value": 10,
                                   "label": "Max Expected Drift"},
               save_as_npy={"value": True,
                            "label": "Save Drift Table as npy"},
               use_roi={"value": False,
                        "label": "Use RoI"},
               save_drift_table_path={"label": "Save Drift Table to",
                                      "mode": "d"},
               apply_correction={"value": True,
                                 "label": "Apply"})
def estimate_drift_correction(img: ImageData, ref_option: int, time_averaging: int,
                              max_expected_drift: int, use_roi: bool,
                              save_as_npy: bool, apply_correction: bool, save_drift_table_path=pathlib.Path.home()) -> ImageData:
    return enanoscopy.estimate_drift_correction(img, save_as_npy=save_as_npy, save_drift_table_path=pathlib.Path.joinpath(save_drift_table_path, "drift_table"), ref_option=ref_option,
                                                time_averaging=time_averaging, max_expected_drift=max_expected_drift,
                                                use_roi=use_roi, apply=apply_correction)

@magic_factory(call_button="Correct",
               npy_path={"mode": "d",
                         "label": "Path to Drift Table"},)
def apply_drift_correction(img: ImageData, npy_path=pathlib.Path.home()) -> ImageData:
    return enanoscopy.apply_drift_correction(img, path=npy_path)

