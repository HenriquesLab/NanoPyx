import os
import pathlib
import platform
import multiprocessing as mp
from tkinter import Image
import enanoscopy
from magicgui import magic_factory, magicgui
from napari.types import ImageData
from napari.layers import Image
from napari import Viewer
from napari.qt.threading import thread_worker


@magic_factory(call_button="Estimate",
               img={"label": "Image Stack"},
               ref_option={"widget_type": "RadioButtons",
                           "orientation": "vertical",
                           "value": 0,
                           "choices": [("First Frame", 0), ("Previous Frame", 1)],
                           "label": "Reference Frame"},
               shift_calc_method={"widget_type": "RadioButtons",
                                  "orientation": "vertical",
                                  "value": "Max Fitting",
                                  "choices": [("Max", "Max"), ("Subpixel Fitting", "Max Fitting")],
                                  "label": "Shift Calculation Method"},
               time_averaging={"value": 100,
                               "label": "Time Averaging"},
               max_expected_drift={"value": 10,
                                   "label": "Max Expected Drift"},
               save_as_npy={"value": False,
                            "label": "Save Drift Table as npy"},
               use_roi={"value": False,
                        "label": "Use RoI"},
               save_drift_table_path={"label": "Save Drift Table to",
                                      "mode": "w"},
               apply_correction={"value": True,
                                 "label": "Apply"})
def estimate_drift_correction(viewer: Viewer, img: Image, ref_option: int, time_averaging: int,
                              max_expected_drift: int, use_roi: bool, shift_calc_method: str,
                              save_as_npy: bool, apply_correction: bool, save_drift_table_path=pathlib.Path.home()) -> ImageData:

    result = enanoscopy.estimate_drift_correction(img.data, save_as_npy=save_as_npy, save_drift_table_path=str(save_drift_table_path), ref_option=ref_option,
                                                  time_averaging=time_averaging, max_expected_drift=max_expected_drift, shift_calc_method=shift_calc_method,
                                                  use_roi=use_roi, apply=apply_correction)

    if result is not None:
        result_name = img.name + "_aligned"
        try:
            # if the layer exists, update the data
            viewer.layers[result_name].data = result
        except KeyError:
            # otherwise add it to the viewer
            viewer.add_image(result, name=result_name)


@magic_factory(call_button="Correct",
               drift_table_path={"mode": "r",
                                 "label": "Path to Drift Table"})
def apply_drift_correction(viewer: Viewer, img: Image, drift_table_path=pathlib.Path.home()) -> ImageData:
    result = enanoscopy.apply_drift_correction(img.data, path=str(drift_table_path))
    if result is not None:
        result_name = img.name + "_aligned"
        try:
            # if the layer exists, update the data
            viewer.layers[result_name].data = result
        except KeyError:
            # otherwise add it to the viewer
            viewer.add_image(result, name=result_name)

