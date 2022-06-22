from tkinter import Image
from matplotlib import use
import enanoscopy
from magicgui import magic_factory, magicgui
from napari.types import ImageData


@magic_factory(call_button="Estimate",
               ref_option={"widget_type": "RadioButtons",
                           "orientation": "horizontal",
                           "value": 0,
                           "choices": [("First Frame", 0), ("Previous Frame", 1)]},
               time_averaging={"value": 100},
               max_expected_drift={"value": 10},
               save_as_npy={"value": True},
               use_roi={"value": False})
def estimate_drift_correction(img: ImageData, ref_option: int, time_averaging: int,
                              max_expected_drift: int, use_roi: bool,
                              save_as_npy: bool) -> Image:
    return enanoscopy.estimate_drift_correction(img, save_as_npy=save_as_npy, ref_option=ref_option,
                                                time_averaging=time_averaging, max_expected_drift=max_expected_drift,
                                                use_roi=use_roi)

@magic_factory(call_button="Correct",
               path={"mode": "r"})
def apply_drift_correction(img: ImageData, path: str) -> ImageData:
    return enanoscopy.apply_drift_correction(img, path=path)

