#@title Create apply drift correction GUI
import os
import sys

NANOPYX_INSTALLED = "nanopyx" in sys.modules
if not NANOPYX_INSTALLED:
    IN_COLAB = "google.colab" in sys.modules
    if IN_COLAB:
        !pip install -q "nanopyx[colab]"
        from google.colab import output
        output.enable_custom_widget_manager()
        from google.colab import drive
        drive.mount('/content/drive')
    else:
        !pip install -q "nanopyx[jupyter]"

import stackview
import tifffile as tiff
import matplotlib as mpl
from nanopyx.core.utils.easy_gui import EasyGui
from nanopyx.methods.drift_alignment.corrector import DriftCorrector
from IPython.display import display, clear_output

gui_drift_apply = EasyGui("Apply Drift Correction")

def on_button_apply(b):
    clear_output()
    gui_drift_apply.show()

    gui_drift_apply["Align image"].disabled = True
    gui_drift_apply["Align image"].description = "Aligning..."

    drift_table_path = gui_drift_apply["upload"].selected
    img_path = gui_drift_apply["upload image"].selected

    corrector = DriftCorrector()
    corrector.load_estimator_table(drift_table_path)
    aligned_image = corrector.apply_correction(tiff.imread(img_path))

    if gui_drift_apply["save"].value:
        path = gui_drift_apply["upload image"].selected_path
        name = gui_drift_apply["upload image"].selected_filename.split(".")[0]
        save_path = path + os.sep + name + "_aligned.tif"
        tiff.imwrite(save_path, aligned_image)



    gui_drift_apply["Align image"].disabled = False
    gui_drift_apply["Align image"].description = "Align"
    gui_drift_apply._main_display.children = gui_drift_apply._main_display.children + (stackview.slice(aligned_image, colormap=gui_drift_apply["cmaps"].value, continuous_update=True),)

gui_drift_apply.add_label(value="Load drift table:")
gui_drift_apply.add_file_upload("upload")
gui_drift_apply.add_label(value="Load image to align:")
gui_drift_apply.add_file_upload("upload image")
gui_drift_apply.add_dropdown("cmaps", description="Colormap:",
                      options=sorted(list(mpl.colormaps)),
                      value="viridis", remember_value=True)
gui_drift_apply.add_checkbox("save", description="Save Output", value=True)
gui_drift_apply.add_button("Align image", description="Align image")
gui_drift_apply["Align image"].on_click(on_button_apply)
gui_drift_apply.show()