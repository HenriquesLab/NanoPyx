#@title Create channel registration GUI
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
from nanopyx.methods.channel_registration import apply_channel_registration
from IPython.display import display, clear_output

gui_reg_apply = EasyGui("Channel Registration Apply")

def on_button_apply(b):
    clear_output()
    gui_reg_apply.show()

    gui_reg_apply["Register Image"].disabled = True
    gui_reg_apply["Register Image"].description = "Aligning..."

    translation_mask_path = gui_reg_apply["upload"].selected
    img_path = gui_reg_apply["upload image"].selected

    aligned_image = apply_channel_registration(tiff.imread(img_path), tiff.imread(translation_mask_path))

    if gui_reg_apply["save"].value:
        path = gui_reg_apply["upload image"].selected_path
        name = gui_reg_apply["upload image"].selected_filename.split(".")[0]
        save_path = path + os.sep + name + "_registered.tif"
        tiff.imwrite(save_path, aligned_image)



    gui_reg_apply["Register Image"].disabled = False
    gui_reg_apply["Register Image"].description = "Align"
    gui_reg_apply._main_display.children = gui_reg_apply._main_display.children + (stackview.slice(aligned_image, colormap=gui_reg_apply["cmaps"].value, continuous_update=True),)

gui_reg_apply.add_label(value="Load translation mask:")
gui_reg_apply.add_file_upload("upload")
gui_reg_apply.add_label(value="Load image to register:")
gui_reg_apply.add_file_upload("upload image")
gui_reg_apply.add_dropdown("cmaps", description="Colormap:",
                      options=sorted(list(mpl.colormaps)),
                      value="viridis", remember_value=True)
gui_reg_apply.add_checkbox("save", description="Save Output", value=True)
gui_reg_apply.add_button("Register Image", description="Register Image")
gui_reg_apply["Register Image"].on_click(on_button_apply)
gui_reg_apply.show()