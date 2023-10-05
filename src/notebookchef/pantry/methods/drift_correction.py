#@title Create drift correction GUI
from nanopyx.methods import drift_alignment

gui_drift = EasyGui("Drift Correction")

def on_button_align(b):
    clear_output()
    gui_drift.show()
    if gui_drift["ref"].value == "First frame":
        ref_option = 0
    else:
        ref_option = 1
    avg = gui_drift["time_averaging"].value
    max_drift = gui_drift["max"].value
    global dataset_aligned
    gui_drift["align"].disabled = True
    gui_drift["align"].description = "Aligning..."
    dataset_aligned = drift_alignment.estimate_drift_alignment(dataset_original,
                                                               save_drift_table_path="",
                                                               time_averaging=avg,
                                                               max_expected_drift=max_drift,
                                                               ref_option=ref_option,
                                                               apply=True)
    if gui_drift["save"].value:
        if own_data:
            path = gui_data["upload"].selected_path
            name = gui_data["upload"].selected_filename.split(".")[0]
            tiff.imwrite(path + os.sep + name + "_aligned.tif", dataset_aligned)
        else:
            name = gui_data["data_source"].value.replace("Example dataset: ", "")
            tiff.imwrite(name + "_aligned.tif", dataset_aligned)
    gui_drift["align"].disabled = False
    gui_drift["align"].description = "Align"
    gui_drift._main_display.children = gui_drift._main_display.children + (stackview.slice(dataset_aligned, colormap=gui_drift["cmaps"].value, continuous_update=True),)

gui_drift.add_label("Drift Correction parameters:")
gui_drift.add_dropdown("ref", description="Reference frame", options=["First frame", "Previous frame"], value="First frame")
gui_drift.add_int_slider("max", description="Max expected drift", min=0, max=1000, value=10)
gui_drift.add_int_slider("time_averaging", description="Time averaging", min=1, max=dataset_original.shape[0], value=1)
gui_drift.add_dropdown("cmaps", description="Colormap:",
                      options=sorted(list(mpl.colormaps)),
                      value="viridis", remember_value=True)
gui_drift.add_checkbox("save", description="Save Output", value=True)
gui_drift.add_button("align", description="Align")
gui_drift["align"].on_click(on_button_align)
gui_drift.show()