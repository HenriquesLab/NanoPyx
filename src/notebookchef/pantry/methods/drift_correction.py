# @title Create drift correction GUI
from nanopyx.methods.drift_alignment.estimator import DriftEstimator

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

    estimator = DriftEstimator()
    dataset_aligned = estimator.estimate(
        dataset_original,
        time_averaging=avg,
        max_expected_drift=max_drift,
        ref_option=ref_option,
        apply=True,
    )

    if gui_drift["save"].value:
        if own_data:
            path = gui_data["upload"].selected_path
            name = gui_data["upload"].selected_filename.split(".")[0]
            drift_table_path = path + os.sep + name + "_drift_table.csv"
            save_path = path + os.sep + name + "_aligned.tif"
        else:
            path = ""
            name = gui_data["data_source"].value.replace(
                "Example dataset: ", ""
            )
            drift_table_path = path + os.sep + name + "_drift_table.csv"
            save_path = name + "_aligned.tif"
        tiff.imwrite(save_path, dataset_aligned)
        txt = ""
        for key in estimator.estimator_table.params.keys():
            txt += (
                key + ";" + str(estimator.estimator_table.params[key]) + "\n"
            )
        txt += "Drift Table\n"
        txt += "XY;X;Y\n"
        for i in range(estimator.estimator_table.drift_table.shape[0]):
            txt += (
                str(estimator.estimator_table.drift_table[i][0])
                + ";"
                + str(estimator.estimator_table.drift_table[i][1])
                + ";"
                + str(estimator.estimator_table.drift_table[i][2])
                + "\n"
            )
        open(drift_table_path, "w").writelines(txt)
    gui_drift["align"].disabled = False
    gui_drift["align"].description = "Align"
    gui_drift._main_display.children = gui_drift._main_display.children + (
        stackview.slice(
            dataset_aligned,
            colormap=gui_drift["cmaps"].value,
            continuous_update=True,
        ),
    )


gui_drift.add_label(value="Drift Correction parameters:")
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
