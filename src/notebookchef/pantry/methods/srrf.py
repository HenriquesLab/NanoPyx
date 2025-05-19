# @title Create SRRF GUI
gui_srrf = EasyGui("srrf")
from nanopyx.methods import SRRF
from nanopyx.core.transform.sr_temporal_correlations import (
    calculate_SRRF_temporal_correlations,
)


def run_srrf(b):
    clear_output()
    gui_srrf.show()
    gui_srrf.save_settings()
    ring_radius = gui_srrf["ring_radius"].value
    magnification = gui_srrf["magnification"].value
    frames_per_timepoint = gui_srrf["frames_per_timepoint"].value
    srrf_order = gui_srrf["srrf_order"].value
    mpcorrection = gui_srrf["mpcorrection"].value
    # disable button while running
    gui_srrf["run"].disabled = True
    gui_srrf["run"].description = "Running..."
    if frames_per_timepoint == 0:
        frames_per_timepoint = dataset_original.shape[0]
    elif frames_per_timepoint > dataset_original.shape[0]:
        frames_per_timepoint = dataset_original.shape[0]

    output = []

    for i in range(dataset_original.shape[0] // frames_per_timepoint):
        block = dataset_original[
            i * frames_per_timepoint : (i + 1) * frames_per_timepoint
        ]
        result = SRRF(
            block,
            magnification=magnification,
            ringRadius=ring_radius,
            radialityPositivityConstraint=True,
            doIntensityWeighting=True,
            macro_pixel_correction=mpcorrection,
        )
        output.append(calculate_SRRF_temporal_correlations(result, srrf_order))

    global dataset_srrf
    dataset_srrf = np.array(output)
    # enable button again
    gui_srrf["run"].disabled = False
    gui_srrf["run"].description = "Run"
    if gui_srrf["save"].value:
        if own_data:
            path = gui_data["upload"].selected_path
            name = gui_data["upload"].selected_filename.split(".")[0]
            tiff.imwrite(path + os.sep + name + "_srrf.tif", dataset_srrf)
        else:
            name = gui_data["data_source"].value.replace(
                "Example dataset: ", ""
            )
            tiff.imwrite(name + "_srrf.tif", dataset_srrf)
    gui_srrf._main_display.children = gui_srrf._main_display.children + (
        stackview.slice(
            dataset_srrf,
            colormap=gui_srrf["cmaps"].value,
            continuous_update=True,
        ),
    )


gui_srrf.add_float_slider(
    "ring_radius", description="Ring Radius:", min=0.1, max=3.0, value=0.5
)
gui_srrf.add_int_slider(
    "magnification", description="Magnification:", min=1, max=10, value=5
)
gui_srrf.add_int_slider(
    "srrf_order", description="SRRF order:", min=-1, max=4, value=3
)
gui_srrf.add_label(value="-=-= Time-Lapse =-=-")
gui_srrf.add_int_slider(
    "frames_per_timepoint",
    description="Frames per time-point (0 - auto)",
    min=1,
    max=dataset_original.shape[0],
    value=dataset_original.shape[0] // 2,
)
gui_srrf.add_checkbox(
    "mpcorrection", description="Macro Pixel Correction", value=True
)
gui_srrf.add_checkbox("save", description="Save Output", value=True)
gui_srrf.add_dropdown(
    "cmaps",
    description="Colormap:",
    options=sorted(list(mpl.colormaps)),
    value="viridis",
    remember_value=True,
)
gui_srrf.add_button("run", description="Run")
gui_srrf["run"].on_click(run_srrf)
gui_srrf.show()
