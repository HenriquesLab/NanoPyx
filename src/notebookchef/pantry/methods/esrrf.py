#@title Create eSRRF GUI
gui_esrrf = EasyGui("esrrf")
from nanopyx.methods import eSRRF
from nanopyx.core.transform.sr_temporal_correlations import calculate_eSRRF_temporal_correlations

def run_esrrf(b):
    clear_output()
    gui_esrrf.show()
    gui_esrrf.save_settings()
    ring_radius = gui_esrrf["ring_radius"].value
    magnification = gui_esrrf["magnification"].value
    frames_per_timepoint = gui_esrrf["frames_per_timepoint"].value
    sensitivity = gui_esrrf["sensitivity"].value

    esrrf_order = gui_esrrf["esrrf_order"].value
    if esrrf_order == 1:
        esrrf_order = "AVG"
    elif esrrf_order == 2:
        esrrf_order = "VAR"
    elif esrrf_order == 3:
        esrrf_order = "TAC2"
    # disable button while running
    gui_esrrf["run"].disabled = True
    gui_esrrf["run"].description = "Running..."
    if frames_per_timepoint == 0:
        frames_per_timepoint = dataset_original.shape[0]
    elif frames_per_timepoint > dataset_original.shape[0]:
        frames_per_timepoint = dataset_original.shape[0]

    output= []

    for i in range(dataset_original.shape[0] // frames_per_timepoint):
        block = dataset_original[i*frames_per_timepoint:(i+1)*frames_per_timepoint]
        result = eSRRF(block, magnification=magnification, radius=ring_radius,
                                sensitivity=sensitivity,
                                doIntensityWeighting=True)
        output.append(calculate_eSRRF_temporal_correlations(result[0], esrrf_order))

    global dataset_esrrf
    dataset_esrrf = np.array(output)
    # enable button again
    gui_esrrf["run"].disabled = False
    gui_esrrf["run"].description = "Run"
    if gui_esrrf["save"].value:
        if own_data:
            path = gui_data["upload"].selected_path
            name = gui_data["upload"].selected_filename.split(".")[0]
            tiff.imwrite(path + os.sep + name + "_esrrf.tif", dataset_esrrf)
        else:
            name = gui_data["data_source"].value.replace("Example dataset: ", "")
            tiff.imwrite(name + "_esrrf.tif", dataset_esrrf)
    gui_esrrf._main_display.children = gui_esrrf._main_display.children + (stackview.slice(dataset_esrrf, colormap=gui_esrrf["cmaps"].value, continuous_update=True),)


default_radius = 1.5
default_sensitivity = 1
default_magnification = 5
default_esrrf_order = 1
gui_esrrf.add_float_slider("ring_radius", description="Ring Radius:", min=0.1, max=3.0, value=default_radius, remember_value=True)
gui_esrrf.add_int_slider("sensitivity", description="Sensitivity:", min=1, max=10, value=default_sensitivity)
gui_esrrf.add_int_slider("magnification", description="Magnification:", min=1, max=10, value=default_magnification)
gui_esrrf.add_int_slider("esrrf_order", description="eSRRF order:", min=1, max=3, value=default_esrrf_order)
gui_esrrf.add_label("-=-= Time-Lapse =-=-")
gui_esrrf.add_int_slider("frames_per_timepoint", description="Frames per time-point (0 - auto)", min=0, max=dataset_original.shape[0], value=dataset_original.shape[0]//2)
gui_esrrf.add_checkbox("save", description="Save Output", value=True)
gui_esrrf.add_dropdown("cmaps", description="Colormap:",
                      options=sorted(list(mpl.colormaps)),
                      value="viridis", remember_value=True)
gui_esrrf.add_button("run", description="Run")
gui_esrrf['run'].on_click(run_esrrf)
gui_esrrf.show()