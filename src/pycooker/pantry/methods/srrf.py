
#@title Create SRRF GUI
gui_srrf = EasyGui("srrf")
from nanopyx.methods.srrf import SRRF

def run_srrf(b):
    clear_output()
    gui_srrf.show()
    gui_srrf.save_settings()
    ring_radius = gui_srrf["ring_radius"].value
    magnification = gui_srrf["magnification"].value
    frames_per_timepoint = gui_srrf["frames_per_timepoint"].value
    srrf_order = gui_srrf["srrf_order"].value
    # disable button while running
    gui_srrf["run"].disabled = True
    gui_srrf["run"].description = "Running..."
    srrf = SRRF(magnification, ring_radius)
    global dataset_srrf
    dataset_srrf = srrf.calculate(dataset_original, frames_per_timepoint, srrf_order)
    # enable button again
    gui_srrf["run"].disabled = False
    gui_srrf["run"].description = "Run"
    display(stackview.curtain(dataset_srrf[0], dataset_srrf[1],
                             continuous_update=True,
                             colormap=gui_data["cmaps"].value,
                             curtain_colormap=gui_data["cmaps"].value))

gui_srrf.add_float_slider("ring_radius", description="Ring Radius:", min=0.1, max=3.0, value=0.5, remember_value=True)
gui_srrf.add_int_slider("magnification", description="Magnification:", min=1, max=10, value=5)
gui_srrf.add_int_slider("srrf_order", description="SRRF order:", min=-1, max=4, value=3)
gui_srrf.add_label("-=-= Time-Lapse =-=-")
gui_srrf.add_int_slider("frames_per_timepoint", description="Frames per time-point (0 - auto)", min=1, max=dataset_original.shape[0], value=dataset_original.shape[0]//2)
gui_srrf.add_dropdown("cmaps", description="Colormap:",
                      options=sorted(list(mpl.colormaps)),
                      value="viridis", remember_value=True)
gui_srrf.add_button("run", description="Run")
gui_srrf['run'].on_click(run_srrf)
gui_srrf.show()