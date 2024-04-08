# @title Create Parameter Sweep GUI
gui_param_sweep = EasyGui("Parameter Sweep")
from nanopyx import run_esrrf_parameter_sweep


def run_param_sweep(b):
    clear_output()
    gui_param_sweep.show()
    gui_param_sweep.save_settings()
    radii = [float(val) for val in gui_param_sweep["raddi"].value.split(",")]
    sensitivities = [float(val) for val in gui_param_sweep["sensitivities"].value.split(",")]
    global global_mag
    global_mag = gui_param_sweep["magnification"].value
    temp_corr = gui_param_sweep["temp_corr"].value
    use_decorr = gui_param_sweep["use_decorr"].value

    global g_temp_corr
    if temp_corr == "AVG":
        g_temp_corr = 1
    elif temp_corr == "VAR":
        g_temp_corr = 2
    elif temp_corr == "TAC2":
        g_temp_corr = 3

    gui_param_sweep["run"].disabled = True
    gui_param_sweep["run"].description = "Running..."

    param_sweep_out = run_esrrf_parameter_sweep(
        dataset_original, magnification=global_mag, sensitivities=sensitivities, radii=radii, temporal_correlation=temp_corr, plot_sweep=True, return_qnr=True, use_decorr=use_decorr
    )

    sens_idx, rad_idx = np.unravel_index(np.argmax(param_sweep_out), param_sweep_out.shape)
    global optimal_sensitivity
    optimal_sensitivity = sensitivities[sens_idx]
    global optimal_radius
    optimal_radius = radii[rad_idx]
    print(f"Optimal sensitivity is: {optimal_sensitivity}; and optimal radius is: {optimal_radius}")


gui_param_sweep.add_int_slider("magnification", description="Magnification:", min=1, max=10, value=2)
gui_param_sweep.add_text("sensitivities", description="List of sensitivities (comma-separated):", value="1, 2")
gui_param_sweep.add_text("raddi", description="List of radii (comma-separated):", value="1, 1.5")
gui_param_sweep.add_dropdown(
    "temp_corr", description="Temperature Correction:", options=["AVG", "VAR", "TAC2"], value="AVG"
)
gui_param_sweep.add_checkbox("use_decorr", description="Use Decorrelation instead of FRC", value=False)
gui_param_sweep.add_button("run", description="Run")
gui_param_sweep.add_dropdown(
    "cmaps", description="Colormap:", options=sorted(list(mpl.colormaps)), value="viridis", remember_value=True
)
gui_param_sweep["run"].on_click(run_param_sweep)
gui_param_sweep.show()
