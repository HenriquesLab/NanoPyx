#@title Create Decorr GUI for original image
gui_decorr_1 = EasyGui("DecorrAnalysis")

from nanopyx.core.analysis.decorr import DecorrAnalysis

def run_decorr(b):
    clear_output()
    gui_decorr_1.show()
    gui_decorr_1.save_settings()
    pixel_size = gui_decorr_1["pixel_size"].value
    units = gui_decorr_1["units"].value
    rmin = gui_decorr_1["rmin"].value
    rmax = gui_decorr_1["rmax"].value
    gui_decorr_1["run"].disabled = True
    gui_decorr_1["run"].description = "Calculating..."
    global decorr_calculator_raw
    decorr_calculator_raw = DecorrAnalysis(pixel_size=pixel_size, units=units, rmin=rmin, rmax=rmax)
    decorr_calculator_raw.run_analysis(np.mean(dataset_original, axis=0))
    gui_decorr_1["run"].disabled = False
    gui_decorr_1["run"].description = "Calculate"
    plt.imshow(decorr_calculator_raw.plot_results())
    plt.axis("off")
    plt.show()
    
gui_decorr_1.add_int_slider("pixel_size", description="Pixel Size:", min=0.01, max=1000, value=100, remember_value=True)
gui_decorr_1.add_dropdown("units", description="Units: ", options=["nm", "um", "mm"], value="nm")
gui_decorr_1.add_float_slider("rmin", description="Radius Min:", min=0.0, max=0.5, value=0.0)
gui_decorr_1.add_float_slider("rmax", description="Radius Max:", min=0.5, max=1.0, value=1.0)
gui_decorr_1.add_button("run", description="Calculate")
gui_decorr_1["run"].on_click(run_decorr)
gui_decorr_1.show()