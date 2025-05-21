#@title Create Decorrelation Analysis GUI for eSRRF data
gui_decorr = EasyGui("DecorrAnalysis")

from nanopyx.core.analysis.decorr import DecorrAnalysis

def run_decorr(b):
    clear_output()
    gui_decorr.show()
    gui_decorr.save_settings()
    pixel_size = gui_decorr["pixel_size"].value
    units = gui_decorr["units"].value
    first_frame = gui_decorr["first_frame"].value
    rmin = gui_decorr["rmin"].value
    rmax = gui_decorr["rmax"].value
    gui_decorr["run"].disabled = True
    gui_decorr["run"].description = "Calculating..."
    global decorr_calculator
    decorr_calculator = DecorrAnalysis(pixel_size=pixel_size, units=units, rmin=rmin, rmax=rmax)
    decorr_calculator.run_analysis(dataset_original[first_frame])
    gui_decorr["run"].disabled = False
    gui_decorr["run"].description = "Calculate"
    plot = decorr_calculator.plot_results()
    if gui_decorr["save"].value:
        if own_data:
            path = gui_data["upload"].selected_path
            name = gui_data["upload"].selected_filename.split(".")[0]
            tiff.imwrite(path + os.sep + name + "_eSRRF_decorr_analysis.tif", plot)
        else:
            name = gui_data["data_source"].value.replace("Example dataset: ", "")
            tiff.imwrite(name + "_eSRRF_decorr_analysis.tif", plot)
    plt.imshow(plot)
    plt.axis("off")
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="jpeg")
    output_plot = widgets.Output()
    with output_plot:
        display(Image.open(img_buf))
    gui_decorr._main_display.children = gui_decorr._main_display.children + (output_plot,)
    plt.clf()

gui_decorr.add_int_slider("pixel_size", description="Pixel Size:", min=0.01, max=1000, value=100)
gui_decorr.add_dropdown("units", description="Units: ", options=["nm", "um", "mm"], value="nm")
gui_decorr.add_int_slider("first_frame", description="Frame to be used:", min=0, max=dataset_original.shape[0]-1, value=0)
gui_decorr.add_float_slider("rmin", description="Radius Min:", min=0.0, max=0.5, value=0.0)
gui_decorr.add_float_slider("rmax", description="Radius Max:", min=0.5, max=1.0, value=1.0)
gui_decorr.add_checkbox("save", description="Save Output", value=True)
gui_decorr.add_button("run", description="Calculate")
gui_decorr["run"].on_click(run_decorr)
gui_decorr.show()