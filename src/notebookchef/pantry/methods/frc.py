#@title create FRC GUI for original image
gui_frc = EasyGui("FRC")

import numpy as np
from nanopyx.core.analysis.frc import FIRECalculator

def run_frc(b):
    clear_output()
    gui_frc.show()
    gui_frc.save_settings()
    pixel_size = gui_frc["pixel_size"].value
    units = gui_frc["units"].value
    first_frame = gui_frc["first_frame"].value
    second_frame = gui_frc["second_frame"].value
    gui_frc["run"].disabled = True
    gui_frc["run"].description = "Calculating..."
    global frc_calculator_raw
    frc_calculator_raw = FIRECalculator(pixel_size=pixel_size, units=units)
    frc_calculator_raw.calculate_fire_number(dataset_original[first_frame], dataset_original[second_frame])
    gui_frc["run"].disabled = False
    gui_frc["run"].description = "Calculate"
    plot = frc_calculator_raw.plot_frc_curve()
    if gui_frc["save"].value:
        if own_data:
            path = gui_data["upload"].selected_path
            name = gui_data["upload"].selected_filename.split(".")[0]
            tiff.imwrite(path + os.sep + name + "_original_FRC.tif", plot)
        else:
            name = gui_data["data_source"].value.replace("Example dataset: ", "")
            tiff.imwrite(name + "_FRC_df.tif", plot)
    plt.imshow(plot)
    plt.axis("off")
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="jpeg")
    output_plot = widgets.Output()
    with output_plot:
        display(Image.open(img_buf))
    gui_frc._main_display.children = gui_frc._main_display.children + (output_plot,)
    plt.clf()
    
gui_frc.add_int_slider("pixel_size", description="Pixel Size:", min=0.01, max=1000, value=100)
gui_frc.add_dropdown("units", description="Units: ", options=["nm", "um", "mm"], value="nm")
gui_frc.add_int_slider("first_frame", description="First Frame:", min=0, max=dataset_original[0].shape[0]-1, value=0)
gui_frc.add_int_slider ("second_frame", description="Second Frame:", min=0, max=dataset_original[0].shape[0]-1, value=1)
gui_frc.add_checkbox("save", description="Save Output", value=True)
gui_frc.add_button("run", description="Calculate")
gui_frc["run"].on_click(run_frc)
gui_frc.show()