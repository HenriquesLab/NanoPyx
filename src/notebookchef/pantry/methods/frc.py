#@title create FRC GUI for original image
gui_frc_1 = EasyGui("FRC")

import numpy as np
from nanopyx.core.analysis.frc import FIRECalculator

def run_frc(b):
    clear_output()
    gui_frc_1.show()
    gui_frc_1.save_settings()
    pixel_size = gui_frc_1["pixel_size"].value
    units = gui_frc_1["units"].value
    first_frame = gui_frc_1["first_frame"].value
    second_frame = gui_frc_1["second_frame"].value
    gui_frc_1["run"].disabled = True
    gui_frc_1["run"].description = "Calculating..."
    global frc_calculator_raw
    frc_calculator_raw = FIRECalculator(pixel_size=pixel_size, units=units)
    frc_calculator_raw.calculate_fire_number(dataset_original[first_frame], dataset_original[second_frame])
    gui_frc_1["run"].disabled = False
    gui_frc_1["run"].description = "Calculate"
    plot = frc_calculator_raw.plot_frc_curve()
    if gui_frc_1["save"].value:
        if own_data_df:
            path = gui_data["upload"].selected_path
            name = gui_data["upload"].selected_filename.split(".")[0]
            tiff.imwrite(path + os.sep + name + "_original_FRC.tif", plot)
        else:
            name = gui_data["data_source"].value.replace("Example dataset: ", "")
            tiff.imwrite(name + "_FRC_df.tif", plot)
    plt.imshow(plot)
    plt.axis("off")
    plt.show()
    
gui_frc_1.add_int_slider("pixel_size", description="Pixel Size:", min=0.01, max=1000, value=100, remember_value=True)
gui_frc_1.add_dropdown("units", description="Units: ", options=["nm", "um", "mm"], value="nm")
gui_frc_1.add_int_slider("first_frame", description="First Frame:", min=0, max=dataset_df[0].shape[0]-1, value=0)
gui_frc_1.add_int_slider ("second_frame", description="Second Frame:", min=0, max=dataset_df[0].shape[0]-1, value=1)
gui_frc_1.add_checkbox("save", description="Save Output", value=True)
gui_frc_1.add_button("run", description="Calculate")
gui_frc_1["run"].on_click(run_frc)
gui_frc_1.show()