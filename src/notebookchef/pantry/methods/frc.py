#@title Create FRC GUI for original image
gui_frc_1 = EasyGui("FRC")

import numpy as np
from nanopyx.core.analysis.frc import FIRECalculator

def run_frc(b):
    clear_output()
    gui_frc_1.show()
    gui_frc_1.save_settings()
    pixel_size = gui_frc_1["pixel_size"].value
    units = gui_frc_1["units"].value
    gui_frc_1["run"].disabled = True
    gui_frc_1["run"].description = "Calculating..."
    global frc_calculator_raw
    frc_calculator_raw = FIRECalculator(pixel_size=pixel_size, units=units)
    frc_calculator_raw.calculate_fire_number(dataset_original[3], dataset_original[11])
    gui_frc_1["run"].disabled = False
    gui_frc_1["run"].description = "Calculate"
    plt.imshow(frc_calculator_raw.plot_frc_curve())
    plt.axis("off")
    plt.show()
    
gui_frc_1.add_int_slider("pixel_size", description="Pixel Size:", min=0.01, max=1000, value=100, remember_value=True)
gui_frc_1.add_dropdown("units", description="Units: ", options=["nm", "um", "mm"], value="nm")
gui_frc_1.add_button("run", description="Calculate")
gui_frc_1["run"].on_click(run_frc)
gui_frc_1.show()