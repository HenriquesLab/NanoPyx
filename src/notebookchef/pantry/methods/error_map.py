#@title Create Error Map GUI
gui_error = EasyGui("Error")

import numpy as np
from matplotlib import pyplot as plt
from nanopyx.core.transform.new_error_map import ErrorMap

def run_error(b):
    clear_output()
    gui_error.show()
    gui_error.save_settings()
    gui_error["run"].disabled = True
    gui_error["run"].description = "Calculating..."
    global error_map
    error_map = ErrorMap()
    error_map.optimise(np.mean(dataset_original, axis=0), np.mean(dataset_srrf[0], axis=0))
    errormap = np.array(error_map.imRSE)
    gui_error["run"].disabled = False
    gui_error["run"].description = "Calculate"
    print("RSE: ", error_map.getRSE())
    print("RSP: ", error_map.getRSP())
    plt.imshow(errormap)
    plt.axis("off")
    plt.show()

gui_error.add_button("run", description="Calculate")
gui_error["run"].on_click(run_error)
gui_error.show()