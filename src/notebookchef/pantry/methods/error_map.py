#@title Create Error Map GUI
gui_error = EasyGui("Error")

import numpy as np
from matplotlib import pyplot as plt
from nanopyx.core.transform import ErrorMap

def run_error(b):
    clear_output()
    gui_error.show()
    gui_error.save_settings()
    gui_error["run"].disabled = True
    gui_error["run"].description = "Calculating..."
    global errormap
    error_map = ErrorMap()
    error_map.optimise(np.mean(dataset_original, axis=0), np.mean(dataset_sr, axis=0))
    gui_error["run"].disabled = False
    gui_error["run"].description = "Calculate"
    print("RSE: ", error_map.getRSE())
    print("RSP: ", error_map.getRSP())
    errormap = np.array(error_map.imRSE)
    if gui_error["save"].value:
        if own_data:
            path = gui_data["upload"].selected_path
            name = gui_data["upload"].selected_filename.split(".")[0]
            tiff.imwrite(path + os.sep + name + "_error_map.tif", errormap)
        else:
            name = gui_data["data_source"].value.replace("Example dataset: ", "")
            tiff.imwrite(name + "_error_map.tif", errormap)
    plt.imshow(errormap)
    plt.axis("off")
    plt.show()

gui_error.add_checkbox("save", description="Save output", value=True)
gui_error.add_dropdown("cmaps", description="Colormap:",
                       options=sorted(list(mpl.colormaps)),
                       value="viridis", remember_value=True)
gui_error.add_button("run", description="Calculate")
gui_error["run"].on_click(run_error)
gui_error.show()
