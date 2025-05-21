# @title Create Error Map GUI
gui_error = EasyGui("Error")

import numpy as np
import pandas as pd
from tqdm import tqdm
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
    error_map.optimise(
        np.mean(dataset_original, axis=0), np.mean(dataset_sr, axis=0)
    )
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
            name = gui_data["data_source"].value.replace(
                "Example dataset: ", ""
            )
            tiff.imwrite(name + "_error_map.tif", errormap)
    plt.imshow(errormap)
    plt.axis("off")
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="jpeg")
    output_plot = widgets.Output()
    with output_plot:
        display(Image.open(img_buf))
    gui_error._main_display.children = gui_error._main_display.children + (
        widgets.Label(value="RSE: " + str(error_map.getRSE())),
        widgets.Label(value="RSP: " + str(error_map.getRSP())),
        output_plot,
    )
    plt.clf()


def run_error_stack(b):
    clear_output()
    gui_error.show()
    gui_error.save_settings()
    gui_error["run"].disabled = True
    gui_error["run"].description = "Calculating..."

    global errormap_stack, rse_rsp_table
    global dataset_original, dataset_sr
    errormap_stack = []
    rse_rsp_table = []

    # Ensure datasets are 3D
    if np.ndim(dataset_original) == 2:
        dataset_original = np.expand_dims(dataset_original, axis=0)
    if np.ndim(dataset_sr) == 2:
        dataset_sr = np.expand_dims(dataset_sr, axis=0)

    # Ensures datasets have same number of frames

    if dataset_original.shape[0] > dataset_sr.shape[0]:
        factor = dataset_original.shape[0] // dataset_sr.shape[0]
        remainder = dataset_original.shape[0] % dataset_sr.shape[0]
        averaged_blocks = [
            np.mean(dataset_original[i * factor : (i + 1) * factor], axis=0)
            for i in range(dataset_sr.shape[0])
        ]
        if remainder > 0:
            averaged_blocks.append(
                np.mean(dataset_original[-remainder:], axis=0)
            )
        dataset_original = np.array(averaged_blocks)

    # Iterate through each slice
    print("Processing slices...")
    for i in tqdm(range(dataset_original.shape[0]), desc="Slices processed"):
        slice_df = dataset_original[i]  #
        slice_sr = dataset_sr[i]  #

        error_map = ErrorMap()
        error_map.optimise(slice_df, slice_sr)

        # Store the error map and RSE/RSP values
        errormap_stack.append(np.array(error_map.imRSE))
        rse_rsp_table.append(
            {"Slice": i, "RSE": error_map.getRSE(), "RSP": error_map.getRSP()}
        )

    # Convert results to arrays
    errormap_stack = np.array(errormap_stack)  # 3D stack of error maps
    rse_rsp_table = pd.DataFrame(rse_rsp_table)  # Tabular results
    # Save error map stack as .tif and RSE/RSP table to CSV if required
    if gui_error["save"].value:
        if own_data:
            path = gui_data["upload"].selected_path
            name = gui_data["upload"].selected_filename.split(".")[0]
            tiff.imwrite(
                path + os.sep + name + "_error_map_stack.tif", errormap_stack
            )
            rse_rsp_table.to_csv(
                path + os.sep + name + "_rse_rsp_table.csv", index=False
            )
        else:
            name = gui_data["data_source"].value.replace(
                "Example dataset: ", ""
            )
            tiff.imwrite(name + "_error_map_stack.tif", errormap_stack)
            rse_rsp_table.to_csv(name + "_rse_rsp_table.csv", index=False)

    gui_error["run"].disabled = False
    gui_error["run"].description = "Calculate"

    # Display summary
    print("Calculation completed for all slices.")
    display(rse_rsp_table)
    plt.imshow(errormap_stack.mean(axis=0))  # Show mean error map
    plt.axis("off")
    plt.show()
    plt.clf()


gui_error.add_checkbox("save", description="Save output", value=True)
gui_error.add_dropdown(
    "cmaps",
    description="Colormap:",
    options=sorted(list(mpl.colormaps)),
    value="viridis",
    remember_value=True,
)
gui_error.add_button("run", description="Calculate")
gui_error["run"].on_click(run_error_stack)
gui_error.show()
