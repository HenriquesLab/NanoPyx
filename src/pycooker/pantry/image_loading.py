#@title Create image loader GUI
# Create a GUI
from nanopyx.core.utils.easy_gui import EasyGui
from nanopyx.core.utils.find_files import find_files

gui_data = EasyGui("Data Loader")

def on_button_load_data_clicked(b):
    clear_output()
    gui_data.show()
    global dataset_original
    # disable button
    gui_data["load_data"].disabled = True
    gui_data["load_data"].description = "Loading..."

    if gui_data["data_source"].value.startswith("Example dataset: "):
        dataset_name = gui_data["data_source"].value.replace(
            "Example dataset: ", "")
        dataset_original = EDM.get_ZipTiffIterator(dataset_name, as_ndarray=True)
        display(stackview.slice(dataset_original, continuous_update=True,
                                colormap=gui_data["cmaps"].value))
    else:
        dataset_original = skimage.io.imread(gui_data["data_source"].value)
        display(stackview.slice(dataset_original, continuous_update=True,
                                colormap=gui_data["cmaps"].value))
    
    # enable button
    gui_data["load_data"].disabled = False
    gui_data["load_data"].description = "Load data"
    gui_data.save_settings()

gui_data.add_label("Select data to use:")
gui_data.add_dropdown("data_source", options=image_files,
                 value="Example dataset: "+example_datasets[4], remember_value=True)
gui_data.add_dropdown("cmaps", description="Colormap:",
                      options=sorted(list(mpl.colormaps)),
                      value="viridis", remember_value=True)
gui_data.add_button("load_data", description="Load data")
gui_data["load_data"].on_click(on_button_load_data_clicked)
gui_data.show()