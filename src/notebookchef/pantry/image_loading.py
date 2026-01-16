# @title Load image stack
# Create a GUI
gui_data = EasyGui("Data Loader")
global own_data
own_data = True


def on_button_select_own(b):
    clear_output()
    gui_data.add_label(value="Select data to use:")
    gui_data.add_file_upload("upload")
    gui_data.add_dropdown(
        "cmaps",
        description="Colormap:",
        options=sorted(list(mpl.colormaps)),
        value="viridis",
        remember_value=True,
    )
    gui_data.add_callback(
        "load_data_own", on_button_load_data_clicked, {}, description="Load data"
    )
    gui_data.show()


def on_button_select_example(b):
    clear_output()
    gui_data.add_label(value="Select data to use:")
    gui_data.add_dropdown(
        "data_source",
        options=image_files,
        value="Example dataset: " + example_datasets[4],
        remember_value=True,
    )
    gui_data.add_dropdown(
        "cmaps",
        description="Colormap:",
        options=sorted(list(mpl.colormaps)),
        value="viridis",
        remember_value=True,
    )
    gui_data.add_callback(
        "load_data",
        on_button_load_data_clicked_example,
        {},
        description="Load data",
    )
    gui_data.show()


def on_button_load_data_clicked(b):
    clear_output()
    gui_data.show()
    global dataset_original
    global own_data
    own_data = True
    # disable button
    gui_data["load_data_own"].disabled = True
    gui_data["load_data_own"].description = "Loading..."
    dataset_original = tiff.imread(gui_data["upload"].selected)
    gui_data["load_data_own"].disabled = False
    gui_data["load_data_own"].description = "Load data"
    gui_data._main_display.children = gui_data._main_display.children + (
        stackview.slice(
            dataset_original,
            colormap=gui_data["cmaps"].value,
            continuous_update=True,
        ),
    )


def on_button_load_data_clicked_example(b):
    clear_output()
    gui_data.show()
    global dataset_original
    global own_data
    own_data = False
    # disable button
    gui_data["load_data"].disabled = True
    gui_data["load_data"].description = "Loading..."

    if gui_data["data_source"].value.startswith("Example dataset: "):
        dataset_name = gui_data["data_source"].value.replace(
            "Example dataset: ", ""
        )
        dataset_original = EDM.get_ZipTiffIterator(
            dataset_name, as_ndarray=True
        )
        gui_data._main_display.children = gui_data._main_display.children + (
            stackview.slice(
                dataset_original,
                continuous_update=True,
                colormap=gui_data["cmaps"].value,
            ),
        )
    else:
        dataset_original = skimage.io.imread(gui_data["data_source"].value)
        gui_data._main_display.children = gui_data._main_display.children + (
            stackview.slice(
                dataset_original,
                continuous_update=True,
                colormap=gui_data["cmaps"].value,
            ),
        )

    # enable button
    gui_data["load_data"].disabled = False
    gui_data["load_data"].description = "Load data"


gui_data.add_callback(
    "use_own_data", on_button_select_own, {}, description="Use Own Data"
)
gui_data.add_callback(
    "use_example_data",
    on_button_select_example,
    {}, description="Use Example data",
)
gui_data.show()
