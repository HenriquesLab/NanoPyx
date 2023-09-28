#@title Create channel registration GUI
from nanopyx.methods import channel_registration

gui_reg = EasyGui("Channel Registration")

def on_button_register(b):
    clear_output()
    gui_reg.show()
    ref_channel = gui_reg["ref"].value
    max_shift = gui_reg["max"].value
    n_blocks = gui_reg["blocks"].value
    min_sim = gui_reg["min_sim"].value
    global dataset_registered
    gui_reg["register"].disabled = True
    gui_reg["register"].description = "Aligning..."
    dataset_registered = channel_registration.estimate_channel_registration(dataset_original,
                                                                            ref_channel,
                                                                            max_shift,
                                                                            n_blocks,
                                                                            min_sim,
                                                                            save_translation_masks=False,
                                                                            save_ccms=False,
                                                                            apply=True)
    if gui_reg["save"].value:
        if own_data:
            path = gui_data["upload"].selected_path
            name = gui_data["upload"].selected_filename.split(".")[0]
            tiff.imwrite(path + os.sep + name + "_registered.tif", dataset_registered)
        else:
            name = gui_data["data_source"].value.replace("Example dataset: ", "")
            tiff.imwrite(name + "_registered.tif", dataset_registered)
    gui_reg["register"].disabled = False
    gui_reg["register"].description = "Align"
    display(stackview.slice(dataset_registered, colormap=gui_reg["cmaps"].value,
                            continuous_update=True))

gui_reg.add_label("Channel Registration parameters:")
gui_reg.add_int_slider("ref", description="Reference channel", min=0, max=dataset_original.shape[0]-1, value=0)
gui_reg.add_int_slider("max", description="Max expected drift", min=0, max=1000, value=10)
gui_reg.add_int_slider("blocks", description="Blocks per axis", min=1, max=10, value=5)
gui_reg.add_float_slider("min_sim", description="Minimum similarity", min=0, max=1, value=0.5, step=0.1)
gui_reg.add_dropdown("cmaps", description="Colormap:",
                     options=sorted(list(mpl.colormaps)),
                     value="viridis", remember_value=True)
gui_reg.add_checkbox("save", description="Save Output", value=True)
gui_reg.add_button("register", description="Register")
gui_reg["register"].on_click(on_button_register)
gui_reg.show()