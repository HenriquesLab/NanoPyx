#@title Create NLM Denoising GUI
gui_nlm = EasyGui("nlm")
from nanopyx.core.transform._le_nlm_denoising import NLMDenoising

def run_nlm(b):
    clear_output()
    gui_nlm.show()
    gui_nlm.save_settings()
    patch_size = gui_nlm["patch_size"].value
    patch_distance = gui_nlm["patch_distance"].value
    h = gui_nlm["h"].value
    sigma = gui_nlm["sigma"].value
    # disable button while running
    gui_nlm["run"].disabled = True
    gui_nlm["run"].description = "Running..."

    denoiser = NLMDenoising()

    global dataset_nlm
    dataset_nlm = denoiser.run(dataset_original, patch_size=patch_size, patch_distance=patch_distance, h=h, sigma=sigma)
    # enable button again
    gui_nlm["run"].disabled = False
    gui_nlm["run"].description = "Run"
    if gui_nlm["save"].value:
        if own_data:
            path = gui_data["upload"].selected_path
            name = gui_data["upload"].selected_filename.split(".")[0]
            tiff.imwrite(path + os.sep + name + "_nlm_denoised.tif", dataset_nlm)
        else:
            name = gui_data["data_source"].value.replace("Example dataset: ", "")
            tiff.imwrite(name + "_nlm_denoised.tif", dataset_nlm)
    gui_nlm._main_display.children = gui_nlm._main_display.children + (stackview.slice(dataset_nlm, colormap=gui_nlm["cmaps"].value, continuous_update=True),)

gui_nlm.add_int_slider("patch_size", description="Patch Size", min=1, max=dataset_original.shape[-1]//2, value=5)
gui_nlm.add_int_slider("patch_distance", description="Patch Distance", min=1, max=dataset_original.shape[-1]//2, value=10)
gui_nlm.add_float_text("h", description="h", value=0.1, remember_value=True)
gui_nlm.add_float_text("sigma", description="sigma", value=0.1, remember_value=True)
gui_nlm.add_checkbox("save", description="Save Output", value=True)
gui_nlm.add_dropdown("cmaps", description="Colormap:",
                      options=sorted(list(mpl.colormaps)),
                      value="viridis", remember_value=True)
gui_nlm.add_button("run", description="Run")
gui_nlm['run'].on_click(run_nlm)
gui_nlm.show()
