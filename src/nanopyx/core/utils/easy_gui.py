"""
A module to help simplify the create of GUIs in Jupyter notebooks using ipywidgets.
"""

import os
import yaml
import platform
import numpy as np
from ipyfilechooser import FileChooser
from skimage.exposure import rescale_intensity

# import cache if python >= 3.9, otherwise import lru_cache
if platform.python_version_tuple() >= ("3", "9"):
    from functools import cache
else:
    from functools import lru_cache as cache

try:
    import ipywidgets as widgets
    from IPython import display as dp
    from IPython.display import display
    from matplotlib import pyplot as plt
except ImportError:
    print("jupyter optional-dependencies not installed, conside installing with 'pip install nanopyx[jupyter]'")
    raise ImportError


class EasyGui:
    def __init__(self, title="basic_gui", width="50%"):
        """
        Container for widgets.
        :param width: width of the widget container
        """
        self._layout = widgets.Layout(width=width)
        self._style = {"description_width": "initial"}
        self._widgets = {}
        self._nLabels = 0
        self._main_display = widgets.Output()
        self._title = title
        self._cfg = {title: {}}
        self.cfg = self._cfg[title]

        # Get the user's home folder
        self._home_folder = os.path.expanduser("~")
        self._config_folder = os.path.join(self._home_folder, ".nanopyx")
        if not os.path.exists(self._config_folder):
            os.makedirs(self._config_folder)

        self._config_file = os.path.join(self._config_folder, "easy_gui.yml")
        if os.path.exists(self._config_file):
            with open(self._config_file, "r") as f:
                self._cfg = yaml.load(f, Loader=yaml.FullLoader)
                if title in self._cfg:
                    self.cfg = self._cfg[title]

    def __getitem__(self, tag: str) -> widgets.Widget:
        return self._widgets[tag]

    def __len__(self) -> int:
        return len(self._widgets)

    def add_label(self, *args, **kwargs):
        """
        Add a label widget to the container.
        :param args: args for the widget
        :param kwargs: kwargs for the widget
        """
        self._nLabels += 1
        self._widgets[f"label_{self._nLabels}"] = widgets.Label(*args, **kwargs, layout=self._layout, style=self._style)

    def add_button(self, tag, *args, **kwargs):
        """
        Add a button widget to the container.
        :param tag: tag to identify the widget
        :param args: args for the widget
        :param kwargs: kwargs for the widget
        """
        self._widgets[tag] = widgets.Button(*args, **kwargs, layout=self._layout, style=self._style)

    def add_text(self, tag, *args, **kwargs):
        """
        Add a text widget to the container.
        :param tag: tag to identify the widget
        :param args: args for the widget
        :param kwargs: kwargs for the widget
        """
        self._widgets[tag] = widgets.Text(*args, **kwargs, layout=self._layout, style=self._style)

    def add_int_slider(self, tag, *args, remember_value=False, **kwargs):
        """
        Add a integer slider widget to the container.
        :param tag: tag to identify the widget
        :param args: args for the widget
        :param remember_value: remember the last value
        :param kwargs: kwargs for the widget
        """
        if remember_value and tag in self.cfg and kwargs["min"] <= self.cfg[tag] <= kwargs["max"]:
            kwargs["value"] = self.cfg[tag]
        self._widgets[tag] = widgets.IntSlider(*args, **kwargs, layout=self._layout, style=self._style)

    def add_float_slider(self, tag, *args, remember_value=False, **kwargs):
        """
        Add a float slider widget to the container.
        :param tag: tag to identify the widget
        :param args: args for the widget
        :param remember_value: remember the last value
        :param kwargs: kwargs for the widget
        """
        if remember_value and tag in self.cfg:
            kwargs["value"] = self.cfg[tag]
        self._widgets[tag] = widgets.FloatSlider(*args, **kwargs, layout=self._layout, style=self._style)

    def add_checkbox(self, tag, *args, remember_value=False, **kwargs):
        """
        Add a checkbox widget to the container.
        :param tag: tag to identify the widget
        :param args: args for the widget
        :param remember_value: remember the last value
        :param kwargs: kwargs for the widget
        """
        if remember_value and tag in self.cfg:
            kwargs["value"] = self.cfg[tag]
        self._widgets[tag] = widgets.Checkbox(*args, **kwargs, layout=self._layout, style=self._style)

    def add_int_text(self, tag, *args, remember_value=False, **kwargs):
        """
        Add a integer text widget to the container.
        :param tag: tag to identify the widget
        :param args: args for the widget
        :param remember_value: remember the last value
        :param kwargs: kwargs for the widget
        """
        if remember_value and tag in self.cfg:
            kwargs["value"] = self.cfg[tag]

        self._widgets[tag] = widgets.IntText(
            *args, **kwargs, layout=self._layout, style=self._style)
        
    def add_float_text(self, tag, *args, remember_value=False, **kwargs):
        """
        Add a float text widget to the container.
        :param tag: tag to identify the widget
        :param args: args for the widget
        :param remember_value: remember the last value
        :param kwargs: kwargs for the widget
        """
        if remember_value and tag in self.cfg:
            kwargs["value"] = self.cfg[tag]
        self._widgets[tag] = widgets.FloatText(
            *args, **kwargs, layout=self._layout, style=self._style)

    def add_dropdown(self, tag, *args, remember_value=False, **kwargs):
        """
        Add a dropdown widget to the container.
        :param tag: tag to identify the widget
        :param args: args for the widget
        :param remember_value: remember the last value
        :param kwargs: kwargs for the widget
        """
        if remember_value and tag in self.cfg and self.cfg[tag] in kwargs["options"]:
            kwargs["value"] = self.cfg[tag]
        self._widgets[tag] = widgets.Dropdown(*args, **kwargs, layout=self._layout, style=self._style)

    def add_file_upload(self, tag, *args, accept="*", multiple=False, **kwargs):
        """
        Add a file upload widget to the container.
        :param tag: tag to identify the widget
        :param args: args for the widget
        :param accept: file types to accept
        :param multiple: allow multiple files to be uploaded
        :param kwargs: kwargs for the widget
        """
        self._widgets[tag] = FileChooser()

    def save_settings(self):
        # remember widget values for next time and store them in a config file
        for tag in self._widgets:
            if tag.startswith("label_"):
                pass
            elif hasattr(self._widgets[tag], "value"):
                self.cfg[tag] = self._widgets[tag].value
        self._cfg[self._title] = self.cfg
        with open(self._config_file, "w") as f:
            yaml.dump(self._cfg, f)

    def show(self):
        """
        Show the widgets in the container.
        """
        # display the widgets
        display(*self._widgets.values())

    def clear(self):
        """
        Clear the widgets in the container.
        """
        self._widgets = {}
        self._nLabels = 0
        self._main_display.clear_output()


def view_image(image):
    """
    Plot an image.
    :param image: image to be plotted
    """
    fig, ax = plt.subplots()
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    ax.imshow(image)
    plt.axis("off")
    plt.draw()


def view_image_stack(image, cmap="viridis"):
    """
    Plot an image stack with dimensions >= 2.
    Sliders to move across the dimensions are added.
    :param cmap: colormap to be use to plot the image
    """
    cm = plt.get_cmap(cmap)
    dims = image.shape
    params = {}
    if len(dims) > 2:
        for i in range(len(dims) - 2):
            params["dim" + str(i)] = widgets.IntSlider(
                min=0, max=dims[i] - 1, step=1, value=0, description="dim" + str(i)
            )
    fig, ax = plt.subplots()
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    def show_slice(**kwargs):
        tmp_1 = rescale_intensity(image)
        for k, value in kwargs.items():
            if k != "curtain":
                tmp_1 = tmp_1[value]
        ax.imshow(tmp_1, cmap=cmap)
        plt.axis("off")
        plt.draw()

    widgets.interact(show_slice, **params)


def view_curtain_stack(image_1: np.ndarray, image_2: np.ndarray, cmap: str = "viridis"):
    """
    Plot two image stacks with dimensions >= 2.
    Sliders to move across the dimensions are added.

    :param image_1: Left image to be plotted on the curtain
    :param image_2: Right image to be plotted on the curtain
    :param cmap: Matplotlib colormap to be used to plot the images. Defaults to "viridis".
    """
    assert image_1.shape == image_2.shape
    dims = image_1.shape
    params = {}
    params["curtain"] = widgets.IntSlider(
        value=image_1.shape[-1] / 2, min=0, max=image_1.shape[-1], description="Curtain"
    )
    if len(dims) > 2:
        for i in range(len(dims) - 2):
            params["dim" + str(i)] = widgets.IntSlider(
                min=0, max=dims[i] - 1, step=1, value=0, description="dim" + str(i)
            )

    fig, ax = plt.subplots()
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    
    def show_slice(**kwargs):
        tmp_1 = rescale_intensity(image_1)
        tmp_2 = rescale_intensity(image_2)
        for k, value in kwargs.items():
            if k != "curtain":
                tmp_1 = tmp_1[value]
                tmp_2 = tmp_2[value]

        for k, value in kwargs.items():
            if k == "curtain":
                combined = np.zeros((image_1.shape[-2], image_1.shape[-1]))
                combined[:, : int(value)] += tmp_1[:, : int(value)]
                combined[:, int(value) :] += tmp_2[:, int(value) :]
        ax.imshow(combined, cmap=cmap)
        plt.axis("off")
        plt.draw()

    widgets.interact(show_slice, **params)
