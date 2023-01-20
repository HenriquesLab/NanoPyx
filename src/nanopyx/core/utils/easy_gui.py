"""
A module to help simplify the create of GUIs in Jupyter notebooks using ipywidgets.
"""

import ipywidgets as widgets
from IPython.display import display
import os
import yaml


class EasyGui:

    def __init__(self, title="basic_gui", width='50%'):
        """
        Container for widgets.
        :param width: width of the widget container
        """
        self._layout = widgets.Layout(width=width)
        self._style = {'description_width': 'initial'}
        self._widgets = {}
        self._nLabels = 0
        self._main_display = widgets.Output()
        self._title = title
        self._cfg = {title: {}}

        # Get the user's home folder
        self._home_folder = os.path.expanduser("~")
        self._config_folder = os.path.join(self._home_folder, ".nanopyx")
        if not os.path.exists(self._config_folder):
            os.makedirs(self._config_folder)

        self._config_file = os.path.join(self._config_folder, "easy_gui.yml")
        if os.path.exists(self._config_file):
            with open(self._config_file, "r") as f:
                _cfg = yaml.load(f, Loader=yaml.FullLoader)
                if title in _cfg:
                    self._cfg = _cfg

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
        self._widgets[f"label_{self._nLabels}"] = widgets.Label(
            *args, **kwargs, layout=self._layout, style=self._style)

    def add_button(self, tag, *args, **kwargs):
        """
        Add a button widget to the container.
        :param tag: tag to identify the widget
        :param args: args for the widget
        :param kwargs: kwargs for the widget
        """
        self._widgets[tag] = widgets.Button(
            *args, **kwargs, layout=self._layout, style=self._style)

    def add_text(self, tag, *args, **kwargs):
        """
        Add a text widget to the container.
        :param tag: tag to identify the widget
        :param args: args for the widget
        :param kwargs: kwargs for the widget
        """
        self._widgets[tag] = widgets.Text(
            *args, **kwargs, layout=self._layout, style=self._style)

    def add_int_slider(self, tag, *args, remember_value=False, **kwards):
        """
        Add a integer slider widget to the container.
        :param tag: tag to identify the widget
        :param args: args for the widget
        :param kwargs: kwargs for the widget
        """
        if remember_value and tag in self._cfg[self._title]:
            kwards["value"] = self._cfg[self._title][tag]
        self._widgets[tag] = widgets.IntSlider(
            *args, **kwards, layout=self._layout, style=self._style)

    def add_float_slider(self, tag, *args, remember_value=False, **kwards):
        """
        Add a float slider widget to the container.
        :param tag: tag to identify the widget
        :param args: args for the widget
        :param kwargs: kwargs for the widget
        """
        self._widgets[tag] = widgets.FloatSlider(
            *args, **kwards, layout=self._layout, style=self._style)

    def add_checkbox(self, tag, *args, remember_value=False, **kwargs):
        """
        Add a checkbox widget to the container.
        :param tag: tag to identify the widget
        :param args: args for the widget
        :param kwargs: kwargs for the widget
        """
        if remember_value and tag in self._cfg[self._title]:
            kwargs["value"] = self._cfg[self._title][tag]
        self._widgets[tag] = widgets.Checkbox(
            *args, **kwargs, layout=self._layout, style=self._style)

    def add_int_text(self, tag, *args, remember_value=False, **kwargs):
        """
        Add a integer text widget to the container.
        :param tag: tag to identify the widget
        :param args: args for the widget
        :param kwargs: kwargs for the widget
        """
        if remember_value and tag in self._cfg[self._title]:
            kwargs["value"] = self._cfg[self._title][tag]
        self._widgets[tag] = widgets.IntText(
            *args, **kwargs, layout=self._layout, style=self._style)

    def add_dropdown(self, tag, *args, remember_values=False, **kwargs):
        """
        Add a dropdown widget to the container.
        :param tag: tag to identify the widget
        :param args: args for the widget
        :param kwargs: kwargs for the widget
        """
        self._widgets[tag] = widgets.Dropdown(
            *args, **kwargs, layout=self._layout, style=self._style)

    def add_file_upload(self, tag, *args, accept='image/*', multiple=False, **kwargs):
        """
        Add a file upload widget to the container.
        :param tag: tag to identify the widget
        :param args: args for the widget
        :param accept: file types to accept
        :param multiple: allow multiple files to be uploaded
        :param kwargs: kwargs for the widget
        """
        self._widgets[tag] = widgets.FileUpload(
            *args, accept=accept, multiple=multiple, **kwargs, layout=self._layout, style=self._style)

    def show(self):
        """
        Show the widgets in the container.
        """
        for tag in self._widgets:
            if tag.startswith("label_"):
                pass
            elif hasattr(self._widgets[tag], "value"):
                self._cfg[self._title][tag] = self._widgets[tag].value

        with open(self._config_file, "w") as f:
            yaml.dump(self._cfg, f)
        display(*self._widgets.values())

    def clear(self):
        """
        Clear the widgets in the container.
        """
        self._widgets = {}
        self._nLabels = 0
        self._main_display.clear_output()
