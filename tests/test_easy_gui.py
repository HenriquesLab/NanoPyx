import pytest
import os
import shutil
import yaml
from unittest.mock import patch
from nanopyx.core.utils.easy_gui import EasyGui


@pytest.fixture
def gui():
    # Setup: create a GUI instance and ensure the config folder is clean
    gui_instance = EasyGui("test_gui")
    if os.path.exists(gui_instance._config_folder):
        shutil.rmtree(gui_instance._config_folder)
    os.makedirs(gui_instance._config_folder)
    return gui_instance


def test_add_label(gui):
    gui.add_label("Test Label")
    assert len(gui) == 1
    assert "label_1" in gui._widgets
    assert gui._widgets["label_1"].value == "Test Label"


def test_add_button(gui):
    gui.add_button("test_button", description="Click Me")
    assert "test_button" in gui._widgets
    assert gui._widgets["test_button"].description == "Click Me"


def test_add_text(gui):
    gui.add_text("test_text", value="Hello")
    assert "test_text" in gui._widgets
    assert gui._widgets["test_text"].value == "Hello"


def test_add_int_slider(gui):
    gui.add_int_slider("test_int_slider", min=0, max=10, value=5)
    assert "test_int_slider" in gui._widgets
    assert gui._widgets["test_int_slider"].value == 5


def test_add_float_slider(gui):
    gui.add_float_slider("test_float_slider", min=0, max=1, value=0.5)
    assert "test_float_slider" in gui._widgets
    assert gui._widgets["test_float_slider"].value == 0.5


def test_add_checkbox(gui):
    gui.add_checkbox("test_checkbox", value=True)
    assert "test_checkbox" in gui._widgets
    assert gui._widgets["test_checkbox"].value is True


def test_add_int_text(gui):
    gui.add_int_text("test_int_text", value=5)
    assert "test_int_text" in gui._widgets
    assert gui._widgets["test_int_text"].value == 5


def test_add_float_text(gui):
    gui.add_float_text("test_float_text", value=0.5)
    assert "test_float_text" in gui._widgets
    assert gui._widgets["test_float_text"].value == 0.5


def test_add_dropdown(gui):
    gui.add_dropdown("test_dropdown", options=["Option 1", "Option 2"], value="Option 1")
    assert "test_dropdown" in gui._widgets
    assert gui._widgets["test_dropdown"].value == "Option 1"


def test_add_file_upload(gui):
    gui.add_file_upload("test_file_upload")
    assert "test_file_upload" in gui._widgets


def test_save_settings(gui):
    gui.add_text("test_text", value="Hello")
    gui.save_settings()
    config_file = os.path.join(gui._config_folder, "easy_gui.yml")
    assert os.path.exists(config_file)
    with open(config_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        assert cfg["test_gui"]["test_text"] == "Hello"


def test_show(gui):
    gui.add_label("Show Label Test")


def test_clear(gui):
    gui.add_label("Label to clear")
    assert len(gui) == 1
    gui.clear()
    assert len(gui) == 0
    assert len(gui._main_display.children) == 0
