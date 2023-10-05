#@title Install NanoPyx, import necessary libraries and connect to Google Drive
!pip install -q "nanopyx[jupyter]"
import io
import os
import sys
import cv2 as cv
import skimage
import nanopyx
import stackview
import numpy as np
import tifffile as tiff
import matplotlib as mpl
import ipywidgets as widgets
from PIL import Image
from IPython.display import display, clear_output
from matplotlib import pyplot as plt

from nanopyx.core.utils.easy_gui import EasyGui
from nanopyx.core.utils.find_files import find_files
from nanopyx.data.download import ExampleDataManager

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    !pip install -q ipycanvas==0.11.0
    from google.colab import output
    output.enable_custom_widget_manager()
    from google.colab import drive
    drive.mount('/content/drive')

cwd = os.getcwd()
image_folder = "datasets"
image_files = []
EDM = ExampleDataManager()
example_datasets = EDM.list_datasets()

_path = os.path.join("..", image_folder)
if os.path.exists(_path):
    image_files += find_files(_path, ".tif")
if os.path.exists(image_folder):
    image_files += find_files(image_folder, ".tif")
image_files += ["Example dataset: "+dataset for dataset in example_datasets]
