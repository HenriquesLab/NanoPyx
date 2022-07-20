import os
import pathlib
import platform
import multiprocessing as mp
from tkinter import Image
import enanoscopy
from magicgui import magic_factory, magicgui
from napari.types import ImageData
from napari.layers import Image
from napari import Viewer
from napari.qt.threading import thread_worker



# @magic_factory(call_button="Estimate",
#               reg_meth={"regmeth":""},
#               ref_option={"widget_type": "RadioButtons",
#                           "orientation"
#  img={"label": "Image Stack"},
               #  ref_option={"widget_type": "RadioButtons",
               #             "orientation": "vertical",
              #             "value": 0,
               #            "choices": [("First Frame", 0), ("Previous Frame", 1)],
                #           "label": "Reference Frame"},
# )