"""
Place-holder for containing the example test data

Use the `download` module as the main mechanism for downloading the example data.

""" 

import os

def get_path():
    """
    Returns the path to the examples info directory
    """
    return os.path.join(os.path.split(__file__)[0])
