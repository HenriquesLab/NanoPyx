"""
Placeholder for containing the example test data. Within this package folder is the metadata information for each test dataset. 

For example

>>> import os, glob
>>> path = get_path()
>>> "/".join(path.split(os.path.sep)[-3:])
'nanopyx/data/examples'
"""

import os


def get_path():
    """
    Returns the path to the examples info directory
    """
    return os.path.join(os.path.split(__file__)[0])
