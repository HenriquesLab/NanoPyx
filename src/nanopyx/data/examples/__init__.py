"""
Placeholder for containing the example test data. Within this package folder is the metadata information for each test dataset
See `nanopyx.data.download` for tools to download the example data

>>> import os, glob
>>> path = get_path()
>>> "/".join(path.split(os.path.sep)[-3:])
'nanopyx/data/examples'
"""

import os


def get_path() -> str:
    """
    :return: path to the examples info directory
    """
    return os.path.join(os.path.split(__file__)[0])
