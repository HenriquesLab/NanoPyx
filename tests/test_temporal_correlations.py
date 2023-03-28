
from matplotlib import pyplot as plt
import numpy as np
from nanopyx.core.transform.sr_temporal_correlations import *

def test_tc():

    data = np.random.randn(10, 5, 5)
    output = calculate_SRRF_temporal_correlations(data)