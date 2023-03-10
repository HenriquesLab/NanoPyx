
from matplotlib import pyplot as plt
import numpy as np
from nanopyx.core.transform.sr_temporal_correlations import TemporalCorrelation

def test_tc(downloader):

    data = np.random.randn(10, 5, 5)
    tc = TemporalCorrelation("tac2")
    output = tc.calculate_tc(data)
    plt.imshow(output)