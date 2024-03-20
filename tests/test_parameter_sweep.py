import nanopyx
import numpy as np


def test_param_sweep():
    img = np.random.random((10, 100, 100))
    nanopyx.run_esrrf_parameter_sweep(img)
