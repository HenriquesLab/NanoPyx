import nanopyx
import numpy as np
from nanopyx.core.analysis.parameter_sweep import ParameterSweep


def test_param_sweep_n_frames():
    img = np.random.random((10, 100, 100))
    nanopyx.run_esrrf_parameter_sweep(img, n_frames=2, use_decorr=True, plot_sweep=False)

def test_param_sweep_class():
    ps = ParameterSweep()
    img = np.random.random((2, 100, 100))
    ps.run(img, magnification=2, sensitivity_array=[1, 2], radius_array=[1, 1.5], n_frames=None)