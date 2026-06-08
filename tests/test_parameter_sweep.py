import nanopyx
import numpy as np
import pytest
from nanopyx.core.analysis.parameter_sweep import ParameterSweep
import nanopyx.core.analysis.parameter_sweep as parameter_sweep_module


def test_param_sweep_n_frames():
    img = np.random.random((10, 100, 100))
    nanopyx.run_esrrf_parameter_sweep(img, n_frames=2, use_decorr=True, plot_sweep=False)

def test_param_sweep_class():
    ps = ParameterSweep()
    img = np.random.random((2, 100, 100))
    ps.run(img, magnification=2, sensitivity_array=[1, 2], radius_array=[1, 1.5], n_frames=None)


def test_param_sweep_n_frames_processes_chunks(monkeypatch):
    calls = []

    class FakeESRRF:
        def __init__(self, verbose=True):
            pass

        def run(self, im, magnification, radius, sensitivity):
            calls.append(im.shape[0])
            return np.ones((im.shape[0], im.shape[1] * magnification, im.shape[2] * magnification))

    def fake_temporal_correlation(rgc_map, temporal_correlation):
        return np.mean(rgc_map, axis=0)

    monkeypatch.setattr(parameter_sweep_module, "eSRRF", FakeESRRF)
    monkeypatch.setattr(
        parameter_sweep_module,
        "calculate_eSRRF_temporal_correlations",
        fake_temporal_correlation,
    )
    monkeypatch.setattr(ParameterSweep, "calculate_rsp", lambda self, im, reconstruction: 1)
    monkeypatch.setattr(ParameterSweep, "calculate_frc", lambda self, im_odd, im_even: 1)

    img = np.random.random((10, 5, 5))
    ps = ParameterSweep()
    ps.run(img, magnification=2, sensitivity_array=[1], radius_array=[1], n_frames=3)

    assert calls == [3, 3, 3, 1]


def test_param_sweep_rejects_invalid_n_frames():
    ps = ParameterSweep()
    img = np.random.random((2, 5, 5))

    with pytest.raises(ValueError, match="n_frames"):
        ps.run(img, magnification=2, sensitivity_array=[1], radius_array=[1], n_frames=0)


def test_qnr_score_handles_constant_and_non_finite_frc():
    ps = ParameterSweep()
    rsp = np.ones((2, 2), dtype=np.float32)
    frc = np.array([[2, 2], [np.nan, 2]], dtype=np.float32)

    qnr = ps.calculate_qnr_score(rsp, frc)

    assert np.all(np.isfinite(qnr))
    assert qnr[0, 0] > 0


def test_qnr_score_does_not_collapse_on_flat_zero_frc():
    ps = ParameterSweep()
    rsp = np.ones((2, 2), dtype=np.float32)
    frc = np.zeros((2, 2), dtype=np.float32)

    qnr = ps.calculate_qnr_score(rsp, frc)

    assert np.all(qnr > 0)
