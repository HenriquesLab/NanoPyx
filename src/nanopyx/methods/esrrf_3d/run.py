from ...core.transform._le_esrrf3d import eSRRF3D
from ...core.transform.sr_temporal_correlations import calculate_eSRRF3d_temporal_correlations

def run_esrrf3d(img, mag: int = 5, temporal_correlation_window: int = 10):
    esrrf_calculator = eSRRF3D()
    return calculate_eSRRF3d_temporal_correlations(esrrf_calculator.run(img, mag), temporal_correlation_window)