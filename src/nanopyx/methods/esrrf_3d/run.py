from ...core.transform._le_esrrf3d import eSRRF3D
from ...core.transform.sr_temporal_correlations import calculate_eSRRF3d_temporal_correlations

def run_esrrf3d(img,correlation="AVG", framewindow=5, rollingoverlap=2, **kwargs):
    """
    Calculate the eSRRF3D temporal correlations for the given image.

    Args:
        img: The input 3D image.
        magnification_xy: The magnification factor for the x and y axes.
        magnification_z: The magnification factor for the z axis.
        radius: The radius for the xy plane.
        radius_z: The radius for the z axis.
        sensitivity: The sensitivity for the calculation.
        doIntensityWeighting: Whether to perform intensity weighting.
        run_type: The type of the run.
        keep_gradients: Whether to keep the gradients.
        keep_interpolated: Whether to keep the interpolated values.
        correlation: The type of correlation to use.
        framewindow: The window size for frame.
        rollingoverlap: The overlap size for rolling.

    Returns:
        The calculated eSRRF3D temporal correlations.
    """
    esrrf_calculator = eSRRF3D()
    return calculate_eSRRF3d_temporal_correlations(esrrf_calculator.run(img, **kwargs), correlation=correlation, framewindow=framewindow, rollingoverlap=rollingoverlap)