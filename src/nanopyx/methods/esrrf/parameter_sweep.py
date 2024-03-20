import numpy as np
from ...core.analysis.parameter_sweep import ParameterSweep


def run_esrrf_parameter_sweep(img: np.ndarray, magnification: int = 2, sensitivities: list = [1, 2], radii: list = [1, 1.5], temporal_correlation: str = "AVG"):
    """
    Conducts a parameter sweep for enhanced Super-Resolution Radial Fluctuations (eSRRF) analysis on an image.

    Parameters
    ----------
    img : np.ndarray
        The input image as a 2D numpy array.
    magnification : int, optional
        The magnification factor to be applied during the ESRRF analysis. Default is 2.
    sensitivities : list of int, optional
        A list of sensitivity values to be used in the parameter sweep. Default is [1, 2].
    radii : list of float, optional
        A list of radii values to be used in the parameter sweep. Default is [1, 1.5].
    temporal_correlation : str, optional
        The method of temporal correlation to be used. Default is "AVG".

    Returns
    -------
    np.ndarray
        An array with QnR value for each combination of sensitivities and radii. Indices of dimension 0 corresponds to indices of sensitivities list and dimension 1 to radii list.
        Suggestion: sensitivity_index, radius_index = np.argmax(run_esrrf_parameter_sweep())
        Optimal parameters will then correspond to: sensitivities[sensitivity_index] and radii[radius_index]

    Notes
    -----
    This function performs a parameter sweep for the ESRRF method, which is used for super-resolution imaging analysis. It varies sensitivity and radius to find optimal settings for image enhancement.
    """
    ps = ParameterSweep()
    return ps.run(img, magnification, sensitivity_array=sensitivities, radius_array=radii, temporal_correlation=temporal_correlation)