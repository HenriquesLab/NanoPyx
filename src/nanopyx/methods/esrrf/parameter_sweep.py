import numpy as np
from matplotlib import pyplot as plt
from ...core.analysis.parameter_sweep import ParameterSweep


def run_esrrf_parameter_sweep(
    img: np.ndarray,
    magnification: int = 2,
    sensitivities: list = [1, 2],
    radii: list = [1, 1.5],
    temporal_correlation: str = "AVG",
    use_decorr: bool = False,
    plot_sweep=False,
    return_qnr=False,
    n_frames=None,
):
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
    return_qnr : bool, optional
        If True, return an array with QnR value for each combination of sensitivities and radii. 
        If False, return a tuple of optimal sensitivity and radius values.
    n_frames : int, optional
        if None, uses all frames, otherwise splits the data into batches with size = n_frames.


    Returns
    -------
    np.ndarray or tuple
        If return_qnr is True, returns an array with QnR value for each combination of sensitivities and radii. Indices of dimension 0 corresponds to indices of sensitivities list and dimension 1 to radii list.
        Suggestion: sensitivity_index, radius_index = np.argmax(run_esrrf_parameter_sweep())
        Optimal parameters will then correspond to: sensitivities[sensitivity_index] and radii[radius_index]
        If return_qnr is False, returns a tuple of optimal sensitivity and radius values.

    Notes
    -----
    This function performs a parameter sweep for the ESRRF method, which is used for super-resolution imaging analysis. It varies sensitivity and radius to find optimal settings for image enhancement.
    """
    ps = ParameterSweep()
    out = ps.run(
        img,
        magnification,
        sensitivity_array=sensitivities,
        radius_array=radii,
        temporal_correlation=temporal_correlation,
        use_decorr=use_decorr,
        n_frames=n_frames
    )

    if plot_sweep:
        fig, ax = plt.subplots()
        ax.imshow(out, cmap="plasma")
        ax.set_xticks(np.arange(len(radii)), labels=radii)
        ax.set_yticks(np.arange(len(sensitivities)), labels=sensitivities)
        print(range(len(sensitivities)), range(len(radii)))
        for i in range(len(sensitivities)):
            for j in range(len(radii)):
                ax.text(j, i, round(out[i, j], 2), ha="center", va="center", color="w")
        ax.set_title("Parameter Sweep QnR")
        ax.set_xlabel("Radii")
        ax.set_ylabel("Sensitivities")
        fig.tight_layout()
        plt.show()

    if return_qnr:
        return out
    else:
        sens_idx, rad_idx = np.unravel_index(np.argmax(out), out.shape)
        return sensitivities[sens_idx], radii[rad_idx]
