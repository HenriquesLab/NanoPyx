import numpy as np
from matplotlib import pyplot as plt
from ...core.analysis.decorr import DecorrAnalysis
from ...core.analysis.frc import FIRECalculator


def calculate_frc(
    frame_1: np.ndarray, frame_2: np.ndarray, pixel_size: float = 1, units: str = "pixel", plot_frc_curve: bool = False
):
    """
    Calculate the Fourier Ring Correlation (FRC) between two images to estimate their resolution.

    Parameters
    ----------
    frame_1 : np.ndarray
        The first image frame as a 2D numpy array.
    frame_2 : np.ndarray
        The second image frame as a 2D numpy array.
    pixel_size : float, optional
        The physical size of a pixel in the specified units. Default is 1.
    units : str, optional
        The units of measurement for the pixel size (e.g., "pixel", "nm", "um"). Default is "pixel".
    plot_frc_curve : bool, optional
        If True, plots the FRC curve. Default is False.

    Returns
    -------
    float
        The resolution estimate based on the FRC calculation.

    Notes
    -----
    The Fourier Ring Correlation (FRC) is a method to estimate the resolution of images, particularly useful in microscopy. It compares the similarity of two images in the frequency domain, providing a measure of their resolution.
    """
    frc_calc = FIRECalculator(pixel_size=pixel_size, units=units)
    res = frc_calc.calculate_fire_number(frame_1, frame_2)

    if plot_frc_curve:
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        plt.imshow(frc_calc.plot_frc_curve())
        plt.show()

    return res


def calculate_decorr_analysis(
    frame: np.ndarray,
    rmin: float = 0,
    rmax: float = 1,
    n_r: int = 50,
    n_g: int = 10,
    pixel_size: float = 1,
    units: str = "pixel",
    roi: tuple = (0, 0, 0, 0),
    plot_decorr_analysis=False,
):
    """
    Perform decorrelation analysis on a given image frame to estimate its resolution.

    Parameters
    ----------
    frame : np.ndarray
        The image frame as a 2D numpy array.
    rmin : float, optional
        The minimum radius for decorrelation analysis. Default is 0.
    rmax : float, optional
        The maximum radius for decorrelation analysis. Default is 1.
    n_r : int, optional
        The number of radial divisions for analysis. Default is 50.
    n_g : int, optional
        The number of angular divisions for analysis. Default is 10.
    pixel_size : float, optional
        The physical size of a pixel in the specified units. Default is 1.
    units : str, optional
        The units of measurement for the pixel size (e.g., "pixel", "nm", "um"). Default is "pixel".
    roi : tuple, optional
        The region of interest in the format (x_min, y_min, x_max, y_max). Default is (0, 0, 0, 0).
    plot_decorr_analysis : bool, optional
        If True, plots the results of the decorrelation analysis. Default is False.

    Returns
    -------
    float
        The resolution estimate based on the decorrelation analysis.

    Notes
    -----
    Decorrelation analysis is a technique used to estimate the resolution of images by analyzing the decorrelation of intensity values over different spatial scales. It is particularly useful in microscopy and imaging where direct resolution measurement is challenging.
    """
    decorr_calc = DecorrAnalysis(pixel_size=pixel_size, units=units, rmin=rmin, rmax=rmax, n_r=n_r, n_g=n_g, roi=roi)
    decorr_calc.run_analysis(frame)

    if plot_decorr_analysis:
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        plt.imshow(decorr_calc.plot_results())
        plt.show()

    return decorr_calc.resolution
