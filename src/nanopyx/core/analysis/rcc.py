# REF: based on https://github.com/jungmannlab/picasso/blob/d867f561ffeafce752f37968a20698556d04dafb/picasso/imageprocess.py

import numpy as np
import lmfit
from tqdm import tqdm

from .ccm import calculate_ccm_from_ref
from .ccm_helper_functions import check_even_square, make_even_square

# TODO: fix max_shift parameter


def calculate_x_corr(im1: np.ndarray, im2: np.ndarray):
    """
    Calculate the cross-correlation between two images.

    Args:
        im1 (np.ndarray): The first image as a NumPy array.
        im2 (np.ndarray): The second image as a NumPy array.

    Returns:
        np.ndarray: The cross-correlation matrix as a NumPy array.
    """
    ccm = calculate_ccm_from_ref(np.array([im2]).astype(np.float32), im1.astype(np.float32))[0]

    return np.array(ccm)


def get_image_shift(im1: np.ndarray, im2: np.ndarray, box: int, max_shift: int = None):
    """
    Calculate the shift in coordinates between two input images.

    Parameters:
    - im1: A numpy array representing the first image.
    - im2: A numpy array representing the second image.
    - box: An integer specifying the size of the fitting region of interest (ROI) in pixels.
    - max_shift: An optional integer specifying the maximum allowed shift in pixels. If not provided, the entire image will be used for correlation.

    Returns:
    - A tuple of two integers representing the shift in the x and y coordinates, respectively. The shift is calculated as the difference between the centroid of the fitting ROI and the centroid of the correlation peak.

    Note:
    - If either of the input images is completely black (i.e., all pixels have a value of zero), the function will return (0, 0).
    - The correlation between the two images is computed using the `calculate_x_corr` function.
    - The correlation matrix is cropped based on the `max_shift` parameter. If the `max_shift` is positive, the correlation matrix is cropped symmetrically around the center of the image. If the `max_shift` is zero or negative, the entire correlation matrix is used.
    - The brightest pixel in the correlation matrix is identified, and a fitting ROI centered on this pixel is extracted. The size of the fitting ROI is specified by the `box` parameter.
    - The function fits a 2D Gaussian model to the values in the fitting ROI using the `lmfit` library. The parameters of the Gaussian model are then used to calculate the centroid coordinates.
    - The centroid coordinates are adjusted to account for the cropping of the correlation matrix and the size of the fitting ROI.
    - Finally, the calculated shift in coordinates is returned as a negative value to align the images.

    Example usage:
    shift_x, shift_y = get_image_shift(image1, image2, 10, max_shift=5)
    """
    if (np.sum(im1) == 0) or (np.sum(im2) == 0):
        return 0, 0

    # Compute image correlation
    x_corr = calculate_x_corr(im1, im2)

    # crop XCorr based on max_shift
    w, h = im1.shape
    if max_shift > 0:
        x_border = int((w - max_shift) / 2)
        y_border = int((h - max_shift) / 2)
        if x_border > 0:
            x_corr = x_corr[x_border:-x_border, :]
        else:
            x_border = 0
        if y_border > 0:
            x_corr = x_corr[:, y_border:-y_border]
        else:
            y_border = 0
    else:
        x_border = y_border = 0

    # A quarter of the fit ROI
    fit_box = int(box / 2)

    # A coordinate grid for the fitting ROI
    x, y = np.mgrid[-fit_box : fit_box + 1, -fit_box : fit_box + 1]

    # Find the brightest pixel and cut out the fit ROI
    x_max_xc, y_max_xc = np.unravel_index(x_corr.argmax(), x_corr.shape)
    fit_roi = x_corr[
        x_max_xc - fit_box : y_max_xc + fit_box + 1,
        y_max_xc - fit_box : y_max_xc + fit_box + 1,
    ]

    dimensions = fit_roi.shape

    if 0 in dimensions or dimensions[0] != dimensions[1]:
        xc, yc = 0, 0
    else:
        # The fit model
        def flat_2d_gaussian(a, xc, yc, s, b):
            A = a * np.exp(-0.5 * ((x - xc) ** 2 + (y - yc) ** 2) / s**2) + b
            return A.flatten()

        gaussian2d = lmfit.Model(flat_2d_gaussian, name="2D Gaussian", independent_vars=[])

        # Set up initial parameters and fit
        params = lmfit.Parameters()
        params.add("a", value=fit_roi.max(), vary=True, min=0)
        params.add("xc", value=0, vary=True)
        params.add("yc", value=0, vary=True)
        params.add("s", value=1, vary=True, min=0)
        params.add("b", value=fit_roi.min(), vary=True, min=0)
        tmp = fit_roi.flatten()
        results = gaussian2d.fit(tmp, params)

        # Get maximum coordinates and add offsets
        xc = results.best_values["xc"]
        yc = results.best_values["yc"]
        xc += y_border + x_max_xc
        yc += x_border + y_max_xc

        xc -= np.floor(w / 2)
        yc -= np.floor(h / 2)

    return -xc, -yc


def rcc(im_frames: np.ndarray, max_shift=None) -> tuple:
    """
    Calculate the relative shifts between a set of input images using the RCC (Relative Cross Correlation) algorithm.

    Parameters:
    - im_frames: A NumPy array representing a set of input images. The shape of the array is (n_frames, height, width).
    - max_shift: An optional parameter specifying the maximum shift allowed between image pairs. If not provided, the default value is None.

    Returns:
    - A tuple containing the minimized shifts in the x and y directions between all image pairs.

    References:
    - REF: https://github.com/yinawang28/RCC
    """
    if not check_even_square(im_frames.astype(np.float32)):
        im_frames = np.array(make_even_square(im_frames.astype(np.float32)))
    n_frames = im_frames.shape[0]
    shifts_x = np.zeros((n_frames, n_frames))
    shifts_y = np.zeros((n_frames, n_frames))
    n_pairs = int(n_frames * (n_frames - 1) / 2)
    flag = 0

    with tqdm(total=n_pairs, desc="Correlating image pairs", unit="pairs") as progress_bar:
        for i in range(n_frames - 1):
            for j in range(i + 1, n_frames):
                progress_bar.update()
                shifts_x[i, j], shifts_y[i, j] = get_image_shift(im_frames[i], im_frames[j], 5, max_shift)
                flag += 1

    return minimize_shifts(shifts_x, shifts_y)


def minimize_shifts(shifts_x, shifts_y, shifts_z=None):
    """
    Functions that minimizes from the input arrays.

    Parameters:
        shifts_x (ndarray): An array containing the shifts in the x direction.
        shifts_y (ndarray): An array containing the shifts in the y direction.
        shifts_z (ndarray, optional): An array containing the shifts in the z direction. Defaults to None.

    Returns:
        tuple: A tuple containing the shift values in the x, y, and z directions (if shifts_z is not None) or just the shift values in the x and y directions.
    """
    n_channels = shifts_x.shape[0]
    n_pairs = int(n_channels * (n_channels - 1) / 2)
    n_dims = 2 if shifts_z is None else 3
    rij = np.zeros((n_pairs, n_dims))
    A = np.zeros((n_pairs, n_channels - 1))
    flag = 0
    for i in range(n_channels - 1):
        for j in range(i + 1, n_channels):
            rij[flag, 0] = shifts_x[i, j]
            rij[flag, 1] = shifts_y[i, j]
            if n_dims == 3:
                rij[flag, 2] = shifts_z[i, j]
            A[flag, i:j] = 1
            flag += 1
    Dj = np.dot(np.linalg.pinv(A), rij)
    shift_x = np.insert(np.cumsum(Dj[:, 0]), 0, 0)
    shift_y = np.insert(np.cumsum(Dj[:, 1]), 0, 0)
    if n_dims == 2:
        return shift_x, shift_y
    else:
        shift_z = np.insert(np.cumsum(Dj[:, 2]), 0, 0)
        return shift_x, shift_y, shift_z
