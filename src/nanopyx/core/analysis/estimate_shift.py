import numpy as np
from scipy.optimize import minimize

from ..transform._interpolation import cr_interpolate


class GetMaxOptimizer(object):
    """
    Class GetMaxOptimizer, used to extract the maximum value from a cross correlation matrix with subpixel precision.
    """

    def __init__(self, slice_ccm) -> None:
        """
        Creates an instance of GetMaxOptimizer.
        :param slice_ccm: numpy array with shape (y, x); ccm from which to extract the maximum value with subpixel
        precision.
        """
        self.slice_ccm = slice_ccm

    def get_interpolated_px_value(self, coords):
        """
        Method to be used for calculating the interpolated values of cross correlation matrices.
        :param coords: tuple of coordinates.
        :return: float; value of cross correlation matrix at given coordinates.
        For minimizer reasons -> negatives values become positive and positive become negative.
        """
        return -cr_interpolate(self.slice_ccm, coords[0], coords[1])

    def get_max(self):
        """
        Method used to calculate the maximum value and corresponding coordinates of a ccm. Uses a minimizer approach.
        :return: tuple; coordinates of maximum value of ccm with subpixel precision
        """
        y_max, x_max = np.unravel_index(self.slice_ccm.argmax(), self.slice_ccm.shape)
        minimizer = minimize(
            self.get_interpolated_px_value, (y_max, x_max), method="Nelder-Mead", options={"maxiter": 1000}
        )
        return minimizer.x
