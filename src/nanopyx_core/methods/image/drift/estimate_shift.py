import numpy as np
from scipy.interpolate import griddata, interp2d
from scipy.optimize import minimize

from ...utils.timeit import timeit


# interp2d is faster than griddata approach
class GetMaxOptimizer(object):

    def __init__(self, slice_ccm) -> None:
        self.slice_ccm = slice_ccm
        self.w = slice_ccm.shape[1]
        self.h = slice_ccm.shape[0]
        self.interpolator = interp2d([y for y in range(self.h)], [x for x in range(self.w)], self.slice_ccm.reshape((self.w * self.h)), kind="cubic")
        
    def get_interpolated_px_value(self, coords):
        points = [(y, x) for y in range(self.slice_ccm.shape[0]) for x in range(self.slice_ccm.shape[1])]
        values = self.slice_ccm.reshape((self.slice_ccm.shape[0] * self.slice_ccm.shape[1]))

        return -griddata(points, values, coords, method="cubic")

    def get_interpolated_px_value_interp2d(self, coords):
        return -self.interpolator(coords[0], coords[1])

    def get_max(self):
        y_max, x_max = np.unravel_index(self.slice_ccm.argmax(), self.slice_ccm.shape)
        minimizer = minimize(self.get_interpolated_px_value_interp2d, (y_max, x_max), method="Nelder-Mead", options={"maxiter": 1000})
        return minimizer.x

