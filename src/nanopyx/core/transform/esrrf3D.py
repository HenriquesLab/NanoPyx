import numpy as np
from ._le_interpolation_catmull_rom import ShiftAndMagnify

class ESRRF3D:
    def __init__(self):
        pass

    def interpolate_image_3d(self, image, magnification_xy: int = 4, magnification_z: int = 4):
        interpolator = ShiftAndMagnify()

        xy_interpolated = interpolator.run(image, 0, 0, magnification_xy, magnification_xy)

        xyz_interpolated = interpolator.run(np.transpose(xy_interpolated, axes=[1, 0, 2]).copy(), 0, 0, magnification_z, 1)

        return np.transpose(xyz_interpolated, axes=[1, 0, 2]).copy()

    def run(self, image, magnification_xy: int = 4, magnification_z: int = 4, radius: int = 4, sentivity: int = 1, doIntensityWeighting: bool = True):

        interpolated_image = self.interpolate_image_3d(
            image,
            magnification_xy=magnification_xy,
            magnification_z=magnification_z
            )

        output = interpolated_image

        return output
