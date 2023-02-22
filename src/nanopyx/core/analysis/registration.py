
import numpy as np

from .ccm.estimate_shift import GetMaxOptimizer
from .ccm.ccm import calculate_ccm_cartesian, calculate_ccm_polar, calculate_ccm_logpolar

from ..transform.interpolation.catmull_rom import Interpolator

class Registration:

    def __init__(self, image, ref_image):
        """
        Register an image against a reference image
        :param image: 2D array to register
        :param ref_image: 2D array to use as reference
        """
        self.image = image
        self.ref_image = ref_image

        self.w = image.shape[1]
        self.h = image.shape[0]

        self.wref = ref_image.shape[1]
        self.href = ref_image.shape[0]


    def register(self, translation:bool, rotation:bool, scaling:bool):
        """
        Register an image against its reference according to the possible transformations
        :param translation: bool defining translation registration
        :param rotation: bool defining rotation registration
        :param scaling: bool defining rotation registration
        """
        
        if translation and (not rotation and not scaling):
            print("Looking for pure translation...", flush=True)

            ccm = calculate_ccm_cartesian(self.image, self.ref_image)
            shifts = self.calculate_peak(ccm)

            y_shift = (self.h/2.0 - shifts[0] - 1)
            x_shift = (self.w/2.0 - shifts[1] - 1)

            translated = Interpolator(self.image).shift(x_shift,y_shift)

            print((y_shift, x_shift))

            return translated

        elif rotation and (not translation and not scaling):

            print("Looking for pure rotation...", flush=True)
            
            ccm = calculate_ccm_polar(self.image, self.ref_image)
            shifts = self.calculate_peak(ccm)

            angle = -np.deg2rad((180-shifts[0] - 1))

            print(angle)

            rotated = Interpolator(self.image).rotate(angle)

            return rotated

        elif scaling and (not translation and not rotation):

            print("Looking for pure scaling...", flush=True)

            ccm = calculate_ccm_logpolar(self.image, self.ref_image)
            shifts = self.calculate_peak(ccm)
            
            radius = np.hypot(self.w/2, self.h/2)
            log_translation = (radius/2 - shifts[1] - 1) * np.log(radius) / radius
            scaling = np.exp(log_translation)
            
            print(scaling)

            scaled = Interpolator(self.image).scale_xy(scaling,scaling)

            return scaled
        
        if scaling and rotation and not translation:

            print("Looking for scaling and rotation assuming NO translation...", flush=True)

            ccm = calculate_ccm_logpolar(self.image, self.ref_image)
            shifts = self.calculate_peak(ccm)
            
            radius = np.hypot(self.w/2, self.h/2)
            log_translation = (radius/2 - shifts[1] - 1) * np.log(radius) / radius
            scaling = np.exp(log_translation)
            
            angle = -np.deg2rad((180-shifts[0] - 1))
            
            print(angle, scaling)

            scaled = Interpolator(self.image).scale_xy(scaling,scaling)
            rotated = Interpolator(scaled).rotate(angle)

            return rotated

        else:
            print("Looking for rotation, scaling and translation", flush=True)

            return None

    def calculate_peak(self, ccm:np.ndarray):
        optimizer = GetMaxOptimizer(ccm)
        return optimizer.get_max()

