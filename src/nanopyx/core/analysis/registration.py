
import numpy as np

from skimage.filters import window

from .estimate_shift import GetMaxOptimizer
from .ccm import calculate_ccm_cartesian, calculate_ccm_polar, calculate_ccm_logpolar

from ..transform.interpolation_catmull_rom import Interpolator

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

            y_shift = (self.h/2.0 - shifts[0])
            x_shift = (self.w/2.0 - shifts[1])

            translated = Interpolator(self.image).shift(x_shift,y_shift)

            print((y_shift, x_shift))

            return translated

        elif rotation and (not translation and not scaling):

            print("Looking for pure rotation...", flush=True)
            
            ccm = calculate_ccm_polar(self.image, self.ref_image)
            shifts = self.calculate_peak(ccm)

            angle = -np.deg2rad((180-shifts[0]))

            print(angle)

            rotated = Interpolator(self.image).rotate(angle)

            return rotated

        elif scaling and (not translation and not rotation):

            print("Looking for pure scaling...", flush=True)

            ccm = calculate_ccm_logpolar(self.image, self.ref_image)
            shifts = self.calculate_peak(ccm)
            
            radius = np.hypot(self.w/2, self.h/2)
            log_translation = (radius/2 - shifts[1]) * np.log(radius) / radius
            scaling = np.exp(log_translation)
            
            print(scaling)

            scaled = Interpolator(self.image).scale_xy(scaling,scaling)

            return scaled
        
        if scaling and rotation and not translation:

            print("Looking for scaling and rotation assuming NO translation...", flush=True)

            ccm = calculate_ccm_logpolar(self.image, self.ref_image)
            shifts = self.calculate_peak(ccm)
            
            radius = np.hypot(self.w/2, self.h/2)
            log_translation = (radius/2 - shifts[1]) * np.log(radius) / radius
            scaling = np.exp(log_translation)
            
            angle = -np.deg2rad((180-shifts[0]))
            
            print(angle, scaling)

            scaled = Interpolator(self.image).scale_xy(scaling,scaling)
            rotated = Interpolator(scaled).rotate(angle)

            return rotated
        
        elif scaling and rotation and translation:

            print("Looking for scaling and rotation and translation...", flush=True)

            angle = 0
            scale = 1

                       
            # Highpass filtering as in:
            # An FFT-Based Technique for Translation,Rotation, 
            # and Scale-Invariant Image Registration 
            # B. Srinivasa Reddy and B. N. Chatterji
            n_row = self.h
            n_col = self.w
            row_freq_arr = np.fft.fftshift(np.fft.fftfreq(n_row))
            col_freq_arr = np.fft.fftshift(np.fft.fftfreq(n_col))
            row_f,col_f = np.meshgrid(row_freq_arr, col_freq_arr, indexing='ij')
            X = np.cos(np.pi*row_f) * np.cos(np.pi*col_f) 
            H = (1-X)*(2-X)
           
            iter_image = self.image
            windowed_ref_image = self.ref_image * window('hann', self.image.shape)
            f_ref_image = np.abs(np.fft.fftshift(np.fft.fft2(windowed_ref_image))*H)

            for iter in range(10):
                
                windowed_image = iter_image * window('hann', iter_image.shape)
                f_image = np.fft.fftshift(np.fft.fft2(windowed_image)) * H
                f_image = np.abs(f_image)
            
                ccm = calculate_ccm_logpolar(f_image.astype(np.float32),f_ref_image.astype(np.float32)) 
                shifts = self.calculate_peak(ccm)
            
                # Get translation in frequency domain ==> scaling 
                n_col = int(np.hypot(self.w/2, self.h/2))
                f_log_translation = (n_col/2 - shifts[1]) * np.log(n_col) / n_col
                f_scaling = np.exp(-1*f_log_translation) # NEGATIVE SIGN
            
                # Get angle in frequency domain ==> rotation
                f_angle = -np.deg2rad((180-shifts[0]))

                print(f_angle, f_scaling)
                angle += f_angle
                scale *= f_scaling
            
                scaled = Interpolator(self.image).scale_xy(scale,scale)
                rotated_scaled = Interpolator(scaled).rotate(angle)

                iter_image = rotated_scaled

            # Acquire translation
            ccm = calculate_ccm_cartesian(rotated_scaled, self.ref_image)
            shifts = self.calculate_peak(ccm)

            y_shift = (self.h/2.0 - shifts[0] - 1)
            x_shift = (self.w/2.0 - shifts[1] - 1)

            print(f"Angle: {np.rad2deg(angle):.2f}",flush=True)
            print(f"Scaling {scale:.2f}", flush=True)
            print(f"Shift (y,x): ({y_shift:.2f}, {x_shift:.2f})", flush=True)

            translated = Interpolator(rotated_scaled).shift(x_shift,y_shift)

            return translated

        else:
            print("Not implemented yet", flush=True)

            return None

    def calculate_peak(self, ccm:np.ndarray):
        optimizer = GetMaxOptimizer(ccm)
        return optimizer.get_max()

 