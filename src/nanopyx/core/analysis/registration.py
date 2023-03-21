
from functools import cache

import numpy as np

from skimage.filters import window

from .estimate_shift import GetMaxOptimizer
from .ccm_helper_functions import make_even_square
from .ccm import calculate_slice_ccm
from ..transform.interpolation_catmull_rom import Interpolator

class Registration:

    def __init__(self, image:np.ndarray, ref_image:np.ndarray):
        """
        Register an image against a reference image
        :param image: 2D array to register
        :param ref_image: 2D array to use as reference
        """

        assert image.ndim == 2, "Image must be 2D"
        assert ref_image.ndim == 2, "Image must be 2D"

        self.image = make_even_square(image[np.newaxis,:,:].astype(np.float32))[0,:,:]
        self.ref_image = make_even_square(ref_image[np.newaxis,:,:].astype(np.float32))[0,:,:]

        self.original_dtype = image.dtype

        self.w = self.image.shape[1]
        self.h = self.image.shape[0]
        self.wref = self.ref_image.shape[1] 
        self.href = self.ref_image.shape[0]

        self.reg_result = {}
    
    
    def translation(self):
        """
        Registers the images considering only translation
        :return: registered image
        """

        shifts, max_sim = self.phase_correlation(self.ref_image, self.image)

        # The size of the CCM array is the same as the image
        y_shift = (self.h/2.0 - shifts[0])
        x_shift = (self.w/2.0 - shifts[1])

        translated = Interpolator(self.image).shift(x_shift,y_shift).astype(self.original_dtype)

        self.reg_result = {'Image':translated, 'Translation':(y_shift,x_shift), 'Scaling':None, 'Rotation':None, 'Max_Sim':max_sim}

        return translated
    

    def scaled_rotation(self):
        """
        Registers the images considering only rotation and isotropic scaling 
        """
        
        lpolar_image = Interpolator(self.image).polar(scale='log')
        lpolar_ref_image = Interpolator(self.ref_image).polar(scale='log')
        shifts, max_sim = self.phase_correlation(lpolar_ref_image, lpolar_image)

        # Size of the polar transform is always (360,maxradius)
        h = 360
        w = np.hypot(self.w/2, self.h/2)

        angle = -np.deg2rad((h/2 - shifts[0]))
        log_translation = (w/2-shifts[1]) * np.log(w) / w
        scale = np.exp(log_translation)

        scaled = Interpolator(self.image).scale_xy(scale, scale)
        rotated = Interpolator(scaled).rotate(angle).astype(self.original_dtype)

        self.reg_result = {'Image':rotated, 'Translation':None, 'Scaling':scaled, 'Rotation':angle, 'Max_Sim':max_sim}

        return rotated

    def scaling_rotation_translation(self):
        """
        Registers the images considering rotation, isotropic scaling and translation
        Based upon:
                An FFT-Based Technique for Translation,Rotation, 
                and Scale-Invariant Image Registration 
                B. Srinivasa Reddy and B. N. Chatterji
        """
        # Step 0: Prepare some heavily used vars
        h = 360
        w = np.hypot(self.w/2, self.h/2)
        highpass_filter = self.highpass_filter((self.h,self.w))

        # Step 1: Prep the reference image for iteration
        windowed_ref_image = self.ref_image * window('hann', self.ref_image.shape)
        freq_ref_image = np.abs(np.fft.fftshift(np.fft.fft2(windowed_ref_image)) * highpass_filter).astype(np.float32)
        lpolar_ref_image = Interpolator(freq_ref_image).polar('log')

        # Step 2: Iterate to find scale and angle
        total_angle = 0
        total_scale = 1
        iter_image = self.image

        for iter in range(10): 
            windowed_image = iter_image * window('hann', iter_image.shape)
            f_image = np.abs(np.fft.fftshift(np.fft.fft2(windowed_image))*highpass_filter).astype(np.float32)
            lpolar_image = Interpolator(f_image).polar('log')

            shifts, max_sim_1 = self.phase_correlation(lpolar_ref_image, lpolar_image)

            angle = -np.deg2rad((h/2 - shifts[0])) 
            log_translation = (w/2-shifts[1]) * np.log(w) / w
            scale = np.exp(-1*log_translation) # NEGATIVE SIGN HERE IS MANDATORY

            total_angle += angle
            total_scale *= scale

            scaled = Interpolator(self.image).scale_xy(total_scale, total_scale)
            rotated = Interpolator(scaled).rotate(total_angle)
            iter_image = rotated
     
        # Step 3: Find translation AND the correct angle
        # As outlined in the ref, in the frequency space 180-angle or angle have the same effect so there is ambiguity
        # We test for both options and choose the one with the highest peak in the ccm

        final_scaled_img = Interpolator(self.image).scale_xy(total_scale, total_scale)

        final_option_1 = Interpolator(final_scaled_img).rotate(total_angle)
        final_option_2 = Interpolator(final_scaled_img).rotate(np.pi+total_angle)

        shifts_1, sim_1 = self.phase_correlation(self.ref_image, final_option_1)
        shifts_2, sim_2 = self.phase_correlation(self.ref_image, final_option_2)

        if sim_1 > sim_2:
            final_iter_image = final_option_1
            shifts = shifts_1
            max_sim_2 = sim_1
        else:
            final_iter_image = final_option_2
            shifts = shifts_2
            max_sim_2 = sim_2
            
        y_shift = (self.h/2.0 - shifts[0])
        x_shift = (self.w/2.0 - shifts[1])

        translated = Interpolator(final_iter_image).shift(x_shift,y_shift).astype(self.original_dtype)

        self.reg_result = {'Image':translated, 'Translation':(y_shift,x_shift), 'Scaling':total_scale, 'Rotation':total_angle, 'Max_Sim':(max_sim_1,max_sim_2)}

        return translated

    @staticmethod
    def phase_correlation(im1:np.ndarray, im2:np.ndarray)->tuple:
        """
        Perform phase correlation between two images and return the shift that maximizes the overlap between the images
        :param im1: 2D array of np.float32
        :param im2: 2D array of np.float32
        :return: coordinate tuple of the maximum point of the ccm and max value of the ccm
        """

        ccm = calculate_slice_ccm(im1, im2)
        optimizer = GetMaxOptimizer(ccm)
        shifts = optimizer.get_max()
        maxsim = -optimizer.get_interpolated_px_value(shifts)

        return shifts, maxsim

    @staticmethod
    @cache
    def highpass_filter(shape)->np.ndarray:

        n_row = shape[0]
        n_col = shape[1]
        row_freq_arr = np.fft.fftshift(np.fft.fftfreq(n_row))
        col_freq_arr = np.fft.fftshift(np.fft.fftfreq(n_col))
        row_f,col_f = np.meshgrid(row_freq_arr, col_freq_arr, indexing='ij')
        X = np.cos(np.pi*row_f) * np.cos(np.pi*col_f) 
        H = (1-X)*(2-X)

        return H.astype(np.float32)