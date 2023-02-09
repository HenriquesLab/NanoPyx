import math
import numpy as np
import pandas as pd
from math import pi
from skimage.draw import circle_perimeter
from skimage.exposure import rescale_intensity
from ..transform.padding import pad_w_zeros_2d
from ..transform.interpolation.catmull_rom import interpolate
from ..analysis.ccm.ccm import calculate_ccm_from_ref
from ..analysis.ccm.helper_functions import check_even_square, make_even_square

from matplotlib import pyplot as plt

class FIRECalculator(object):
    
    def __init__(self):
        self.fire = None
        self.perimeter_sampling_factor = 1
        self.frc_curve = None
        self.threshold_curve = None
        self.intersections = None
        
    def calculate_fire_number(self, img_1, img_2, method="Fixed 1/7"):
        self.calculate_frc_curve(img_1, img_2)
        self.calculate_threshold_curve(method=method)
        self.get_intersections()
        
        if self.intersections is not None and self.intersections.shape[0] != 0:
            spatial_frequency = self.get_correct_intersections(method=method)
            self.fire = 2 * (self.frc_curve.shape[0] +1) / spatial_frequency

        return self.fire

    # change this to use circle perimeter from skimage
    def _samples_at_radius(self, image, radius):
        center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
        return image[circle_perimeter(center_y, center_x, radius)]
        
    def calculate_frc_curve(self, img_1, img_2):
        max_width = max(img_1.shape[1], img_2.shape[1])
        max_height = max(img_1.shape[0], img_2.shape[0])
        
        img_1 = pad_w_zeros_2d(img_1, max_height, max_width).astype(np.float32)
        img_2 = pad_w_zeros_2d(img_2, max_height, max_width).astype(np.float32)
        
        ccm = calculate_ccm_from_ref(img_1.reshape((1, img_1.shape[0], img_1.shape[1])), img_2)[0]

        # Calculate the 1D cross-correlation curve
        cr = np.abs(ccm)
        
        size = cr.shape[0]

        # Calculate the radii of the samples
        max_radius = int(max(img_1.shape[0], img_1.shape[1])/2) - 1
        radii = np.arange(1, max_radius, dtype=int)
        #radii *= self.perimeter_sampling_factor * pi

        # Calculate the FRC curve
        frc = np.zeros((len(radii)+1, 3))
        frc[0][0] = 0
        frc[0][1] = 1
        frc[0][2] = 1
        for i, radius in enumerate(radii):
            samples = self._samples_at_radius(cr, radius)
            frc[i+1][0] = radius
            frc[i+1][1] = np.mean(samples)
            frc[i+1][2] = len(samples)
        self.frc_curve = frc
    
    def calculate_threshold_curve(self, method="Fixed 1/7"):
        threshold = np.zeros(self.frc_curve.shape[0])
        for i in range(threshold.shape[0]):
            if method == "Half-bit":
                threshold[i] = (0.2071 * math.sqrt(self.frc_curve[i][2]) + 1.9102) / (
                        1.2071 * math.sqrt(self.frc_curve[i][2]) + 0.9102
                )
            elif method == "Three sigma":
                threshold[i] = 3.0 / math.sqrt(self.frc_curve[i][2] / 2.0)
            else:
                threshold[i] = 0.1428
        self.threshold_curve = threshold
    
    def get_intersections(self):
        frc_curve = self.frc_curve
        threshold_curve = self.threshold_curve
        
        if len(frc_curve) != len(threshold_curve):
            print(
                "Error: Unable to calculate FRC curve intersections due to input length mismatch."
            )
            return None
        intersections = np.zeros(len(frc_curve) - 1)
        count = 0
        for i in range(1, len(frc_curve)):
            y1 = frc_curve[i - 1][1]
            y2 = frc_curve[i][1]
            y3 = threshold_curve[i - 1]
            y4 = threshold_curve[i]
            if not ((y3 >= y1 and y4 < y2) or (y1 >= y3 and y2 < y4)):
                continue
            x1 = frc_curve[i - 1][0]
            x2 = frc_curve[i][0]
            x3 = x1
            x4 = x2
            x1_x2 = x1 - x2
            x3_x4 = x3 - x4
            y1_y2 = y1 - y2
            y3_y4 = y3 - y4
            if x1_x2 * y3_y4 - y1_y2 * x3_x4 == 0:
                if y1 == y3:
                    intersections[count] = x1
                    count += 1
            else:
                px = ((x1 * y2 - y1 * x2) * x3_x4 - x1_x2 * (x3 * y4 - y3 * x4)) / (
                    x1_x2 * y3_y4 - y1_y2 * x3_x4
                )
                if px >= x1 and px < x2:
                    intersections[count] = px
                    count += 1

        self.intersections = np.copy(intersections[:count])
        
    def get_correct_intersections(self, method="Fixed 1/7"):
        if self.intersections is None or self.intersections.shape[0] == 0:
            return 0
        if method == "Fixed 1/7":
            return self.intersections[0]
        if self.intersections.shape[0] > 1:
            return self.intersections[1]
        
        return self.intersections[0]


    
def create_distance_map(img):
    distance_map = np.zeros(np.shape(img))
    center = np.floor(np.array(img.shape)/2)
    for x_i in range(0,np.shape(img)[0]):
        for y_i in range(0,np.shape(img)[1]):
            distance_map[x_i,y_i] = np.round(np.sqrt((x_i-center[0])**2+(y_i-center[1])**2))
        
    return distance_map

def get_intersections(frc_curve, threshold_curve):
    
    if len(frc_curve) != len(threshold_curve):
        print(
            "Error: Unable to calculate FRC curve intersections due to input length mismatch."
        )
        return None
    intersections = np.empty((len(frc_curve) - 1))
    count = 0
    for i in range(1, len(frc_curve)):
        y1 = frc_curve[i - 1][1]
        y2 = frc_curve[i][1]
        y3 = threshold_curve[i - 1]
        y4 = threshold_curve[i]
        if not ((y3 >= y1 and y4 < y2) or (y1 >= y3 and y2 < y4)):
            continue
        x1 = frc_curve[i - 1][0]
        x2 = frc_curve[i][0]
        x3 = x1
        x4 = x2
        x1_x2 = x1 - x2
        x3_x4 = x3 - x4
        y1_y2 = y1 - y2
        y3_y4 = y3 - y4
        if x1_x2 * y3_y4 - y1_y2 * x3_x4 == 0:
            if y1 == y3:
                intersections[count] = x1
                count += 1
        else:
            px = ((x1 * y2 - y1 * x2) * x3_x4 - x1_x2 * (x3 * y4 - y3 * x4)) / (
                x1_x2 * y3_y4 - y1_y2 * x3_x4
            )
            if px >= x1 and px < x2:
                intersections[count] = px
                count += 1

    return np.copy(intersections[:count])


class FRCCalculator(object):
    
    def __init__(self):
        self.use_half_circle = False
        self.perimeter_sampling_factor = 1
        self.img_1 = None
        self.img_2 = None
        self.fft_1 = None
        self.fft_2 = None
        self.ccm = None
        self.numerator = None
        self.abs_fft_1 = None
        self.abs_fft_2 = None
        self.frc_curve = None
        self.threshold_curve = None
        self.intersections = None
        self.fire_number = None
        
    def compute_FRC_values(self, data_a1, data_b1, data_a2, data_b2):
        
        self.numerator = np.empty(data_a1.shape)
        self.abs_fft_1 = np.empty(data_a1.shape)
        self.abs_fft_2 = np.empty(data_a1.shape)
        
        for idx in range(0, data_a1.shape[0]):
            
            a1i = data_a1[idx]
            a2i = data_a2[idx]
            b1i = data_b1[idx]
            b2i = data_b2[idx]
            
            self.numerator[idx] = (a1i * a2i + b1i * b2i)**2 + (b1i * b2i - a1i * a2i)**2
            self.abs_fft_1[idx] = a1i * a1i + b1i * b1i
            self.abs_fft_2[idx] = a2i * a2i + b2i * b2i
            
        
    def get_sine(self, angle, cos_a):
        if angle > math.pi:
            return math.sqrt(1 - (cos_a * cos_a)) * -1
        else:
            return math.sqrt(1 - (cos_a * cos_a)) * 1
        
    def get_interpolated_value_ccm(self, x, y, size):
        images = [self.ccm]
        x_base = int(x)
        y_base = int(y)
        x_fraction = x - x_base
        y_fraction = y - y_base
        if x_fraction < 0.0:
            x_fraction = 0.0
        if y_fraction < 0.0:
            y_fraction = 0.0

        lower_left_i = y_base * size + x_base
        lower_right_i = lower_left_i + 1
        upper_left_i = lower_left_i + size
        upper_right_i = upper_left_i + 1

        nImages = 3
        values = np.empty((nImages))
        for i in range(nImages):
            image = images[i]
            lower_left_v = image[lower_left_i]
            lower_right_v = image[lower_right_i]
            upper_right_v = image[upper_left_i]
            upper_left_v = image[upper_right_i]

            upper_average = upper_left_v + x_fraction * (upper_right_v - upper_left_v)
            lower_average = lower_left_v + x_fraction * (lower_right_v - lower_left_v)
            values[i] = lower_average + y_fraction * (upper_average - lower_average)
        return values[0]
    
    def get_interpolated_value(self, x, y, size):
        images = [self.numerator, self.abs_fft_1, self.abs_fft_2]
        x_base = int(x)
        y_base = int(y)
        x_fraction = x - x_base
        y_fraction = y - y_base
        if x_fraction < 0.0:
            x_fraction = 0.0
        if y_fraction < 0.0:
            y_fraction = 0.0

        lower_left_i = y_base * size + x_base
        lower_right_i = lower_left_i + 1
        upper_left_i = lower_left_i + size
        upper_right_i = upper_left_i + 1

        nImages = 3
        values = np.empty((nImages))
        for i in range(nImages):
            image = images[i]
            lower_left_v = image[lower_left_i]
            lower_right_v = image[lower_right_i]
            upper_right_v = image[upper_left_i]
            upper_left_v = image[upper_right_i]

            upper_average = upper_left_v + x_fraction * (upper_right_v - upper_left_v)
            lower_average = lower_left_v + x_fraction * (lower_right_v - lower_left_v)
            values[i] = lower_average + y_fraction * (upper_average - lower_average)
        return values

    def calculate_FRC_curve(self, img_1, img_2):
        
        img_1 = img_1.astype(np.float32)
        img_2 = img_2.astype(np.float32)
        
        if not check_even_square(img_1.reshape(1, img_1.shape[0], img_1.shape[1])):
            img_1 = make_even_square(img_1.reshape(1, img_1.shape[0], img_1.shape[1]))[0]
        if not check_even_square(img_2.reshape(1, img_2.shape[0], img_2.shape[1])):
            img_2 = make_even_square(img_2.reshape(1, img_2.shape[0], img_2.shape[1]))[0]
        
        max_w = max(img_1.shape[1], img_2.shape[1])
        max_h = max(img_1.shape[0], img_2.shape[0])
        
        img_1 = pad_w_zeros_2d(img_1, max_h, max_w)
        img_2 = pad_w_zeros_2d(img_2, max_h, max_w)
        
        fft_1 = np.fft.fft2(img_1)
        fft_2 = np.fft.fft2(img_2)
        
        self.ccm = np.array(calculate_ccm_from_ref(img_2.reshape(1, img_2.shape[0], img_2.shape[1]), img_1)[0], dtype=np.float32)
        
        size = fft_1.shape[0]
        centre = size / 2
        
        data_a1 = fft_1.real.ravel()
        data_b1 = fft_1.imag.ravel()
        data_a2 = fft_2.real.ravel()
        data_b2 = fft_2.imag.ravel()
        
        self.compute_FRC_values(data_a1, data_b1, data_a2, data_b2)
        
        max_r = int(centre - 1)
        results = np.empty((max_r, 3))
        results[0] = [0, 1, 1]
        
        if self.use_half_circle:
            limit = math.pi
        else:
            limit = math.pi * 2
        
        #FRC Curve result = [radius, correlation, n_samples]
        for radius in range(1, max_r):
            sum0, sum1, sum2 = 0, 0, 0
            angle_step = 1 / (self.perimeter_sampling_factor * radius)
            num_sum = 0
            for angle in np.arange(0, limit, angle_step):
                cos_a = math.cos(angle)
                sin_a = self.get_sine(angle, cos_a)
                x = centre + radius * cos_a
                y = centre + radius * sin_a
                value = interpolate(self.ccm, x, y)
                
            results[radius] = [radius, value, num_sum]
            
        self.frc_curve = results

    def calculate_threshold_curve(self, method="1/7"):
        
        threshold = np.empty((self.frc_curve.shape[0]))
        nr = 1
        sigma = 0
        
        if method == "1/7":
            threshold.fill(1/7)
        
        self.threshold_curve = threshold
            
    def get_intersections(self):
        
        frc_curve = self.frc_curve
        threshold_curve = self.threshold_curve
        
        if len(frc_curve) != len(threshold_curve):
            print(
                "Error: Unable to calculate FRC curve intersections due to input length mismatch."
            )
            return None
        intersections = np.empty((len(frc_curve) - 1))
        count = 0
        for i in range(1, len(frc_curve)):
            y1 = frc_curve[i - 1][1]
            y2 = frc_curve[i][1]
            y3 = threshold_curve[i - 1]
            y4 = threshold_curve[i]
            if not ((y3 >= y1 and y4 < y2) or (y1 >= y3 and y2 < y4)):
                continue
            x1 = frc_curve[i - 1][0]
            x2 = frc_curve[i][0]
            x3 = x1
            x4 = x2
            x1_x2 = x1 - x2
            x3_x4 = x3 - x4
            y1_y2 = y1 - y2
            y3_y4 = y3 - y4
            if x1_x2 * y3_y4 - y1_y2 * x3_x4 == 0:
                if y1 == y3:
                    intersections[count] = x1
                    count += 1
            else:
                px = ((x1 * y2 - y1 * x2) * x3_x4 - x1_x2 * (x3 * y4 - y3 * x4)) / (
                    x1_x2 * y3_y4 - y1_y2 * x3_x4
                )
                if px >= x1 and px < x2:
                    intersections[count] = px
                    count += 1

        self.intersections = np.copy(intersections[:count])

    def get_correct_intersections(self, method="Fixed 1/7"):
        if self.intersections is None or self.intersections.shape[0] == 0:
            return 0
        if method == "Fixed 1/7":
            return self.intersections[0]
        if self.intersections.shape[0] > 1:
            return self.intersections[1]
        
        return self.intersections[0]

    def calculate_FIRE_number(self, img_1, img_2, method="1/7"):
        self.calculate_FRC_curve(img_1, img_2)
        self.calculate_threshold_curve(method=method)
        self.get_intersections()
        intersection = self.get_correct_intersections(method=method)
        self.fire_number = self.ccm.shape[0] / intersection
        
        return self.fire_number