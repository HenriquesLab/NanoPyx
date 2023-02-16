import math
import numpy as np
import pandas as pd
from scipy.signal.windows import tukey
from matplotlib import pyplot as plt
from ..transform.padding import pad_w_zeros_2d
from ..analysis.ccm.helper_functions import check_even_square, make_even_square


class FIRECalculator(object):
    
    def __init__(self, pixel_size=1, units="pxs"):
        self.pixel_size = pixel_size
        self.units = units
        self.threshold = 1/7
        self.frc_curve = None
        self.perimeter_sampling_factor = 1
        self.use_half_circle = False
        self.threshold_curve = None
        self.intersections = None
        
    def get_sine(self, angle, cos_a):
        if angle > math.pi:
            return math.sqrt(1 - (cos_a * cos_a)) * -1
        else:
            return math.sqrt(1 - (cos_a * cos_a)) * 1
        
    def compute(self, images, data_a1, data_b1, data_a2, data_b2):
        
        for x_i in range(data_a1.shape[1]):
            for y_i in range(data_a1.shape[0]):
                a1 = data_a1[y_i, x_i]
                a2 = data_a2[y_i, x_i]
                b1 = data_b1[y_i, x_i]
                b2 = data_b2[y_i, x_i]
                
                images[0, y_i, x_i] = a1 * a2 + b1 * b2
                images[1, y_i, x_i] = a1 * a1 + b1 * b1
                images[2, y_i, x_i] = a2 * a2 + b2 * b2
        
    def get_interpolated_values(self, y, x, images, maxx):
        x_base = int(x)
        y_base = int(y)
        x_fraction = x - x_base
        y_fraction = y - y_base
        if x_fraction < 0:
            x_fraction = 0
        if y_fraction < 0:
            y_fraction = 0
            
        lower_left_index = y_base * maxx + x_base
        lower_right_index = lower_left_index + 1
        upper_left_index = lower_left_index + maxx
        upper_right_index = upper_left_index + 1
        
        values = np.empty((images.shape[0]))
        for i in range(images.shape[0]):
            image = images[i].ravel()
            lower_left = image[lower_left_index]
            lower_right = image[lower_right_index]
            upper_left = image[upper_left_index] #TODO upper indexes were swapped in og code make sure it's ok
            upper_right = image[upper_right_index]
            
            upper_average = upper_left + x_fraction * (upper_right - upper_left)
            lower_average = lower_left + x_fraction * (lower_right - lower_left)
            values[i] = lower_average + y_fraction * (upper_average - lower_average)
            
        return values
        
    def get_squared_tapered_image(self, img):
        taper_x = tukey(img.shape[1], alpha=0.25)
        taper_y = tukey(img.shape[0], alpha=0.25)  
        
        img_data = img.ravel()
        new_data = np.empty(img_data.shape, dtype=np.float32)
        
        max_y_1 = img.shape[0]
        max_x_1 = img.shape[1]
        old_width = img.shape[1]
        new_size = img.shape[0]

        for y_i in range(0, max_y_1):
            y_tmp = taper_y[y_i]
            
            i = y_i * old_width
            ii = y_i * new_size
            for x_i in range(0, max_x_1):
                new_data[ii] = img_data[i] * taper_x[x_i] * y_tmp
                i += 1
                ii += 1
                
        return new_data.reshape((img.shape[0], img.shape[1]))
        
    def calculate_frc_curve(self, img_1, img_2):
        
        max_width = img_1.shape[1]
        max_height = img_1.shape[0]
        
        img_1 = pad_w_zeros_2d(img_1, max_height, max_width)
        img_2 = pad_w_zeros_2d(img_2, max_height, max_width)
        
        if not check_even_square(np.array([img_1], dtype=np.float32)):
            img_1 = np.array(make_even_square(np.array([img_1], dtype=np.float32)), dtype=np.float32)
            img_2 = np.array(make_even_square(np.array([img_2], dtype=np.float32)), dtype=np.float32)
        
        img_1 = self.get_squared_tapered_image(img_1)
        img_2 = self.get_squared_tapered_image(img_2)
        
        fft_1 = np.fft.fftshift(np.fft.fft2(img_1))
        fft_2 = np.fft.fftshift(np.fft.fft2(img_2))
        
        size = fft_1.shape[0]
        self.field_of_view = size
        centre = int(size / 2)
        
        data_a1 = fft_1.real
        data_b1 = fft_1.imag
        data_a2 = fft_2.real
        data_b2 = fft_2.imag
        
        images = np.empty((3, data_a1.shape[0], data_a1.shape[1]))
        
        self.compute(images, data_a1, data_b1, data_a2, data_b2)
        
        radius = 1
        max = centre - 1
        
        results = np.empty((max, 3), dtype=np.float32)
        results[0] = [0, 1, 1]
        self.spatial_frequency = np.linspace(0, 1/(2*self.pixel_size), max)
        
        if self.use_half_circle:
            limit = math.pi
        else:
            limit = math.pi * 2
            
        for r in range(radius, max):
            sum_0 = 0
            sum_1 = 0
            sum_2 = 0
            
            angle_step = 1 / (self.perimeter_sampling_factor * r)
            n_sum = 0
            
            for angle in np.arange(0, limit, angle_step):
                cos_a = math.cos(angle)
                x = centre + r * cos_a
                y = centre + r * self.get_sine(angle, cos_a)
                values = self.get_interpolated_values(y, x, images, size)
                sum_0 += values[0]
                sum_1 += values[1]
                sum_2 += values[2]
                n_sum += 1
            results[r] = [self.spatial_frequency[r], sum_0 / math.sqrt(sum_1*sum_2), n_sum]
        self.frc_curve = results
        
    def calculate_threshold_curve(self):
        self.threshold_curve = np.empty((self.frc_curve.shape[0]))
        self.threshold_curve.fill(1/7)
        
    def interpolate_y(self, x1, y1, x2, y2, x):
        
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
        
        return m * x + c
        
    def get_intersections(self):
        frc_curve = self.frc_curve
        threshold_curve = self.threshold_curve
        
        if frc_curve.shape[0] != threshold_curve.shape[0]:
            print(
                "Error: Unable to calculate FRC curve intersections due to input length mismatch."
            )
            return None
        intersections = np.zeros((frc_curve.shape[0] - 1, 2))
        count = 0
        for i in range(1, frc_curve.shape[0]):
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
                    intersections[count] = [x1, y1]
                    count += 1
            else:
                px = ((x1 * y2 - y1 * x2) * x3_x4 - x1_x2 * (x3 * y4 - y3 * x4)) / (
                    x1_x2 * y3_y4 - y1_y2 * x3_x4
                )
                if px >= x1 and px < x2:
                    py = self.interpolate_y(x1, y3, x2, y4, px)
                    intersections[count] = [px, py]
                    count += 1
                else:
                    intersections[count] = [px, y1]
                    count += 1
        self.intersections = np.copy(intersections[:count])
        
    def calculate_fire_number(self, img_1, img_2):
        self.calculate_frc_curve(img_1, img_2)
        self.calculate_threshold_curve()
        self.get_intersections()
        
        if self.intersections.shape[0] > 0:
            self.fire_number = 1 / self.intersections[0, 0]
        else:
            self.fire_number = 0
        
        return self.fire_number

    def plot_frc_curve(self):
        plt.plot(self.spatial_frequency, self.frc_curve[:, 1])
        plt.axhline(1/7, color='r', linestyle='-')
        plt.xlabel('Spatial frequency [1/nm]')
        plt.ylabel('FRC')
        plt.title(f"FRC resolution: {np.round(self.fire_number, 1)} {self.units}");
        plt.show()

