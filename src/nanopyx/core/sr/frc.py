import math
import numpy as np
from math import pi
from skimage.draw import circle_perimeter
from ..transform.padding import pad_w_zeros_2d
from ..analysis.ccm.ccm import calculate_ccm_from_ref

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

