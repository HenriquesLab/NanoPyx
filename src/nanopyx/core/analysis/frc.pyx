# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

import io
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal.windows import tukey
from ..transform.padding import pad_w_zeros_2d

cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt, fabs, pi, cos, fmin
from ..utils.math cimport _get_sine
from .ccm_helper_functions cimport _check_even_square, _make_even_square


cdef class FIRECalculator:

    # autogen_pxd: cdef float pixel_size, threshold
    # autogen_pxd: cdef str units
    # autogen_pxd: cdef float[:] threshold_curve
    # autogen_pxd: cdef float[:, :] frc_curve, intersections
    # autogen_pxd: cdef int field_of_view
    # autogen_pxd: cdef public float fire_number

    def __init__(self, pixel_size: float = 1, units: str = "pixel"):
        self.pixel_size = pixel_size
        self.units = units
        self.threshold = 1/7
        self.fire_number = 0
        self.field_of_view = 0

    cdef float[:, :] _get_squared_tapered_image(self, float[:, :] img):
        cdef float[:] taper_x = tukey(img.shape[1], alpha=0.25).astype(np.float32)
        cdef float[:] taper_y = tukey(img.shape[0], alpha=0.25).astype(np.float32)
        cdef float[:] img_data = np.ravel(img)
        cdef float[:] new_data = np.empty(img_data.shape[0], dtype=np.float32)

        cdef int max_y_1, max_x_1, old_width, new_size, y_i, x_i, i, ii
        cdef float y_tmp

        max_x_1 = img.shape[1]
        max_y_1 = img.shape[0]
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
        return np.reshape(new_data, (img.shape[0], img.shape[1]))

    cdef float _interpolate_y(self, float x1, float y1, float x2, float y2, float x):

        cdef float m = (y2 -y1) / (x2 - x1)
        cdef float c = y1 - m * x1

        return m * x + c

    cdef _compute(self, float[:, :] images, float[:] data_a1, float[:] data_b1, float[:] data_a2, float[:] data_b2):

        cdef int i
        cdef float a1, a2, b1, b2

        cdef int size = data_a1.shape[0]

        with nogil:
            for i in prange(size):
                a1 = data_a1[i]
                a2 = data_a2[i]
                b1 = data_b1[i]
                b2 = data_b2[i]
                
                images[0, i] = a1 * a2 + b1 * b2
                images[1, i] = a1 * a1 + b1 * b1
                images[2, i] = a2 * a2 + b2 * b2

    cdef float[:] _get_interpolated_values(self, float y, float x, float[:, :] images, int maxx) nogil:
        cdef int x_base = int(x)
        cdef int y_base = int(y)
        cdef float x_fraction = x - x_base
        cdef float y_fraction = y - y_base
        if x_fraction < 0:
            x_fraction = 0
        if y_fraction < 0:
            y_fraction = 0

        cdef int lower_left_index = y_base * maxx + x_base
        cdef int lower_right_index = lower_left_index + 1
        cdef int upper_left_index = lower_left_index + maxx
        cdef int upper_right_index = upper_left_index + 1

        cdef float[:] values
        with gil:
            values = np.zeros((images.shape[0]), dtype=np.float32)
        cdef float lower_left, lower_right, upper_left, upper_right, upper_average, lower_average

        cdef int i

        for i in prange(images.shape[0]):
            lower_left = images[i][lower_left_index]
            lower_right = images[i][lower_right_index]
            upper_left = images[i][upper_left_index]
            upper_right = images[i][upper_right_index]

            upper_average = upper_left + x_fraction * (upper_right - upper_left)
            lower_average = lower_left + x_fraction * (lower_right - lower_left)
            values[i] = lower_average + y_fraction * (upper_average - lower_average)

        return values

    cdef _get_smoothed_curve(self):
        cdef int window_l = <int>(0.0707*self.frc_curve.shape[0])
        cdef int poly_order = 3
        if poly_order >= window_l:
            window_l = poly_order + 1
        cdef float[:] smoothed_values = savgol_filter(self.frc_curve[:, 1], window_length=window_l, polyorder=poly_order)
        self.frc_curve[:, 1] = smoothed_values

    cdef _calculate_threshold_curve(self):
        cdef int curve_dims = self.frc_curve.shape[0]
        self.threshold_curve = np.full((curve_dims), 1.0/7.0, dtype=np.float32)

    cdef _calculate_frc_value(self, int centre, int size, float[:, :] images, float pixel_size):
        cdef int radius = 1
        cdef int max_r = centre - 1
        cdef float[:] spatial_frequency = np.linspace(0, 1/(2*pixel_size), max_r, dtype=np.float32)
        cdef float limit = pi * 2
        cdef float[:, :] results = np.zeros((max_r, 3), dtype=np.float32)
        results[0][0] = 0
        results[0][1] = 1
        results[0][2] = 1

        cdef int r, n_sum, i
        cdef float sum_0, sum_1, sum_2, angle_step, x, y, cos_a, angle
        cdef float[:] values, angles
        
        for r in range(radius, max_r):
            sum_0 = 0
            sum_1 = 0
            sum_2 = 0

            angle_step = 1.0/r
            n_sum = 0
            angles = np.arange(0, limit, angle_step, dtype=np.float32)
            for i in range(angles.shape[0]):
                angle = angles[i]
                cos_a = cos(angle)
                x = max_r + 1 + r * cos_a
                y = max_r + 1 + r * _get_sine(angle, cos_a)
                values = self._get_interpolated_values(y, x, images, size)
                sum_0 += values[0]
                sum_1 += values[1]
                sum_2 += values[2]
                n_sum += 1
            results[r][0] = spatial_frequency[r]
            results[r][1] = sum_0/sqrt(sum_1*sum_2)
            results[r][2] = n_sum

        return results

    def calculate_fft(self, img: np.ndarray):
        return np.fft.fftshift(np.fft.fft2(img))

    cdef _calculate_frc_curve(self, float[:, :] img1, float[:, :] img2):

        cdef float[:, :] img_1 = np.copy(img1)
        cdef float[:, :] img_2 = np.copy(img2)
        cdef int max_width = img_1.shape[1]
        cdef int max_height = img_1.shape[0]

        img_1 = pad_w_zeros_2d(np.array(img_1), max_height, max_width)
        img_2 = pad_w_zeros_2d(np.array(img_2), max_height, max_width)

        if not _check_even_square(np.array([img_1], dtype=np.float32)):
            img_1 = _make_even_square(np.array([img_1], dtype=np.float32))[0]
            img_2 = _make_even_square(np.array([img_2], dtype=np.float32))[0]

        img_1 = self._get_squared_tapered_image(img_1)
        img_2 = self._get_squared_tapered_image(img_2)
        
        cdef complex[:, :] fft_1 = self.calculate_fft(np.array(img_1, dtype=np.float32))
        cdef complex[:, :] fft_2 = self.calculate_fft(np.array(img_2, dtype=np.float32))
        
        cdef int size = fft_1.shape[0]
        self.field_of_view = size
        cdef int centre = size // 2
        cdef float[:] data_a1, data_a2, data_b1, data_b2
        cdef float[:, :] images
        data_a1 = np.real(fft_1).ravel().astype(np.float32)
        data_b1 = np.imag(fft_1).ravel().astype(np.float32)
        data_a2 = np.real(fft_2).ravel().astype(np.float32)
        data_b2 = np.imag(fft_2).ravel().astype(np.float32)

        images = np.zeros((3, data_a1.shape[0]), dtype=np.float32)
        self._compute(images, data_a1, data_b1, data_a2, data_b2)

        self.frc_curve = self._calculate_frc_value(centre, size, images, self.pixel_size)

    cdef _get_intersections(self):
        cdef float[:, :] frc_curve = np.copy(self.frc_curve)
        cdef float[:] threshold_curve = np.copy(self.threshold_curve)

        if frc_curve.shape[0] !=  threshold_curve.shape[0]:
            raise ValueError("FRC curve and threshold curve must have the same length")

        cdef float[:, :] intersections = np.zeros((frc_curve.shape[0], 2), dtype=np.float32)
        cdef int count = 0
        cdef int i
        cdef float y1, y2, y3, y4, x1, x2, x3, x4, x1_x2, x3_x4, y1_y2, y3_y4, px, py
        cdef float[:] arr

        for i in range(1, frc_curve.shape[0]):
            y1 = frc_curve[i-1][1]
            y2 = frc_curve[i][1]
            y3 = threshold_curve[i-1]
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
                    arr = np.array([x1, y1], dtype=np.float32)
                    intersections[count] = arr
                    count += 1
            else:
                px = ((x1 * y2 - y1 * x2) * x3_x4 - x1_x2 * (x3 * y4 - y3 * x4)) / (
                    x1_x2 * y3_y4 - y1_y2 * x3_x4
                )
                if px >= x1 and px < x2:
                    py = self._interpolate_y(x1, y3, x2, y4, px)
                    arr = np.array([px, py], dtype=np.float32)
                    intersections[count] = arr
                    count += 1
                else:
                    arr = np.array([px, y1], dtype=np.float32)
                    intersections[count] = arr
                    count += 1

        self.intersections = np.copy(intersections[:count])

    cdef _calculate_fire_number(self, float[:, :] img_1, float[:, :] img_2):
        self._calculate_frc_curve(img_1, img_2)
        self._get_smoothed_curve()
        self._calculate_threshold_curve()
        self._get_intersections()
        print(self.intersections[0, 0])

        if self.intersections.shape[0] > 0:
            self.fire_number = 1 / self.intersections[0, 0]
        else:
            self.fire_number = 0

        return self.fire_number

    def calculate_fire_number(self, img_1, img_2):
        return self._calculate_fire_number(img_1.astype(np.float32), img_2.astype(np.float32))
    
    def plot_frc_curve(self):
        """
        Returns the plot of the results of the analysis as a numpy array
        """
        fig, ax= plt.subplots(dpi=150)
        ax.plot(np.array(self.frc_curve[:, 0]), np.array(self.frc_curve[:, 1]))
        ax.axhline(y=1.0/7.0, color='r', linestyle='-')
        ax.set_xlabel(f'Spatial frequency [1/{self.units}]')
        ax.set_ylabel('FRC')
        ax.set_title(f"FRC resolution: {np.round(self.fire_number, 1)} {self.units}")
        with io.BytesIO() as buf:
            fig.savefig(buf, format="raw", dpi=150)
            buf.seek(0)
            data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))
        plt.close(fig)

        return im