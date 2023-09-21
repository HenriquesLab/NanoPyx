# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

from math import cos, fabs, sqrt

import io
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from ..utils.timeit import timeit2

cimport numpy as np
from libc.math cimport ceil, exp, fmax, fmin, isnan, log, pow, round, sqrt

from cython.parallel import prange

from ..generate.mask cimport _get_circular_mask
from ..transform.edges cimport _apodize_edges
from ..transform.normalize cimport _normalizeFFT
from ..utils.array cimport _get_max, _get_min


cdef class DecorrAnalysis:

    # autogen_pxd: cdef float[:] d0, kc, a_g
    # autogen_pxd: cdef float[:, :] img, img_ref, d
    # autogen_pxd: cdef float rmin, rmin2, rmax, rmax2, kc0, a0, kc_gm, agm, kc_max, a_max, pixel_size
    # autogen_pxd: cdef public float resolution
    # autogen_pxd: cdef int n_r, n_g, x0, x1, y0, y1
    # autogen_pxd: cdef bint do_plot
    # autogen_pxd: cdef str units

    def __init__(self, rmin:float = 0, rmax:float = 1, n_r:int = 50, n_g:int =10, pixel_size: float = 1, units: str = "pixel", roi: tuple = (0, 0, 0, 0), do_plot: bool = False):
        """_summary_

        Args:
            rmin (float, optional): Minimum radius [0,rMax] (normalized frequencies) used for decorrelation analysis. Defaults to 0.
            rmax (float, optional): Maximum radius [rMin,1] (normalized frequencies) used for decorrelation analysis. Defaults to 1.
            n_r (int, optional): [10,100], Sampling of decorrelation curve. Defaults to 50.
            n_g (int, optional): [5,40], Number of high-pass image analyzed. Defaults to 10.
            pixel_size (int, optional): > 1, pixel size value in units. Defaults to 1.
            units (str, optional): string name of the units to use. Defaults to "pixel".
            roi (tuple, optional): Coordinates used to crop the image (x0, y0, x1, y1). Defaults to None.
            do_plot (bool, optional): Defaults to False.
        """
        self.img = None
        self.img_ref = None
        self.rmin, self.rmin2 = rmin, rmin
        self.rmax, self.rmax2 = rmax, rmax
        self.n_r = n_r
        self.n_g = n_g
        self.pixel_size = pixel_size
        self.units = units
        self.x0, self.x1, self.y0, self.y1 = roi
        self.do_plot = do_plot
        self.d0 = np.zeros((self.n_r), dtype=np.float32)
        self.d = np.zeros((self.n_r, 2*self.n_g), dtype=np.float32)
        self.kc = np.zeros((2*self.n_g), dtype=np.float32)
        self.a_g = np.zeros((2*self.n_g), dtype=np.float32)
        self.kc0 = 0
        self.a0 = 0
        self.kc_gm = 0
        self.agm = 0
        self.kc_max = 0
        self.a_max = 0
        self.resolution = 0
        self.units = units

    # @timeit2
    def run_analysis(self,  img: np.ndarray):
        """
        Method used to run the analysis. Starting parameters are defined on class instance initialization.
        Args:
            img (np.ndarray): image to analyze
        """
        self.img = img.astype(np.float32)
        return self._run_analysis()

    cdef float[:, :] _get_preprocessed_image(self, img):

        cdef int new_size, ox, oy, x_in, y_in, x_out, y_out
        cdef float mean_img
        cdef float[:] pixels_in, pixels_out
        cdef float[:, :] output

        new_size = max(img.shape[0], img.shape[1])
        new_size = <int>(pow(2, ceil(log(new_size)/log(2.0))))
        pixels_in = np.array(img).ravel()
        mean_img = pixels_in[0]
        pixels_out = np.full((new_size*new_size), mean_img, dtype=np.float32)

        ox = (new_size-img.shape[1])//2
        oy = (new_size-img.shape[0])//2

        x_in = 0
        x_out = ox
        y_in = 0
        y_out = oy
        cdef int _u
        cdef int _v

        for x_in in range(img.shape[1]):
            y_out = 0
            for y_in in range(img.shape[0]):
                _u = x_out + y_out*new_size
                _v = x_in + y_in*img.shape[1]
                pixels_out[_u] = pixels_in[_v]
                # pixels_out[x_out + y_out*new_size] = pixels_in[x_in + y_in*img.shape[1]] - how it was before
                y_out += 1
            x_out += 1

        output = np.reshape(pixels_out, (new_size, new_size))

        return output


    cdef float _linmap(self, float val, float valmin, float valmax, float mapmin, float mapmax) nogil:

        cdef float out = 0
        out = (val - valmin)/(valmax - valmin)
        out = out * (mapmax-mapmin) + mapmin
        return out

    cdef float _get_corr_coef_norm(self, float[:, :] fft_real, float[:, :] fft_imag, float[:, :] mask):

        cdef int y_i, x_i
        cdef float c = 0

        with nogil:
            for y_i in prange(fft_real.shape[0]):
                for x_i in range(fft_real.shape[1]):
                    if mask[y_i, x_i] == 1:
                        c += fft_real[y_i, x_i]**2 + fft_imag[y_i, x_i]**2

        return sqrt(c)

    cdef float[:] _get_best_score(self, float[:] kc, float[:] a):
        cdef int k
        cdef float gm_max
        cdef float[:] gm, out

        gm = np.empty(kc.shape[0], dtype=np.float32)
        out = np.empty((3), dtype=np.float32)

        gm_max = 0.0

        for k in range(kc.shape[0]):
            gm[k] = kc[k]*a[k]

            if gm[k] > gm_max:
                gm_max = gm[k]
                out[0] = kc[k]
                out[1] = a[k]
                out[2] = k

        return out

    cdef float[:] _get_max_score(self, float[:] kc, float[:] a):
        cdef int k
        cdef float kc_max
        cdef float[:] out

        out = np.empty((3), dtype=np.float32)
        kc_max = 0.0

        for k in range(kc.shape[0]):
            if kc[k] > kc_max:
                kc_max = kc[k]
                out[0] = kc[k]
                out[1] = a[k]
                out[2] = k

        return out

    cdef float [:] _get_corr_coef_ring(self, float[:, :] fft_real, float[:, :] fft_imag, float[:, :] normalized_fft_real, float[:, :] normalized_fft_imag, float crmin, float crmax):

        cdef float[:] out
        cdef int y_i, x_i, d, width, height, ox, oy, w, h
        cdef float dist, k
        width = self.img_ref.shape[1]
        height = self.img_ref.shape[0]
        out = np.zeros((2*int(self.n_r)), dtype=np.float32)

        d = 0
        dist = 0.0
        k = 0

        ox = <int>(width * (1-crmax)/2)
        oy = <int>(height * (1-crmax)/2)
        w = <int>(width * crmax)
        h = <int>(height * crmax)

        with nogil:
            for x_i in range(ox, ox+w):
                for y_i in range(oy, oy+h):
                    dist = (x_i-width/2)**2 + (y_i-height/2)**2
                    dist = sqrt(4*dist/(width**2))
                    k = x_i*self.img_ref.shape[0] + y_i
                    if k > width*height/2 + height/2:
                        return out
                    else:
                        if dist >= 0 and dist <= crmax:
                            dist = self._linmap(dist, crmin, crmax, 0, self.n_r-1)
                            if dist < 0:
                                dist = 0
                            d = <int>(round(dist))
                            if d+self.n_r < out.shape[0]:
                                out[d] += fft_real[y_i, x_i] * normalized_fft_real[y_i, x_i] + fft_imag[y_i, x_i] * normalized_fft_imag[y_i, x_i]
                                out[d+self.n_r] += normalized_fft_real[y_i, x_i]**2 + normalized_fft_imag[y_i, x_i]**2

        return out

    cdef float[:] _compute_d0(self, float[:, :] fft_real, float[:, :] fft_imag):

        cdef int k
        cdef float d, c
        cdef float[:] d0
        cdef float[:, :, :] normalized_fft = _normalizeFFT(fft_real, fft_imag)
        cdef float[:, :] mask = _get_circular_mask(fft_real.shape[1], 1)
        cdef float cr = self._get_corr_coef_norm(fft_real, fft_imag, mask)
        cdef float[:] coef = self._get_corr_coef_ring(fft_real, fft_imag, normalized_fft[0], normalized_fft[1], self.rmin, self.rmax)
        d0 = np.zeros((self.n_r), dtype=np.float32)
        for k in range(self.n_r):
            d = 0
            c = 0

            for n in range(k+1):
                d += coef[n]
                c += coef[n+self.n_r]

            if cr == 0 or c == 0:
                d0[k] = float("nan")
            else:
                d0[k] = sqrt(2)*d/(cr*sqrt(c))

        if isnan(d0[0]):
            d0[0] = 0

        return d0

    cdef float[:] _get_d_corr_max(self, float[:] d, float r1, float r2) nogil:
        cdef float[:] t, out, temp_min
        cdef int d_length
        cdef float dt = 0.001

        with gil:
            t = np.copy(d)

        out = _get_max(d, 0, self.n_r)
        temp_min = _get_min(d, 0, self.n_r)
        d_length = t.shape[0]

        while out[0] == d_length-1:
            t[d_length-1] = 0
            d_length -= 1
            if d_length == 0:
                out[0] = 0
                out[1] = 0
                break
            else:
                out = _get_max(t, 0, self.n_r)
                temp_min = _get_min(t, int(out[0]), d_length-1)

                if t[<int>(out[0])] - temp_min[1] > dt:
                    break
                else:
                    t[<int>(out[0])] = temp_min[1]
                    out[0] = d_length - 1

        out[0] = r1 + (r2-r1)*out[0]/(self.n_r-1)
        return out

    cdef float[:, :] _compute_d(self):

        cdef float[:] coef, kc, a, dg, result, results_gm, results_max
        cdef complex[:, :] fft
        cdef float[:, :] d_curve, img_ref, blurred, mask, fft_real, fft_imag
        cdef float[:, :, :] normalized_fft
        cdef int count = 0
        cdef float g_max, g_min, crmin, crmax, d, c, ind1, ind2, cr
        cdef double sig
        cdef int refine, k, j, h, i

        d_curve = np.zeros((self.n_r, 2*self.n_g), dtype=np.float32)

        if self.kc0 == 0:
            g_max = self.img_ref.shape[1] / 2
        else:
            g_max = 2 / self.kc0

        g_min = 0.14

        img_ref = np.copy(self.img_ref)
        img_ref = self._get_preprocessed_image(img_ref)
        blurred = np.copy(self.img_ref)

        crmin = self.rmin
        crmax = self.rmax

        mask = _get_circular_mask(img_ref.shape[1], 1)

        for refine in range(2):
            for k in range(self.n_g):
                sig = exp(log(g_min) + (log(g_max) - log(g_min))*(<float>(k)/(self.n_g-1)))
                blurred = np.copy(self.img_ref)
                blurred = gaussian_filter(blurred, sig)
                blurred = np.copy(self.img_ref) - blurred
                fft = np.fft.fftshift(np.fft.fft2(blurred))
                fft_real = np.real(fft).astype(np.float32)
                fft_imag = np.imag(fft).astype(np.float32)
                normalized_fft = _normalizeFFT(fft_real, fft_imag)
                cr = self._get_corr_coef_norm(fft_real, fft_imag, mask)
                coef = self._get_corr_coef_ring(fft_real, fft_imag, normalized_fft[0], normalized_fft[1], crmin, crmax)

                for i in range(self.n_r):
                    d = 0
                    c = 0

                    for n in range(i+1):
                        d += coef[n]
                        c += coef[n+self.n_r]
                    if cr == 0 or c == 0:
                        d_curve[i][count] = float("nan") # TODO: check this is ok, this is a workaround for differences in java and python
                    else:
                        d_curve[i][count] = sqrt(2)*d/(cr*sqrt(c))

                if isnan(d_curve[0][count]):
                    d_curve[0][count] = 0
                count += 1

            if refine == 0:
                kc = np.zeros((self.n_g+1), dtype=np.float32)
                a = np.zeros((self.n_g+1), dtype=np.float32)
                dg = np.zeros((self.n_r), dtype=np.float32)
                result = np.zeros((2), dtype=np.float32)

                with nogil:
                    for j in range(self.n_g):
                        for h in prange(self.n_r):
                            dg[h] = d_curve[h][j]
                        result = self._get_d_corr_max(dg, crmin, crmax)
                        kc[j] = result[0]
                        a[j] = result[1]
                        self.kc[j] = result[0]
                        self.a_g[j] = result[1]

                result = self._get_d_corr_max(self.d0, crmin, crmax)

                kc[self.n_g] = result[0]
                a[self.n_g] = result[1]
                self.kc[self.n_g] = result[0]
                self.a_g[self.n_g] = result[1]

                results_gm = self._get_best_score(kc, a)
                results_max = self._get_max_score(kc, a)

                crmin = fmin(results_gm[0], results_max[0]) - 0.05
                if crmin < self.rmin:
                    crmin = self.rmin

                crmax = fmax(results_gm[0], results_max[0]) + 0.3
                if crmax > self.rmax:
                    crmax = self.rmax

                crmax = 0.5
                self.rmin2 = crmin
                self.rmax2 = crmax

                ind1 = fmin(results_gm[2], results_max[2]) - 1
                ind2 = fmax(results_gm[2], results_max[2])

                if ind2 < self.n_g:
                    g_temp = exp(log(g_min) + (log(g_max)-log(g_min))*(ind1/(self.n_g-1)))
                    g_max = exp(log(g_min) + (log(g_max)-log(g_min))*(ind2/(self.n_g-1)))
                    g_min = g_temp
                else:
                    g_max = g_min
                    g_min = 2 / self.img_ref.shape[1]

            else:
                kc = np.zeros((self.n_g), dtype=np.float32)
                a = np.zeros((self.n_g), dtype=np.float32)
                dg = np.zeros((self.n_r), dtype=np.float32)
                result = np.zeros((2), dtype=np.float32)

                with nogil:
                    for j in range(self.n_g):
                        for h in prange(self.n_r):
                            dg[h] = d_curve[h][j+self.n_g]
                        result = self._get_d_corr_max(dg, crmin, crmax)
                        kc[j] = result[0]
                        a[j] = result[1]
                        self.kc[j] = result[0]
                        self.a_g[j] = result[1]
                results_gm = self._get_best_score(kc, a)
                results_max = self._get_max_score(kc, a)

                self.kc_gm = results_gm[0]
                self.agm = results_gm[1]
                self.kc_max = results_max[0]
                self.a_max = results_max[1]

        return d_curve

    cdef _run_analysis(self):

        cdef float[:] out
        cdef complex[:, :] img_fft
        cdef float[:, :] img_ref, img_f, temp, fft_real, fft_imag
        cdef float resolution
        img_ref = np.copy(self.img)
        img_f = np.copy(self.img)

        if self.x0 == self.x1 or self.y0 == self.y1:
            self.img_ref = img_ref
        else:
            self.img_ref = img_ref[self.y0:self.y1, self.x0:self.x1]

        img_f = _apodize_edges(img_f)
        temp = self._get_preprocessed_image(img_f)
        self.img_ref = np.copy(temp)
        img_fft = np.fft.fftshift(np.fft.fft2(temp))
        fft_real = np.real(img_fft).astype(np.float32)
        fft_imag = np.imag(img_fft).astype(np.float32)
        fft_real[fft_real.shape[0]//2, fft_real.shape[1]//2] = 0
        fft_imag[fft_imag.shape[0]//2, fft_imag.shape[1]//2] = 0
        self.d0 = self._compute_d0(fft_real, fft_imag)
        out = self._get_d_corr_max(self.d0, 0, 1)
        self.kc0 = out[0]
        self.a0 = out[1]
        self.d = self._compute_d()

        self.resolution = 2 * self.pixel_size / self.kc_max

        if self.do_plot:
            self.plot_results()

    def plot_results(self):
        """
        Returns the plot of the results of the analysis as a numpy array
        """
        fig, ax = plt.subplots(dpi=150)
        x = np.linspace(0.0, 1.0, self.n_r)
        for k in range(self.d0.shape[0]):
            x[k] = self.rmin + (self.rmax-self.rmin)*k/(self.n_r-1)

        ax.plot(x, np.array(self.d0), c="k")

        dg = np.zeros((self.n_r))
        for k in range(self.n_g):
            for j in range(self.n_r):
                dg[j] = np.array(self.d[j][k])
            plt.plot(x, dg, c="g")
        for k in range(self.d0.shape[0]):
            x[k] = self.rmin2 + (self.rmax2-self.rmin2)*k/(self.n_r-1)
        for k in range(self.n_g, 2*self.n_g):
            for j in range(self.n_r):
                dg[j] = np.array(self.d[j][k])
            plt.plot(x, dg, c="b")
        kc = np.array(self.kc)
        a_g = np.array(self.a_g)
        ax.axvline(x=self.kc_max, color="b", linestyle="-", label="Cut-off frequency")
        ax.set_xlabel(f"Normalized frequency")
        ax.set_ylabel("Cross-correlation coefficients")
        ax.set_title(f"Decorrelation analysis resolution: {np.round(self.resolution, 4)} {self.units}")
        with io.BytesIO() as buf:
            fig.savefig(buf, format="raw", dpi=150)
            buf.seek(0)
            data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))
        plt.close(fig)

        return im