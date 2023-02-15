# https://github.com/Ades91/ImDecorr/blob/master/ijplugin/src/ImageDecorrelationAnalysis.java
import math
import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd

from .decorr_utils import *
from ..analysis.ccm.helper_functions import make_even_square
from ..utils.time.timeit import timeit2

class DecorrAnalysis(object):
    
    def __init__(self, img:np.ndarray, rmin:float = 0, rmax:float = 1, n_r:int = 50, n_g:int =10, pixel_size: int = 1, units: str = "pixel", roi: tuple = None, do_plot: bool = False, save_path : str = None):
        """_summary_

        Args:
            img (np.ndarray): image, numpy array with shape (t, c, z, y, x)
            rmin (float, optional): Minimum radius [0,rMax] (normalized frequencies) used for decorrelation analysis. Defaults to 0.
            rmax (float, optional): Maximum radius [rMin,1] (normalized frequencies) used for decorrelation analysis. Defaults to 1.
            n_r (int, optional): [10,100], Sampling of decorrelation curve. Defaults to 50.
            n_g (int, optional): [5,40], Number of high-pass image analyzed. Defaults to 10.
            pixel_size (int, optional): > 1, pixel size value in units. Defaults to 1.
            units (str, optional): string name of the units to use. Defaults to "pixel".
            roi (tuple, optional): Coordinates used to crop the image (x0, y0, x1, y1). Defaults to None.
            do_plot (bool, optional): Defaults to False.
            save_path (str, optional): Defaults to None.
        """
        self.img = img
        self.img_ref = None
        self.rmin, self.rmin2 = rmin, rmin
        self.rmax, self.rmax2 = rmax, rmax
        self.n_r = n_r
        self.n_g = n_g
        self.pixel_size = pixel_size
        self.units = units
        self.roi = roi
        self.do_plot = do_plot
        self.d0 = np.empty((self.n_r), dtype=np.float32)
        self.d = np.empty((self.n_r, 2*self.n_g), dtype=np.float32)
        self.kc = np.empty((2*self.n_g), dtype=np.float32)
        self.a_g = np.empty((2*self.n_g), dtype=np.float32)
        self.kc0 = 0
        self.a0 = 0
        self.kc_gm = 0
        self.agm = 0
        self.kc_max = 0
        self.a_max = 0
        self.s = 1
        self.f = 1
        self.c = 1
        self.save_path = save_path
        self.results_table = pd.DataFrame(columns=["Frame", "Channel", "Z", "Resolution", "Units", "A0", "Kc", "Kc GM", "rMin", "rMax", "Nr", "Ng"])
    
    def normalizeFFT(self, fft_real, fft_imag):
        return normalizeFFT(fft_real, fft_imag)
    
    def apodize_edges(self, img):
        return apodize_edges(img)
    
    def get_mask(self, w, r2):
        
        return get_mask(w, r2)
    
    def get_corr_coef_norm(self, fft_real, fft_imag, mask):
        
        return get_corr_coef_norm(fft_real, fft_imag, mask)
    
    def get_corr_coef_ring(self, fft_real, fft_imag, normalized_fft_real, normalized_fft_imag):
        out = np.empty((2*int(self.n_r)), dtype=np.float32)
        d = 0
        dist = 0
        width = self.img_ref.shape[1]
        height = self.img_ref.shape[0]
        k = 0
        
        ox = int(width * (1-self.rmax)/2)
        oy = int(height * (1-self.rmax)/2)
        w = int(width * self.rmax)
        h = int(height * self.rmax)
        for x_i in range(ox, ox+w):
            for y_i in range(oy, oy+h):
                dist = (x_i-width/2)**2 + (y_i-height/2)**2
                dist = math.sqrt(4*dist/(width**2))
                
                if k > width+height/2 + height/2:
                    break
                else:
                    if dist >= 0 and dist <= self.rmax:
                        dist = linmap(dist, self.rmin, self.rmax, 0, self.n_r-1)
                        if dist < 0:
                            dist = 0
                        d = math.floor(dist)
                        out[d] += fft_real[y_i, x_i] * normalized_fft_real[y_i, x_i] + fft_imag[y_i, x_i] * normalized_fft_imag[y_i, x_i]
                        out[d+self.n_r] += normalized_fft_real[y_i, x_i]**2 + normalized_fft_imag[y_i, x_i]**2
                        
        return out
    
    def compute_d0(self, fft_real, fft_imag):
        
        normalized_fft = self.normalizeFFT(fft_real, fft_imag)
        normalized_fft_real = normalized_fft[0]
        normalized_fft_imag = normalized_fft[1]

        mask = self.get_mask(fft_real.shape[1], 1)
        
        cr = self.get_corr_coef_norm(fft_real, fft_imag, mask)
        
        coef = self.get_corr_coef_ring(fft_real, fft_imag, normalized_fft_real, normalized_fft_imag)
        
        for k in range(self.n_r):
            d = 0
            c = 0
            
            for n in range(k+1):
                d += coef[n]
                c += coef[n+self.n_r]
                
            self.d0[k] = math.sqrt(2)*d/(cr*math.sqrt(c))
            
        if math.isnan(self.d0[0]):
            self.d0[0] = 0
    
    def get_max(self, arr, x1, x2):
        return get_max(arr, x1, x2)
    
    def get_min(self, arr, x1, x2):
        return get_min(arr, x1, x2)
    
    def get_d_corr_max(self, d, r1, r2):
        
        t = d.copy()
        out = self.get_max(d, 0, self.n_r)
        temp_min = self.get_min(d, 0, self.n_r)
        d_length = t.shape[0]
        dt = 0.001
        
        while out[0] == d_length-1:
            t[d_length-1] = 0
            d_length -= 1
            if d_length == 0:
                out[0] = 0
                out[1] = 0
                break
            else:
                out = self.get_max(t, 0, self.n_r)
                temp_min = self.get_min(t, int(out[0]), d_length-1)
                
                if t[int(out[0])] - temp_min[1] > dt:
                    break
                else:
                    t[int(out[0])] = temp_min[1]
                    out[0] = d_length - 1
                    
        out[0] = r1 + (r2-r1)*out[0]/(self.n_r -1)
        return out
    
    def get_best_score(self, kc, a):
        return get_best_score(kc, a)
    
    def get_max_score(self, kc, a):
        return get_max_score(kc, a)
    
    def compute_d(self):
        
        count = 0
        
        if self.kc0 == 0:
            g_max = self.img_ref.shape[1]
        else:
            g_max = 2 / self.kc0
            
        g_min = 0.14
        
        ref = np.array([self.img_ref.copy()], dtype=np.float32)
        
        ref = np.array(make_even_square(ref)[0], dtype=np.float32)
        
        crmin = self.rmin
        crmax = self.rmax
        
        mask = self.get_mask(ref.shape[1], 1)
        
        for refine in range(2):
            for k in range(self.n_g):                
                sig = math.exp(math.log(g_min) + (math.log(g_max) - math.log(g_min))*((k/(self.n_g - 1))))
                
                blurred = gaussian_filter(ref, sig)
                blurred = ref - blurred
                
                fft = np.fft.fft2(blurred)
                fft_real = fft.real.astype(np.float32)
                fft_imag = fft.imag.astype(np.float32)
                normalized_fft = self.normalizeFFT(fft_real, fft_imag)
                normalized_fft_real = normalized_fft[0]
                normalized_fft_imag = normalized_fft[1]
                cr = self.get_corr_coef_norm(fft_real, fft_imag, mask)
        
                coef = self.get_corr_coef_ring(fft_real, fft_imag, normalized_fft_real, normalized_fft_imag)
                
                for i in range(self.n_r):
                    d = 0
                    c = 0
                    for n in range(i + 1):
                        d += coef[n]
                        c += coef[n+self.n_r]
                    if cr == 0 or c == 0:
                        self.d[i][count] = math.nan # TODO: check this is ok, this is a workaround for differences in java and python
                    else:
                        self.d[i][count] = math.sqrt(2)*d/(cr*math.sqrt(c))
                    
                if math.isnan(self.d[0][count]):
                    self.d[0][count] = 0
                count += 1
                
            if refine == 0:
                kc = np.empty((self.n_g+1), dtype=np.float32)
                a = np.empty((self.n_g+1), dtype=np.float32)
                dg = np.empty((self.n_r), dtype=np.float32)
                result = np.empty((2), dtype=np.float32)
                
                for j in range(self.n_g):
                    for h in range(self.n_r):
                        dg[h] = self.d[h][j]
                    result = self.get_d_corr_max(dg, crmin, crmax)
                    kc[self.n_g] = result[0]
                    a[self.n_g] = result[1]
                    self.kc[self.n_g] = result[0]
                    self.a_g[self.n_g] = result[1]
                result = self.get_d_corr_max(self.d0, crmin, crmax)
                kc[self.n_g] = result[0]
                a[self.n_g] = result[1]
                self.kc[self.n_g] = result[0]
                self.a_g[self.n_g] = result[1]
                
                results_gm = self.get_best_score(kc, a)
                results_max = self.get_max_score(kc, a)
                
                crmin = min(results_gm[0], results_max[0]) - 0.05
                if crmin < self.rmin:
                    crmin = self.rmin
                    
                crmax = max(results_gm[0], results_max[0]) + 0.3
                if crmax > self.rmax:
                    crmax = self.rmax
                    
                crmax = 0.5 # TODO: double check this, as this is the original implementation but makes no sense considering the previous calculations
                self.rmin2 = crmin
                self.rmax2 = crmax
                
                ind1 = min(results_gm[2], results_max[2])-1
                ind2 = max(results_gm[2], results_max[2])
                
                if ind2 < self.n_g:
                    g_temp = math.exp(math.log(g_min) + (math.log(g_max)-math.log(g_min))*(ind1/(self.n_g-1)))
                    g_max = math.exp(math.log(g_min) + (math.log(g_max)-math.log(g_min))*(ind2/(self.n_g-1)))
                    g_min = g_temp
                else:
                    g_max = g_min
                    g_min = 2 / self.img_ref.shape[1]
                    
            else:
                kc = np.empty((self.n_g), dtype=np.float32)
                a = np.empty((self.n_g), dtype=np.float32)
                dg = np.empty((self.n_r), dtype=np.float32)
                result = np.empty((2), dtype=np.float32)
                for j in range(self.n_g):
                    for h in range(self.n_r):
                        dg[h] = self.d[h][j+self.n_g]
                        
                    result = self.get_d_corr_max(dg, crmin, crmax)
                    kc[j] = result[0]
                    a[j] = result[1]
                    self.kc[j+self.n_g] = result[0]
                    self.a_g[j+self.n_g] = result[1]
                results_gm = self.get_best_score(self.kc, self.a_g)
                self.kc_gm = results_gm[0]
                self.agm = results_gm[1]
                results_max = self.get_max_score(self.kc, self.a_g)
                self.kc_max = results_max[0]
                self.a_max = results_max[1]

    def check_even_square(self, image_arr):
        w = image_arr.shape[1]
        h = image_arr.shape[0]

        if w != h:
            return False
        if w % 2 != 0:
            return False

        return True
    
    def get_closest_even_square(self, image_arr):
        w = image_arr.shape[1]
        h = image_arr.shape[0]
        min_size = min(w, h)

        if min_size % 2 != 0:
            min_size -= 1

        return min_size

    def make_even_square(self, image_arr):
        if self.check_even_square(image_arr):
            return image_arr

        w = image_arr.shape[2]
        h = image_arr.shape[1]
        min_size = self.get_closest_even_square_size(image_arr)

        h_start = (h-min_size)//2
        if (h-min_size) % 2 != 0:
            h_finish = h - (h-min_size) // 2 - 1
        else:
            h_finish = h - (h-min_size) // 2

        w_start = int((w-min_size)/2)
        if (w - min_size) % 2 != 0:
            w_finish = w - (w-min_size) // 2 - 1
        else:
            w_finish = w - (w-min_size) // 2 

        return image_arr[:, h_start:h_finish, w_start:w_finish]
    
    @timeit2
    def run_analysis(self):
        
        for f_i  in range(self.img.shape[0]):
            for c_i in range(self.img.shape[1]):
                for s_i in range(self.img.shape[2]):
                    self.f = f_i
                    self.c = c_i
                    self.s = s_i
                    self.d0 = np.empty((self.n_r), dtype=np.float32)
                    self.d = np.empty((self.n_r, 2*self.n_g), dtype=np.float32)
                    self.kc = np.empty((2*self.n_g), dtype=np.float32)
                    self.a_g = np.empty((2*self.n_g), dtype=np.float32)
                    self.kc0 = 0
                    self.a0 = 0
                    self.kc_gm = 0
                    self.agm = 0
                    self.kc_max = 0
                    self.a_max = 0
                    self.img_ref = None
                    
                    img_ref = self.img[f_i, c_i, s_i].astype(np.float32)
                    
                    if self.roi is not None:
                        img_ref = img_slice[y0:y1, x0:x1]
                        
                    img_f = img_ref.copy()
                    img_f = np.array([self.apodize_edges(img_f)])
                    #img_f = img_f.reshape((1, img_f.shape[0], img_f.shape[1]))
                    
                    temp = make_even_square(img_f)[0]
                    self.img_ref = temp.copy()
                    
                    img_fft = np.fft.fftshift(np.fft.fft2(temp))
                    fft_real = img_fft.real.astype(np.float32)
                    fft_imag = img_fft.imag.astype(np.float32)
                    
                    fft_real[fft_real.shape[0]//2, fft_real.shape[1]//2] = 0
                    fft_imag[fft_imag.shape[0]//2, fft_imag.shape[1]//2] = 0
                    
                    self.compute_d0(fft_real, fft_imag)
                    out = self.get_d_corr_max(self.d0, 0, 1)
                    self.kc0 = out[0]
                    self.a0 = out[1]
                    
                    self.compute_d()
                    
                    self.resolution = 2*self.pixel_size/self.kc_max
                    print(f"Resolution: {self.resolution}")
                    
                    self.results_table.loc[len(self.results_table)] = [f_i, c_i, s_i, self.resolution, self.units, self.a0, self.kc_max, self.kc_gm, self.rmin, self.rmax, self.n_r, self.n_g]
                    
                    if self.do_plot:
                        self.do_plot()
                    
    def plot_results(self):
        pass