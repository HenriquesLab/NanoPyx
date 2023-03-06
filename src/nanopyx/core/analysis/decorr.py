# https://github.com/Ades91/ImDecorr/blob/master/ijplugin/src/ImageDecorrelationAnalysis.java
import math
import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd
from math import sqrt, fabs, cos
from matplotlib import pyplot as plt

from .decorr_utils import *
from ..utils.timeit import timeit2

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
		self.f = 1
		self.save_path = save_path
		self.results_table = pd.DataFrame(columns=["Frame", "Resolution", "Units", "A0", "Kc", "Kc GM", "rMin", "rMax", "Nr", "Ng"])
	
	def normalizeFFT(self, fft_real, fft_imag):
		return normalizeFFT(fft_real, fft_imag)
	
	def apodize_edges(self, img):
		pin = img.ravel()
		
		dist = 0
		off = 20
		r = img.shape[1]/2 - off
		
		edge_mean = 0
		count = 0
		for x in range(img.shape[1]):
			for y in range(img.shape[0]):
				dist = (x-img.shape[1]/2)*(x-img.shape[1]/2) + (y-img.shape[0]/2)*(y-img.shape[0]/2)
				if dist > r * r:
					edge_mean += pin[y*img.shape[1] + x]
					count += 1
					
		edge_mean = edge_mean / count
		x0 = 0
		y0 = 0
		
		for x in range(img.shape[1]):
			for y in range(img.shape[0]):
				x0 = abs(x-img.shape[1]/2)
				y0 = abs(y-img.shape[0]/2)
				if abs(x0-img.shape[1]/2) <= off or abs(y0-img.shape[0]/2) <= off:
					d = min(abs(x0-img.shape[1]/2), abs(y0-img.shape[0]/2))
					c = (math.cos(d*math.pi/off-math.pi)+1)/2
					pin[y*img.shape[1] + x] = (c*(pin[y*img.shape[1] + x]-edge_mean))+edge_mean
				elif (abs(x-img.shape[0]/2) > img.shape[0]/2 and abs(y-img.shape[1]/2) > img.shape[1]/2):
					pin[y*img.shape[1] + x] = edge_mean

		return pin.reshape(img.shape[0], img.shape[1])

	def get_preprocessed_image(self, img):
		newSize = max(img.shape[1], img.shape[0])

		newSize = int(math.pow(2, math.ceil(math.log(newSize)/math.log(2.0))))
		
		pixelsIn = img.ravel()
		
		meanIm = pixelsIn[0]; 
		
		pixelsOut = np.zeros((newSize*newSize), dtype=np.float32);
		for k in range(newSize*newSize):
			pixelsOut[k] = meanIm
		
		ox = (newSize-img.shape[1])//2
		oy = (newSize-img.shape[0])//2

		xIn = 0
		xOut = ox
		yIn = 0
		yOut = oy
		for xIn in range(img.shape[1]):
			yOut = 0
			for yIn in range(img.shape[0]):
				pixelsOut[yOut*newSize +xOut] = pixelsIn[yIn*img.shape[1] + xIn]
				yOut += 1
			xOut += 1
		return pixelsOut.reshape((newSize, newSize))
	
	def get_mask(self, w, r2):
		
		return get_mask(w, r2)
	
	def get_corr_coef_norm(self, fft_real, fft_imag, mask):
		return get_corr_coef_norm(fft_real, fft_imag, mask)

	def linmap(self, val, valmin, valmax, mapmin, mapmax):
		return linmap(val, valmin, valmax, mapmin, mapmax)
	
	def get_corr_coef_ring(self, fft_real, fft_imag, normalized_fft_real, normalized_fft_imag, crmin, crmax):
		out = np.zeros((2*int(self.n_r)), dtype=np.float32)
		d = 0
		dist = 0
		width = self.img_ref.shape[1]
		height = self.img_ref.shape[0]
		k = 0
		
		ox = int(width * (1-crmax)/2)
		oy = int(height * (1-crmax)/2)
		w = int(width * crmax)
		h = int(height * crmax)
		for x_i in range(ox, ox+w):
			for y_i in range(oy, oy+h):
				dist = (x_i-width/2)**2 + (y_i-height/2)**2
				dist = math.sqrt(4*dist/(width**2))
				k = x_i*self.img_ref.shape[0] + y_i
				if k > width*height/2 + height/2:
					return out
				else:
					if dist >= 0 and dist <= crmax:
						dist = self.linmap(dist, crmin, crmax, 0, self.n_r-1)
						if dist < 0:
							dist = 0
						d = round(dist)
						if d+self.n_r < out.shape[0]:
							out[d] += fft_real[y_i, x_i] * normalized_fft_real[y_i, x_i] + fft_imag[y_i, x_i] * normalized_fft_imag[y_i, x_i]
							out[d+self.n_r] += normalized_fft_real[y_i, x_i]**2 + normalized_fft_imag[y_i, x_i]**2

		return out
	
	def compute_d0(self, fft_real, fft_imag):

		d0 = np.zeros((self.n_r), dtype=np.float32)
		
		normalized_fft = self.normalizeFFT(fft_real, fft_imag)

		mask = self.get_mask(fft_real.shape[1], 1)
		
		cr = self.get_corr_coef_norm(fft_real, fft_imag, mask)

		coef = self.get_corr_coef_ring(fft_real, fft_imag, normalized_fft[0], normalized_fft[1], self.rmin, self.rmax)

		for k in range(self.n_r):
			d = 0
			c = 0
			
			for n in range(k+1):
				d += coef[n]
				c += coef[n+self.n_r]
			
			if cr == 0 or c == 0:
				d0[k] = float("nan")
			else:
				d0[k] = math.sqrt(2)*d/(cr*math.sqrt(c))
			
		if math.isnan(d0[0]):
			d0[0] = 0

		return d0
	
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
				return out
			else:
				out = self.get_max(t, 0, self.n_r)
				temp_min = self.get_min(t, int(out[0]), d_length-1)
				
				if t[int(out[0])] - temp_min[1] > dt:
					return out
				else:
					t[int(out[0])] = temp_min[1]
					out[0] = d_length - 1
					
		out[0] = r1 + (r2-r1)*out[0]/(self.n_r-1)
		return out
	
	def get_best_score(self, kc, a):
		return get_best_score(kc, a)
	
	def get_max_score(self, kc, a):
		return get_max_score(kc, a)
	
	def compute_d(self):

		d_curve = np.zeros((self.n_r, 2*self.n_g), dtype=np.float32)
		
		count = 0
		
		if self.kc0 == 0:
			g_max = self.img_ref.shape[1] / 2
		else:
			g_max = 2 / self.kc0
			
		g_min = 0.14
		
		img_ref = self.img_ref.copy()
		img_ref = self.get_preprocessed_image(img_ref)
		blurred = self.img_ref.copy()
		
		crmin = 0
		crmax = 0
		crmin += self.rmin
		crmax += self.rmax
		
		mask = self.get_mask(img_ref.shape[1], 1)
		
		for refine in range(2):
			for k in range(self.n_g):                
				sig = math.exp(math.log(g_min) + (math.log(g_max)-math.log(g_min))*(k/(self.n_g-1)))
				blurred = self.img_ref.copy()
				blurred = gaussian_filter(blurred, sig)
				blurred = self.img_ref.copy() - blurred
								
				fft = np.fft.fftshift(np.fft.fft2(blurred))
				fft_real = fft.real.astype(np.float64)
				fft_imag = fft.imag.astype(np.float64)
				normalized_fft = self.normalizeFFT(fft_real, fft_imag)
				cr = self.get_corr_coef_norm(fft_real, fft_imag, mask)
				coef = self.get_corr_coef_ring(fft_real, fft_imag, normalized_fft[0], normalized_fft[1], crmin, crmax)
				
				for i in range(self.n_r):
					d = 0
					c = 0
					for n in range(i + 1):
						d += coef[n]
						c += coef[n+self.n_r]
					if cr == 0 or c == 0:
						d_curve[i][count] = float("nan") # TODO: check this is ok, this is a workaround for differences in java and python
					else:
						d_curve[i][count] = math.sqrt(2)*d/(cr*math.sqrt(c))
					
				if math.isnan(d_curve[0][count]):
					d_curve[0][count] = 0
				count += 1
				
			if refine == 0:
				kc = np.zeros((self.n_g+1), dtype=np.float32)
				a = np.zeros((self.n_g+1), dtype=np.float32)
				dg = np.zeros((self.n_r), dtype=np.float32)
				result = np.zeros((2), dtype=np.float32)
				
				for j in range(self.n_g):
					for h in range(self.n_r):
						dg[h] = d_curve[h][j]
					result = self.get_d_corr_max(dg, crmin, crmax)
					kc[j] = result[0]
					a[j] = result[1]
					self.kc[j] = result[0]
					self.a_g[j] = result[1]

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
				kc = np.zeros((self.n_g), dtype=np.float32)
				a = np.zeros((self.n_g), dtype=np.float32)
				dg = np.zeros((self.n_r), dtype=np.float32)
				result = np.zeros((2), dtype=np.float32)
				for j in range(self.n_g):
					for h in range(self.n_r):
						dg[h] = d_curve[h][j+self.n_g]
						
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

		return d_curve
	
	@timeit2
	def run_analysis(self):

		for f_i  in range(self.img.shape[0]):
			self.f = f_i
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
			self.img_ref = None
			
			img_ref = np.zeros((self.img.shape[-2], self.img.shape[-1]), dtype=np.float32)
			img_ref += self.img[f_i]
			
			if self.roi is not None:
				self.img_ref = img_ref[y0:y1, x0:x1].copy()
			else:
				self.img_ref = img_ref.copy()
				
			img_f = img_ref.copy()
			img_f = self.apodize_edges(img_f)
			
			temp = self.get_preprocessed_image(img_f)
			self.img_ref = temp.copy()
			img_fft = np.fft.fftshift(np.fft.fft2(temp))
			fft_real = img_fft.real.astype(np.float64)
			fft_imag = img_fft.imag.astype(np.float64)
			fft_real[fft_real.shape[0]//2, fft_real.shape[1]//2] = 0
			fft_imag[fft_imag.shape[0]//2, fft_imag.shape[1]//2] = 0
			
			self.d0 = self.compute_d0(fft_real, fft_imag)
			out = self.get_d_corr_max(self.d0, 0, 1)
			self.kc0 = out[0]
			self.a0 = out[1]
			self.d = self.compute_d()
			
			self.resolution = 2*self.pixel_size/self.kc_max
			print(f"Resolution: {self.resolution}")
			
			self.results_table.loc[len(self.results_table)] = [f_i, self.resolution, self.units, self.a0, self.kc_max, self.kc_gm, self.rmin, self.rmax, self.n_r, self.n_g]
			
			if self.do_plot:
				self.plot_results()
					
	def plot_results(self):
		x = np.zeros((self.n_r))
		for k in range(self.d0.shape[0]):
			x[k] = 0 + (1-0)*k/(self.n_r-1)

		plt.plot(x, self.d0, c="r")
		
		for k in range(self.d0.shape[0]):
			x[k] = self.rmin + (self.rmax-self.rmin)*k/(self.n_r-1)

		dg = np.zeros((self.n_r))
		for k in range(self.n_g):
			for j in range(self.n_r):
				dg[j] = self.d[j][k]
			plt.plot(x, dg, c="g")
		for k in range(self.d0.shape[0]):
			x[k] = self.rmin2 + (self.rmax2-self.rmin2)*k/(self.n_r-1)
		for k in range(self.n_g, 2*self.n_g):
			for j in range(self.n_r):
				dg[j] = self.d[j][k]
			plt.plot(x, dg, c="b")
		plt.plot(self.kc, self.a_g, c="g", marker="*")
		plt.plot(self.kc0, self.a0, c="r", marker="x")
		plt.axvline(x=self.kc_max, color='b', linestyle='-', label="Cut-off frequency")
		plt.xlabel(f'Spatial frequency [1/{self.units}]')
		plt.ylabel('Cross-correlation coefficients')
		plt.title(f"Decorrelation analysis resolution: {np.round(self.resolution, 4)} {self.units}")
		plt.show()

