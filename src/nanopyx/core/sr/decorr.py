# https://github.com/Ades91/ImDecorr/blob/master/ijplugin/src/ImageDecorrelationAnalysis.java
import numpy as np

from .utils import apodize_edges

class DecorrAnalysis(object):
    
    def __init__(self, img:np.ndarray, rmin:float, rmax:float, n_r:int, n_g:int, roi: tuple = None, do_plot: bool = False, save_path : str = None):
        """_summary_

        Args:
            img (np.ndarray): image array with shape (t, c, z, y, x)
            roi (tuple): (x0, y0, x1, y1)
            rmin (float): _description_
            rmax (float): _description_
            n_r (int): _description_
            n_g (int): _description_
            do_plot (bool, optional): _description_. Defaults to False.
            save_path (str, optional): _description_. Defaults to None.
        """
        self.img = img
        self.rmin, self.rmin2 = rmin, rmin
        self.rmax, self.rmax2 = rmax, rmax
        self.n_r = n_r
        self.n_g = n_g
        self.roi = roi
        self.do_plot = do_plot
        self.d0 = np.empty((self.n_r), dtype=np.float32)
        self.d = np.empty((self.n_r, 2*self.n_g), dtype=np.float32)
        self.kc = np.empty((2*self.n_g), dtype=np.float32)
        self.a_g = np.empty((2*self.n_g), dtype=np.float32)
        self.kc_gm = 0
        self.agm = 0
        self.kc_max = 0
        self.a_mac = 0
        self.s = 1
        self.f = 1
        self.c = 1
        self.save_path = save_path
    
    def run_analysis(self):
        
        for f_i  in self.image.shape[0]:
            for c_i in self.image.shape[1]:
                for s_i in self.image.shape[2]:
                    self.f = f_i
                    self.c = c_i
                    self.s = s_i
                    
                    img_ref = self.img[f_i, c_i, s_i].astype(np.float32)
                    
                    if self.roi is not None:
                        img_ref = img_slice[y0:y1, x0:x1]
                        
                    img_slice = img_ref.copy()
                    img_slice = apodize_edges(img_slice)