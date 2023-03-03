# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=False

from libc.math cimport sqrt, fabs, exp, isnan, floor

from ..transform.interpolation_catmull_rom cimport _interpolate, Interpolator
from ..transform.image_magnify import cv2_zoom as zoom
# from ..transform.image_magnify import fourier_zoom as zoom
from ..utils.timeit import timeit2


import numpy as np
cimport numpy as np

from cython.parallel import prange

cdef float Gx_Gy_MAGNIFICATION = 2.0

cdef class RadialGradientConvergence:

    cdef int magnification
    cdef float fwhm
    cdef float sensitivity
    cdef float tSS # two sigma squared
    cdef float tSO # two sigma plus one
    cdef bint doIntensityWeighting


    def __init__(self, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1 , doIntensityWeighting: bool = True):
        """
        Calculate the Radial Gradient Convergence (RGC) of an image.
        :param magnification: magnification of the image
        :param radius: radius of the RGC (the PSF Full-Width-Half-Maximum)
        :param sensitivity: sensitivity of the RGC (sharpening factor)
        :param doIntensityWeighting: whether to do intensity weighting
        """
        self.magnification = magnification
        self.fwhm = radius
        self.sensitivity = sensitivity
        self.doIntensityWeighting = doIntensityWeighting
        
        cdef float sigma = radius / 2.355
        self.tSS = 2 * sigma * sigma
        self.tSO = 2 * sigma + 1

    def calculate(self, im: np.ndarray):
        """
        Calculate the RGC of an image-stack.
        :param im: the image to calculate the RGC of
        :return: the RGC of the image
        """
        assert im.ndim == 3

        nFrames = im.shape[0]

        cdef float [:,:,:] imRaw = im.astype(np.float32)
        cdef float [:,:,:] imRad = np.zeros((im.shape[0], im.shape[1]*self.magnification, im.shape[2]*self.magnification), dtype=np.float32)
        cdef float [:,:,:] imInt = np.zeros((im.shape[0], im.shape[1]*self.magnification, im.shape[2]*self.magnification), dtype=np.float32) # interpolated image
        cdef float [:,:,:] imGx = np.zeros_like(imRaw) # Gradient of the original image
        cdef float [:,:,:] imGy = np.zeros_like(imRaw)
        cdef float [:,:,:] imIntGx = np.zeros((im.shape[0], im.shape[1]*self.magnification*2, im.shape[2]*self.magnification*2), dtype=np.float32)
        cdef float [:,:,:] imIntGy = np.zeros((im.shape[0], im.shape[1]*self.magnification*2, im.shape[2]*self.magnification*2), dtype=np.float32)

        cdef int n
        for n in range(nFrames):
            self._single_frame_RGC_map(imRaw[n,:,:], imRad[n,:,:], imInt[n,:,:], imGx[n,:,:], imGy[n,:,:], imIntGx[n,:,:], imIntGy[n,:,:])

        return imRad, imInt, imIntGx, imIntGy


    cdef void _single_frame_RGC_map(self, float[:,:] imRaw, float[:,:] imRad, float[:,:] imInt, float[:,:] imGx, float[:,:] imGy, float[:,:] imIntGx, float[:,:] imIntGy):
        cdef int w = imRaw.shape[1]
        cdef int h = imRaw.shape[0]
        cdef int yM, xM

        #TODO: change interpolation techniques

        self._calculate_gradient(imRaw, imGx, imGy) # calculate gradients of the raw image

        cdef Interpolator interpolator = Interpolator(imRaw)
        imInt[:,:] = interpolator._magnify(self.magnification) #for Intensity Weighting

        cdef Interpolator interpolator_gx = Interpolator(imGx) #interpolate gradients for RGC calculation
        imIntGx[:,:] = interpolator_gx._magnify(2*self.magnification) 

        cdef Interpolator interpolator_gy = Interpolator(imGy)
        imIntGy[:,:] = interpolator_gy._magnify(2*self.magnification) #fix this

        with nogil:
            for yM in prange(self.magnification, h * self.magnification): 
                for xM in range(self.magnification, w * self.magnification):
                    if self.doIntensityWeighting:
                        imRad[yM, xM] = self._calculateRGC(xM, yM, imIntGx, imIntGy) * imInt[yM, xM]
                    else:
                        imRad[yM, xM] = self._calculateRGC(xM, yM, imIntGx, imIntGy) 


    cdef void _calculate_gradient(self, float[:,:] image, float[:,:] imGx, float[:,:] imGy): # Calculate gradients via Robert's cross
        cdef int w = image.shape[1]
        cdef int h = image.shape[0]
        cdef int x0, y0, x1, y1

        cdef int i, j
        with nogil:
            for j in range(1, h): #prange
                y1 = j
                y0 = j - 1
                for i in range(1, w):
                    x1 = i
                    x0 = i - 1    
                    imGx[j,i] = image[y1, x1] - image[y1, x0]
                    imGy[j,i] = image[y1, x1] - image[y0, x1]
                    # as in REF: https://github.com/HenriquesLab/NanoJ-eSRRF/blob/785c71b3bd508c938f63bb780cba47b0f1a5b2a7/resources/liveSRRF.cl under calculateGradient_2point
    
    # @timeit2
    # def calculateRGC(self, int xM, int yM, np.ndarray imGx, np.ndarray imGy):
    #     "Calculate RGC value for a subpixel"
    #     return self._calculateRGC(xM, yM, imGx, imGy)

    cdef float _calculateRGC(self, int xM, int yM, float[:,:] imIntGx, float[:,:] imIntGy) nogil: 

        cdef int w = imIntGx.shape[1]
        cdef int h = imIntGy.shape[0]
        
        cdef float vx, vy, Gx, Gy

        cdef float dx, dy
        cdef float distance, distanceWeight, GdotR, Dk

        cdef float xc = (xM + 0.5) / self.magnification # subpixel in the centre
        cdef float yc = (yM + 0.5) / self.magnification # subpixel in the centre

        cdef float RGC = 0 # Radial Gradient Convergence
        cdef float distanceWeightSum = 0

        cdef int _start = -(<int>(Gx_Gy_MAGNIFICATION * self.fwhm))
        cdef int _end = <int>(Gx_Gy_MAGNIFICATION * self.fwhm + 1) 

        cdef int i, j

        for j in range(_start, _end): #prange
            vy = (<int>(Gx_Gy_MAGNIFICATION * yc) + j) / Gx_Gy_MAGNIFICATION # position in continuous space
            
            if 0 < vy <= h/2 - 1:
            
                for i in range(_start, _end):
                    vx = (<int>(Gx_Gy_MAGNIFICATION * xc) + i) / Gx_Gy_MAGNIFICATION # position in continuous space

                    if 0 < vx <= w/2 - 1:

                        dx = vx - xc
                        dy = vy - yc
                        distance = sqrt(dx * dx + dy * dy)

                        if distance != 0 and distance <= self.tSO:
                            #TODO: change these indexes - turns out this speeded version is generating artifacts
                            Gx = imIntGx[<int>(vy*self.magnification*2) - self.magnification*2 - 1, <int>(vx*self.magnification*2) - self.magnification*2 - 1]
                            Gy = imIntGy[<int>(vy*self.magnification*2) - self.magnification*2 - 1, <int>(vx*self.magnification*2) - self.magnification*2 - 1]
                            distanceWeight = self._calculateDW(distance)
                            distanceWeightSum += distanceWeight
                            GdotR = Gx*dx + Gy*dy

                            if GdotR < 0: # if the vector is pointing inwards
                                Dk = self._calculateDk(Gx, Gy, dx, dy, distance)
                                RGC += Dk * distanceWeight 

        RGC /= distanceWeightSum

        if RGC >= 0:
            RGC = RGC ** self.sensitivity
        else:
            RGC = 0

        return RGC


    cdef float _calculateDW(self, float distance) nogil: # distance weight
        return (distance * exp((-distance * distance) / self.tSS)) ** 4


    cdef float _calculateDk(self, float Gx, float Gy, float dx, float dy, float distance) nogil:
        Dk = fabs(Gy * dx - Gx * dy) / sqrt(Gx * Gx + Gy * Gy) # GMag = sqrt(Gx*Gx + Gy*Gy)
        if isnan(Dk):
            Dk = distance
        Dk = 1 - Dk / distance # if 1: vector pointing exactly to the centre
        return Dk