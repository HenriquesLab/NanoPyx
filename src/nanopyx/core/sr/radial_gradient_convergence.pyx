# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=False

from libc.math cimport sqrt, fabs, exp, isnan, floor

from ..transform.interpolation.catmull_rom cimport _interpolate, Interpolator
from ..transform.image_magnify import cv2_zoom as zoom
# from ..transform.image_magnify import fourier_zoom as zoom
from ..utils.time.timeit import timeit2


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

    @timeit2
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
        cdef float [:,:,:] imGx = np.zeros_like(imInt) # Gradient of the interpolated image
        cdef float [:,:,:] imGy = np.zeros_like(imInt)

        cdef int n
        for n in range(nFrames):
            self._single_frame_RGC_map(imRaw[n,:,:], imRad[n,:,:], imInt[n,:,:], imGx[n,:,:], imGy[n,:,:])

        return imRad, imInt, imGx, imGy


    cdef void _single_frame_RGC_map(self, float[:,:] imRaw, float[:,:] imRad, float[:,:] imInt, float[:,:] imGx, float[:,:] imGy):
        """
        Calculate the RGC map of an image frame.
        :param imRaw: the frame to calculate the RGC map of
        :param imRad: the RGC map of the frame (previously initialised as a 2D array the size of imRaw x magnification)
        :param imInt: the interpolated image (previously initialised as a 2D array the size of imRaw x magnification)
        :param imGx: the intensity gradients of the interpolated image in the horizontal direction (same size as imInt)
        :param imGy: the intensity gradients of the interpolated image in the vertical direction (same size as imInt)
        :return: the RGC of the image frame.
        """

        cdef int w = imRaw.shape[1]
        cdef int h = imRaw.shape[0]
        cdef int magnification = self.magnification
        cdef int yM, xM

        #TODO: change interpolation technique
        cdef Interpolator interpolator = Interpolator(imRaw)
        imInt[:,:] = interpolator._magnify(self.magnification) 

        self._calculate_gradient(imInt, imGx, imGy) # calculate gradients of the interpolated image

        with nogil:
            for yM in prange(magnification, h * magnification): 
                for xM in range(magnification, w * magnification):
                    if self.doIntensityWeighting:
                        imRad[yM, xM] = self._calculateRGC(xM, yM, imGx, imGy) * imInt[yM, xM]
                    else:
                        imRad[yM, xM] = self._calculateRGC(xM, yM, imGx, imGy) 


    cdef void _calculate_gradient(self, float[:,:] image, float[:,:] imGx, float[:,:] imGy): # Calculate gradients via Robert's cross
        cdef int w = image.shape[1]
        cdef int h = image.shape[0]
        cdef int x0, y0, x1, y1

        cdef int i, j
        with nogil:
            for j in prange(1, h):
                y1 = j
                y0 = j - 1
                for i in range(1, w):
                    x1 = i
                    x0 = i - 1    
                    imGx[j,i] = image[y1, x1] - image[y1, x0]
                    imGy[j,i] = image[y1, x1] - image[y0, x1]
                    # as in REF: https://github.com/HenriquesLab/NanoJ-eSRRF/blob/785c71b3bd508c938f63bb780cba47b0f1a5b2a7/resources/liveSRRF.cl under calculateGradient_2point
    
    @timeit2
    def calculateRGC(self, int xM, int yM, np.ndarray imGx, np.ndarray imGy):
        "Calculate RGC value for a subpixel"
        return self._calculateRGC(xM, yM, imGx, imGy)

    cdef float _calculateRGC(self, int xM, int yM, float[:,:] imGx, float[:,:] imGy) nogil: 

        cdef int w = imGx.shape[1]
        cdef int h = imGx.shape[0]
        
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

        for j in prange(_start, _end): 
            vy = (<int>(Gx_Gy_MAGNIFICATION * yc) + j) / Gx_Gy_MAGNIFICATION # position in continuous space
            
            if 0 < vy <= h - 1:
            
                for i in range(_start, _end):
                    vx = (<int>(Gx_Gy_MAGNIFICATION * xc) + i) / Gx_Gy_MAGNIFICATION # position in continuous space

                    if 0 < vx <= w - 1:

                        dx = vx - xc
                        dy = vy - yc
                        distance = sqrt(dx * dx + dy * dy)

                        if distance != 0 and distance <= self.tSO:
                            Gx = _interpolate(imGx, vx * self.magnification, vy * self.magnification) # get interpolated value (in continuous space) via Catmull-Rom interpolation
                            Gy = _interpolate(imGy, vx * self.magnification, vy * self.magnification)
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