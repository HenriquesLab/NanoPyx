# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True

from libc.math cimport sqrt, fabs, exp, isnan, floor

from ..transform.interpolation.catmull_rom cimport _interpolate
from ..transform.image_magnify import cv2_zoom as zoom

import numpy as np
cimport numpy as np

from cython.parallel import prange

cdef int Gx_Gy_MAGNIFICATION = 2

cdef class RadialGradientConvergence:

    cdef int magnification
    cdef float fwhm
    cdef float sensitivity
    cdef float tSS # two sigma squared
    cdef float tSO # two sigma plus one
    cdef bint doIntensityWeighting


    def __init__(self, magnification: int, full_width_half_maximum: float, sensitivity: float, doIntensityWeighting: bool):
        self.magnification = magnification
        self.fwhm = full_width_half_maximum
        self.sensitivity = sensitivity
        self.doIntensityWeighting = doIntensityWeighting
        
        cdef float sigma = full_width_half_maximum / 2.355
        self.tSS = 2 * sigma * sigma
        self.tSO = 2 * sigma + 1


    cdef void _single_frame_RGC_map(self, float[:,:] imRaw, float[:,:] imRad, float[:,:] imInt, float[:,:] imGx, float[:,:] imGy):
        cdef int w = imRaw.shape[1]
        cdef int h = imRaw.shape[0]
        cdef int magnification = self.magnification
        cdef int yM, xM

        imInt[:,:] = zoom(imRaw, self.magnification) # using the OpenCV2 interpolation

        self._calculate_gradient(imInt, imGx, imGy) # calculate gradients of the interpolated image

        with nogil:
            for yM in prange(magnification, h * magnification):
                for xM in range(magnification, w * magnification):
                    if self.doIntensityWeighting:
                        imRad[yM, xM] = self._calculateRGC(xM, yM, w, h, imGx, imGy)
                    else:
                        imRad[yM, xM] = self._calculateRGC(xM, yM, w, h, imGx, imGy) * imInt[xM, yM]


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
    

    cdef float _calculateRGC(self, int xM, int yM, int w, int h, float[:,:] imGx, float[:,:] imGy) nogil:
        
        cdef float vx, vy, Gx, Gy

        cdef float dx, dy
        cdef float distance, distanceWeight, GdotR, Dk

        cdef float xc = (xM + 0.5) / self.magnification # subpixel in the centre
        cdef float yc = (yM + 0.5) / self.magnification # subpixel in the centre

        cdef float RGC = 0 # Radial Gradient Convergence
        cdef float distanceWeightSum = 0

        cdef int _start = -(<int>(Gx_Gy_MAGNIFICATION * self.fwhm))
        cdef int _end = <int>(Gx_Gy_MAGNIFICATION * self.fwhm + 1) + 1

        cdef int i, j

        for j in range(_start, _end):
            vy = (<int>(Gx_Gy_MAGNIFICATION * yc) + j) / Gx_Gy_MAGNIFICATION # position in continuous space
            
            if 0 <= vy <= h - 1:
            
                for i in range(_start, _end):
                    vx = (<int>(Gx_Gy_MAGNIFICATION * xc) + i) / Gx_Gy_MAGNIFICATION # position in continuous space

                    if 0 <= vx <= w - 1:

                        dx = vx - xc
                        dy = vy - yc
                        distance = sqrt(dx * dx + dy * dy)

                        if distance != 0 and distance <= self.tSO:
                            Gx = _interpolate(imGx, (vx * self.magnification) / 2, (vy * self.magnification) / 2) # get interpolated value (in continuous space) via Catmull-Rom interpolation
                            Gy = _interpolate(imGy, (vx * self.magnification) / 2, (vy * self.magnification) / 2)
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