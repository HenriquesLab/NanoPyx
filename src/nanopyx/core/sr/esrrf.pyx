# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True

from libc.math cimport sqrt, fabs, exp, isnan

import numpy as np
cimport numpy as np

from nanopyx.core.transform.interpolation.catmull_rom cimport _interpolate
from skimage.transform import rescale as ski_rescale
from nanopyx.core.transform.image_magnify import *

import cython
from cython.parallel import prange


# code based on https://github.com/HenriquesLab/NanoJ-eSRRF/blob/master/resources/liveSRRF.cl

cdef class CalculateRGC:
    cdef int magnification, GxGyMagnification
    cdef float radius, sensitivity, tSS, tSO
    cdef bint doIntWeighting

    def __init__(self, int magnification, float radius, float sensitivity, bint doIntWeighting):
        self.magnification = magnification
        self.radius = radius
        self.sensitivity = sensitivity
        self.GxGyMagnification = 2
        self.tSS = 2 * (radius / 2.355) ** 2
        self.tSO = 2 * radius / 2.355 + 1
        self.doIntWeighting = doIntWeighting


    def calculate(self, im: np.ndarray):
        assert im.ndim == 3

        nFrames = im.shape[0]

        cdef float [:,:,:] imRaw = im.astype(np.float32)
        cdef float [:,:,:] imRad = np.zeros((im.shape[0], im.shape[1]*self.magnification, im.shape[2]*self.magnification), dtype=np.float32)
        cdef float [:,:,:] imInt = np.zeros((im.shape[0], im.shape[1]*self.magnification, im.shape[2]*self.magnification), dtype=np.float32) # interpolated image
        cdef float [:,:,:] imGx = np.zeros_like(imInt) # Gradient of the interpolated image
        cdef float [:,:,:] imGy = np.zeros_like(imInt)

        cdef int n
        for n in range(nFrames):
            self.single_frame_RGC_map(imRaw[n,:,:], imRad[n,:,:], imInt[n,:,:], imGx[n,:,:], imGy[n,:,:])

        return imRad, imInt, imGx, imGy


    cdef void single_frame_RGC_map(self, float[:,:] imRaw, float[:,:] imRad, float[:,:] imInt, float[:,:] imGx, float[:,:] imGy):
        cdef int w = imRaw.shape[0]
        cdef int h = imRaw.shape[1]
        cdef int magnification = self.magnification
        cdef int yM, xM

        imInt[:,:] = cv2_zoom(imRaw, self.magnification) # using the OpenCV2 interpolation

        self._calculate_grad(imInt, imGx, imGy) # calculate gradients of the interpolated image

        for yM in range(magnification, h * magnification):
            for xM in range(magnification, w * magnification):
                if self.doIntWeighting:
                    imRad[xM, yM] = self._calculateRGC(xM, yM, w, h, imInt, imGx, imGy)
                else:
                    imRad[xM, yM] = self._calculateRGC(xM, yM, w, h, imInt, imGx, imGy) * imInt[xM, yM]



    cdef void _calculate_grad(self, float[:,:] imInt, float[:,:] imGx, float[:,:] imGy): # Calculate gradients via Robert's cross
        cdef int wInt = imInt.shape[0]
        cdef int hInt = imInt.shape[1]

        cdef int i, j
        for j in range(1, hInt - 1):
            for i in range(1, wInt - 1):
                imGx[i,j] = -imInt[i,j] + imInt[i-1,j-1]
                imGy[i,j] = -imInt[i,j-1] + imInt[i-1,j]


    cdef float _calculateRGC(self, int xM, int yM, int w, int h, float[:,:] imGx, float[:,:] imGy):
        cdef float xc, yc

        cdef float vx, vy, Gx, Gy

        cdef float dx, dy
        cdef float distance, distanceWeight, distanceWeightSum, GdotR, Dk

        cdef float RGC # Radial Gradient Convergence

        xc = (xM + 0.5) / self.magnification # subpixel in the centre
        yc = (yM + 0.5) / self.magnification

        RGC = 0
        distanceWeightSum = 0


        for j in range(-int(self.GxGyMagnification * self.radius), int(self.GxGyMagnification * self.radius + 1) + 1):
            vy = (int(self.GxGyMagnification * yc) + j) / self.GxGyMagnification # position in continuous space

            if vy > 0 and vy < h:
                for i in range(-int(self.GxGyMagnification * self.radius), int(self.GxGyMagnification * self.radius + 1) + 1):
                    vx = (int(self.GxGyMagnification * xc) + i) / self.GxGyMagnification

                    if vx > 0 and vx < w:
                        dx = vx - xc
                        dy = vy - yc
                        distance = sqrt(dx * dx + dy * dy)

                        if distance != 0 and distance <= self.tSO:
                            Gx = _interpolate(imGx, (vx * self.magnification) / 2, (vy * self.magnification) / 2) # get interpolated value (in continuous space) via Catmull-Rom interpolation
                            Gy = _interpolate(imGy, (vx * self.magnification) / 2, (vy * self.magnification) / 2)
                            distanceWeight = self._calculateDW(distance)
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

