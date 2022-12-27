# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True

from libc.math cimport sqrt, pi, floor, fabs, cos, sin

import numpy as np
cimport numpy as np

from nanopyx.core.transform.interpolation.catmull_rom cimport _interpolate

from cython.parallel import prange

# code based on https://github.com/HenriquesLab/NanoJ-SRRF/blob/master/SRRF/src/nanoj/srrf/java/SRRF.java

cdef class CalculateRadiality:
    cdef int magnification, symmetryAxis, border, nRingCoordinates
    cdef float ringRadius, psfWidth, gradRadius
    cdef bint doIntegrateLagTimes, radialityPositivityConstraint, doIntensityWeighting
    cdef float[12] xRingCoordinates, yRingCoordinates

    def __init__(self, int magnification, float ringRadius, int border, bint radialityPositivityConstraint, bint doIntensityWeighting):
        self.magnification = magnification
        self.border = border
        self.ringRadius = ringRadius * magnification
        self.border = border
        self.radialityPositivityConstraint = radialityPositivityConstraint
        self.doIntensityWeighting = doIntensityWeighting
        self.nRingCoordinates = 12

        cdef float angleStep = (pi * 2.) / self.nRingCoordinates
        with nogil:
            for angleIter in range(self.nRingCoordinates):
                self.xRingCoordinates[angleIter] = cos(angleStep * angleIter) * self.ringRadius
                self.yRingCoordinates[angleIter] = sin(angleStep * angleIter) * self.ringRadius

    def calculate(self, im: np.ndarray):
        assert im.ndim == 3

        nFrames = im.shape[0]

        cdef float [:,:,:] imRaw = im.astype(np.float32)
        cdef float [:,:,:] imGx = np.zeros_like(imRaw)
        cdef float [:,:,:] imGy = np.zeros_like(imRaw)
        cdef float [:,:,:] imRad = np.zeros((im.shape[0], im.shape[1]*self.magnification, im.shape[2]*self.magnification), dtype=np.float32)
        cdef float [:,:,:] imIW = np.zeros((im.shape[0], im.shape[1]*self.magnification, im.shape[2]*self.magnification), dtype=np.float32)

        cdef int n
        with nogil:
            for n in prange(nFrames):
                self._calculate_radiality(imRaw[n,:,:], imRad[n,:,:], imIW[n,:,:], imGx[n,:,:], imGy[n,:,:], 0, 0)

        return imRad, imIW, imGx, imGy

    cdef void _calculate_radiality(self, float[:,:] imRaw, float[:,:] imRad, float[:,:] imIW, float[:,:] imGx, float[:,:] imGy, float shiftX, float shiftY) nogil:
        """
        Note that Gx and Gy are initialized but zeroed
        """

        cdef int w = imRaw.shape[0]
        cdef int h = imRaw.shape[1]
        cdef int i, j, sampleIter
        cdef float x0, y0, xc, yc, GMag, xRing, yRing

        # calculate Gx and Gy
        cdef float vGx, vGy
        cdef float CGH # for Culley Gustafsson Henriques transform 

        # Radiality Variable
        cdef float Dk, DivDFactor = 0
 
        for j in range(1, h-1):
            for i in range(1, w-1):
                imGx[i,j] = -imRaw[i-1,j]+imRaw[i+1,j]
                imGy[i,j] = -imRaw[i,j-1]+imRaw[i,j+1]

        for j in range((1 + self.border) * self.magnification, (h - 1 - self.border) * self.magnification):
            for i in range((1 + self.border) * self.magnification, (w - 1 - self.border) * self.magnification):
                xc = i + 0.5 + shiftX * self.magnification
                yc = j + 0.5 + shiftY * self.magnification

                imIW[i,j] = _interpolate(imRaw, xc / self.magnification, yc / self.magnification)

                # Output
                CGH = 0
                for sampleIter in range(0, self.nRingCoordinates):
                    xRing = self.xRingCoordinates[sampleIter]
                    yRing = self.yRingCoordinates[sampleIter]

                    x0 = xc + xRing
                    y0 = yc + yRing

                    vGx = _interpolate(imGx, x0 / self.magnification, y0 / self.magnification)
                    vGy = _interpolate(imGy, x0 / self.magnification, y0 / self.magnification)
                    GMag = sqrt(vGx * vGx + vGy * vGy)

                    Dk = 1 - self._calculateDk(x0, y0, xc, yc, vGx, vGy, GMag) / self.ringRadius
                    Dk = Dk * Dk
                    
                    if (vGx * xRing + vGy * yRing) > 0: # inwards or outwards vector
                        DivDFactor -= Dk
                    else:
                        DivDFactor += Dk
                
                DivDFactor /= self.nRingCoordinates

                if self.radialityPositivityConstraint:
                    CGH = max(DivDFactor, 0)
                else:
                    CGH = DivDFactor
                
                if self.doIntensityWeighting:
                    imRad[i,j] = CGH

                else:
                    imRad[i,j] = CGH * imIW[i,j]

                
    cdef float _calculateDk(self, float x, float y, float xc, float yc, float vGx, float vGy, float vGx2Gy2) nogil:
        if vGx2Gy2 == 0:
            return self.ringRadius
        else:
            return fabs(vGy * (xc - x) - vGx * (yc - y)) / vGx2Gy2
