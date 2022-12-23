# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3

from libc.math cimport pow, log, sqrt, exp, pi, floor, fabs, fmax, fmin, cos, sin

import numpy as np
cimport numpy as np

from cython.parallel import prange

# code based on https://github.com/HenriquesLab/NanoJ-SRRF/blob/master/SRRF/src/nanoj/srrf/java/SRRF.java

cdef class SRRF:
    cdef int magnification, order, symmetryAxis, border, nRingCoordinates
    cdef float spatialRadius, psfWidth, gradRadius
    cdef bint doIntegrateLagTimes, radialityPositivityConstraint, doIntensityWeighting

    cdef float[:] xRingCoordinates, yRingCoordinates

    def __init__(self, int magnification, int order, int symmetryAxis, float spatialRadius, int border, 
        bint doIntegrateLagTimes, bint radialityPositivityConstraint, bint doIntensityWeighting):
        self.magnification = magnification
        self.border = border
        self.symmetryAxis = symmetryAxis
        self.spatialRadius = spatialRadius
        self.border = border
        self.doIntegrateLagTimes = doIntegrateLagTimes
        self.radialityPositivityConstraint = radialityPositivityConstraint
        self.doIntensityWeighting = doIntensityWeighting

        self.nRingCoordinates = symmetryAxis * 2

        self.xRingCoordinates = np.empty(self.nRingCoordinates, dtype=np.float32) # new float[nRingCoordinates];
        self.yRingCoordinates = np.empty(self.nRingCoordinates, dtype=np.float32)
        cdef float angleStep = (pi * 2.) / self.nRingCoordinates
        with nogil:
            for angleIter in range(self.nRingCoordinates):
                self.xRingCoordinates[angleIter] = cos(angleStep * angleIter)
                self.yRingCoordinates[angleIter] = sin(angleStep * angleIter)

    def calculateRadiality(self, im: np.ndarray):
        assert im.ndim == 3

        nFrames = im.shape[0]

        cdef float [:,:,:] imRaw = im.astype(np.float32)
        cdef float [:,:,:] imGx = np.zeros_like(imRaw)
        cdef float [:,:,:] imGy = np.zeros_like(imRaw)
        cdef float [:,:,:] imRad = np.zeros((im.shape[0], im.shape[1]*self.magnification, im.shape[2]*self.magnification), dtype=np.float32)
        cdef float [:,:,:] imIW = np.zeros((im.shape[0], im.shape[1]*self.magnification, im.shape[2]*self.magnification), dtype=np.float32)

        with nogil:
            for n in range(nFrames):
                self._calculate_radiality(imRaw[n,:,:], imRad[n,:,:], imIW[n,:,:], imGx[n,:,:], imGy[n,:,:], 0, 0)

        return imRad, imIW, imGx, imGy

    cdef void _calculate_radiality(self, float[:,:] imRaw, float[:,:] imRad, float[:,:] imIW, float[:,:] imGx, float[:,:] imGy, float shiftX, float shiftY) nogil:
        """
        Note that Gx and Gy are initialized but zeroed
        """

        cdef int w = imRaw.shape[0]
        cdef int h = imRaw.shape[1]
        cdef int i, j, sampleIter
        cdef float x0, y0, xc, yc, GMag, GdotR, xRing, yRing

        # calculate Gx and Gy
        cdef float vGx, vGy, GMagRadius
        cdef float CGH # for Culley Gustafsson Henriques transform 

        # Radiality Variable
        cdef float Dk, DivDFactor = 0
        cdef int isDiverging = -1
 
        for j in range(1, h-1):
            for i in range(1, w-1):
                imGx[i,j] = -imRaw[i-1,j]+imRaw[i+1,j]
                imGy[i,j] = -imRaw[i,j-1]+imRaw[i,j+1]

        for j in range(1 * self.magnification, (h - 1) * self.magnification):
            for i in range(1 * self.magnification, (w - 1) * self.magnification):
                xc = i + 0.5 + shiftX * self.magnification
                yc = j + 0.5 + shiftY * self.magnification

                imIW[i,j] = _interpolate(imRaw, xc, yc, self.magnification)

                # Output
                CGH = 0
                for sampleIter in range(0, self.nRingCoordinates):
                    xRing = self.xRingCoordinates[sampleIter] * self.spatialRadius
                    yRing = self.yRingCoordinates[sampleIter] * self.spatialRadius

                    x0 = xc + xRing
                    y0 = yc + yRing

                    vGx = _interpolate(imGx, x0, y0, self.magnification)
                    vGy = _interpolate(imGy, x0, y0, self.magnification)
                    GMag = sqrt(vGx * vGx + vGy * vGy)
                    GMagRadius = GMag * self.spatialRadius

                    Dk = 1 - self.calculateDk(x0, y0, xc, yc, vGx, vGy, GMag) / self.spatialRadius
                    Dk = Dk * Dk
                    
                    if (vGx * xRing + vGy * yRing) > 0:
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

                
    cdef float calculateDk(self, float x, float y, float xc, float yc, float vGx, float vGy, float vGx2Gy2) nogil:
        if vGx2Gy2 == 0:
            return self.spatialRadius
        else:
            return fabs(vGy * (xc - x) - vGx * (yc - y)) / vGx2Gy2

cdef float cubic(float x) nogil:
    cdef float a = 0.5  # Catmull-Rom interpolation
    cdef float z = 0
    if x < 0: 
        x = -x
    if x < 1:
        z = x * x * (x * (-a + 2.) + (a - 3.)) + 1.
    elif x < 2:
        z = -a * x * x * x + 5. * a * x * x - 8. * a * x + 4.0 * a
    return z

cdef float _interpolate(float[:,:] im, float x, float y, int magnification) nogil:
    cdef int w = im.shape[0]
    cdef int h = im.shape[1]
    x = x / magnification
    y = y / magnification

    if x<1.5 or x>w-1.5 or y<1.5 or y>h-1.5:
        return 0
    
    cdef int u0 = int(floor(x - 0.5))
    cdef int v0 = int(floor(y - 0.5))
    cdef float q = 0
    cdef float p
    cdef int v, u, i, j

    for j in range(4):
        v = v0 - 1 + j
        p = 0
        for i in range(4):
            u = u0 - 1 + i
            p = p + im[u, v] * cubic(x - (u + 0.5))
        q = q + p * cubic(y - (v + 0.5))

    return q
