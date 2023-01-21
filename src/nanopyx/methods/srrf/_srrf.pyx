# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=False

from ...core.transform.radiality cimport Radiality

import numpy as np
cimport numpy as np

cdef class SRRF:
    cdef Radiality radiality
    cdef float[:] _shift_x, _shift_y
    cdef int magnification, border
    cdef float ringRadius
    cdef bint radialityPositivityConstraint, doIntensityWeighting

    def __init__(self, magnification: int = 5, 
                 ringRadius: float = 0.5, 
                 border: int = 0, 
                 radialityPositivityConstraint: bool = True, 
                 doIntensityWeighting: bool = True):

        self.magnification = magnification
        self.ringRadius = ringRadius
        self.border = border
        self.radialityPositivityConstraint = radialityPositivityConstraint
        self.doIntensityWeighting = doIntensityWeighting

        self.radiality = Radiality(magnification, ringRadius, border, radialityPositivityConstraint, doIntensityWeighting)

    def calculate(self, dataset: np.ndarray, int frames_per_timepoint):

        data_srrf = []
        data_intensity = []

        for i in range(dataset.shape[0] // frames_per_timepoint):
            data_block = dataset[i*frames_per_timepoint:(i+1)*frames_per_timepoint]

            data_block_radiality, data_block_intensity = self.radiality.calculate(data_block)[:2]

            data_block_srrf = np.asarray(data_block_radiality).mean(axis=0)
            data_block_intensity = np.asarray(data_block_intensity).mean(axis=0)
            
            data_srrf.append(data_block_srrf)
            data_intensity.append(data_block_intensity)

        return np.asarray(data_srrf), np.asarray(data_intensity)
