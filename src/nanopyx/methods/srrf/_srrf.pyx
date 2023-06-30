# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

from ...liquid._le_radiality import Radiality
from ...core.transform.sr_temporal_correlations import *

import numpy as np
cimport numpy as np

from tqdm import tqdm

class SRRF:
    # cdef Radiality radiality
    # cdef float[:] _shift_x, _shift_y
    # cdef int magnification, border
    # cdef float ringRadius
    # cdef bint radialityPositivityConstraint, doIntensityWeighting

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

        self.radiality = Radiality()

    def calculate(self, dataset: np.ndarray, int frames_per_timepoint, int SRRForder = 1):

        if frames_per_timepoint == 0:
            frames_per_timepoint = dataset.shape[0]
        elif frames_per_timepoint > dataset.shape[0]:
            frames_per_timepoint = dataset.shape[0]

        data_srrf = []
        data_intensity = []

        with tqdm(total=dataset.shape[0] // frames_per_timepoint, desc="Calculating SRRF", unit="frame") as pbar:
            for i in range(dataset.shape[0] // frames_per_timepoint):
                data_block = dataset[i*frames_per_timepoint:(i+1)*frames_per_timepoint]
                data_block_radiality, data_block_intensity = self.radiality.run(data_block, magnification=self.magnification, ringRadius=self.ringRadius, border=self.border, radialityPositivityConstraint=self.radialityPositivityConstraint, doIntensityWeighting=self.doIntensityWeighting)[:2]

                data_block_srrf = calculate_SRRF_temporal_correlations(([data_block_radiality]), SRRForder)
                data_block_intensity = calculate_SRRF_temporal_correlations([data_block_intensity], SRRForder)

                # data_block_srrf = np.asarray(data_block_radiality).mean(axis=0) 
                # data_block_intensity = np.asarray(data_block_intensity).mean(axis=0)

                data_srrf.append(data_block_srrf)
                data_intensity.append(data_block_intensity)

                pbar.update(1)

        return np.asarray(data_srrf), np.asarray(data_intensity)
