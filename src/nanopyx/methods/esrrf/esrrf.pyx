# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

from nanopyx.core.transform.sr_radial_gradient_convergence cimport RadialGradientConvergence as RGC
# from nanopyx.liquid._le_radial_gradient_convergence import RadialGradientConvergence as RGC
from nanopyx.core.transform.sr_temporal_correlations import calculate_eSRRF_temporal_correlations
from nanopyx.core.transform.parameter_sweep import ParameterSweep
from ..drift_alignment.estimator import DriftEstimator

import numpy as np
cimport numpy as np

from tqdm import tqdm

cdef class eSRRF:

    cdef RGC rgc
    cdef int magnification
    cdef float radius, sensitivity, tSS, tSO
    cdef bint doIntensityWeighting, doParameterSweep, doDriftCorrection

    def __init__(self, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1 , doIntensityWeighting: bool = True, doParameterSweep: bool = False, doDriftCorrection: bool = False):

        self.magnification = magnification
        self.radius = radius
        self.sensitivity = sensitivity
        self.doIntensityWeighting = doIntensityWeighting
        self.doParameterSweep = doParameterSweep
        self.doDriftCorrection = doDriftCorrection

        cdef float sigma = radius / 2.355
        self.tSS = 2 * sigma * sigma
        self.tSO = 2 * sigma + 1

        self.rgc = RGC(magnification, radius, sensitivity, doIntensityWeighting)
    
    def calculate(self, img: np.array, frames_per_timepoint: int, temporal_correlation: str = "AVG" ):

        if self.doDriftCorrection:
            estimate_drif = DriftEstimator()
            img = estimate_drif.estimate(img, apply=True)
        
        if frames_per_timepoint == 0:
            frames_per_timepoint = img.shape[0]
        elif frames_per_timepoint > img.shape[0]:
            frames_per_timepoint = img.shape[0]

        data_esrrf = []
        data_intensity = []

        with tqdm(total=img.shape[0] // frames_per_timepoint, desc="Calculating eSRRF", unit="frame") as pbar:
            for i in range(img.shape[0] // frames_per_timepoint):
                data_block = img[i*frames_per_timepoint:(i+1)*frames_per_timepoint]

                data_block_rgc, data_block_intensity = self.rgc.calculate(data_block)[:2]

                data_block_esrrf = calculate_eSRRF_temporal_correlations(data_block_rgc, temporal_correlation)
                data_block_intensity = calculate_eSRRF_temporal_correlations(data_block_intensity, temporal_correlation)

                data_esrrf.append(data_block_esrrf)
                data_intensity.append(data_block_intensity)

                pbar.update(1)

        return np.asarray(data_esrrf), np.asarray(data_intensity)


    # for now not doing any frame per timepoint analysis - taking the entire stack for the parameter sweep
    def run_parameter_sweep(self, img: np.array, sensitivity_array: np.array, radius_array: np.array, doErrorMapping: bool = True, doFRCMapping: bool = True, temporal_correlation: str = "AVG"):
        
        ps = ParameterSweep(doErrorMapping, doFRCMapping)
        QnR = ps.run(img, sensitivity_array, radius_array, temporal_correlation)
        best_sensitivity, best_radius = np.unravel_index(np.argmax(QnR), QnR.shape)

        print("The best sensitivity found is", sensitivity_array[best_sensitivity], "and the best radius is", radius_array[best_radius])
        return QnR