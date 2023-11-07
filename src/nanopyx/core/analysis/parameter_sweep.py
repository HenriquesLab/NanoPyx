from nanopyx.core.transform.error_map import ErrorMap
from nanopyx.core.analysis.frc import FIRECalculator
from nanopyx.core.transform._le_radial_gradient_convergence import RadialGradientConvergence as RGC
from nanopyx.core.transform.sr_temporal_correlations import calculate_eSRRF_temporal_correlations
import numpy as np


# TODO double check this implementation and confirm that this gives the same results as NanoJ-eSRRF
class ParameterSweep:
    def __init__(self, doErrorMapping: bool = True, doFRCMapping: bool = True):
        self.doErrorMapping = doErrorMapping
        self.doFRCMapping = doFRCMapping

    # check for image dimensions in the method
    def run(self, im: np.array, sensitivity_array: np.array, radius_array: np.array, temporal_correlation: str = "AVG"):
        RSP_map = np.zeros((len(sensitivity_array), len(radius_array)))
        FRC_map = np.zeros((len(sensitivity_array), len(radius_array)))
        s_size = len(sensitivity_array)
        r_size = len(radius_array)

        for s in range(s_size):
            for r in range(r_size):
                rgc = RGC(radius=radius_array[r], sensitivity=sensitivity_array[s])
                if self.doErrorMapping:
                    rgc_map = rgc.calculate(im)[0]
                    reconstruction = calculate_eSRRF_temporal_correlations(rgc_map, temporal_correlation)
                    RSP_map[s, r] = self.calculate_rsp(im, reconstruction)
                if self.doFRCMapping:
                    rgc_map_odd = rgc.calculate(im[1::2, :, :])[0]
                    rgc_map_even = rgc.calculate(im[::2, :, :])[0]
                    reconstruction_odd = calculate_eSRRF_temporal_correlations(rgc_map_odd, temporal_correlation)
                    reconstruction_even = calculate_eSRRF_temporal_correlations(rgc_map_even, temporal_correlation)
                    FRC_map[s, r] = self.calculate_frc(reconstruction_odd, reconstruction_even)

        QnR = self.calculate_qnr_score(RSP_map, FRC_map)

        return QnR

    def calculate_rsp(self, im, reconstruction):
        error_map = ErrorMap()
        error_map.optimise(np.mean(im, axis=0), reconstruction)
        return error_map.getRSP()

    def calculate_frc(self, im_odd, im_even):
        frc_calculator = FIRECalculator(pixel_size=20, units="nm")
        fire_nb = frc_calculator.calculate_fire_number(np.asarray(im_odd), np.asarray(im_even))
        return fire_nb

    def logistic_image_conversion(self, im, min_val=50, max_val=200):  # max and min in nm
        M1 = 0.075
        M2 = 0.925
        A1 = np.log((1 - M1) / M1)
        A2 = np.log((1 - M2) / M2)
        x0 = (A2 * max_val - A1 * min_val) / (A2 - A1)
        k = 1 / (x0 - max_val) * A1

        im_out = []
        for image in im:
            normalized_image = 1 / (np.exp(-k * (image - x0)) + 1)
            im_out.append(normalized_image)

        return np.asarray(im_out)

    def calculate_qnr_score(self, RSP: np.array, FRC: np.array):
        assert RSP.shape == FRC.shape
        nFRC = self.logistic_image_conversion(FRC)
        QnR = (2 * np.asarray(RSP) * np.asarray(nFRC)) / (np.asarray(RSP) + np.asarray(nFRC))
        return QnR
