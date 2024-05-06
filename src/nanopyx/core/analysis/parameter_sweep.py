import numpy as np
from tqdm import tqdm
from nanopyx.core.transform.error_map import ErrorMap
from nanopyx.core.analysis.frc import FIRECalculator
from nanopyx.core.analysis.decorr import DecorrAnalysis
from nanopyx.core.transform._le_esrrf import eSRRF
from nanopyx.core.transform.sr_temporal_correlations import calculate_eSRRF_temporal_correlations


# TODO double check this implementation and confirm that this gives the same results as NanoJ-eSRRF
class ParameterSweep:
    def __init__(self, doErrorMapping: bool = True, doFRCMapping: bool = True):
        self.doErrorMapping = doErrorMapping
        self.doFRCMapping = doFRCMapping

    # check for image dimensions in the method
    def run(
        self,
        im: np.array,
        magnification: int,
        sensitivity_array: list,
        radius_array: list,
        temporal_correlation: str = "AVG",
        use_decorr: bool = False,
        n_frames=None
    ):
        RSP_map = np.zeros((len(sensitivity_array), len(radius_array)))
        FRC_map = np.zeros((len(sensitivity_array), len(radius_array)))
        s_size = len(sensitivity_array)
        r_size = len(radius_array)

        with tqdm(total=s_size*r_size, desc="Parameters pairs", unit="pairs") as progress_bar:
            for s in range(s_size):
                for r in range(r_size):
                    rgc_map = eSRRF(verbose=True).run(
                        im, magnification=magnification, radius=radius_array[r], sensitivity=sensitivity_array[s]
                    )
                    if n_frames is None:
                        if self.doErrorMapping:
                            reconstruction = calculate_eSRRF_temporal_correlations(rgc_map, temporal_correlation)
                            RSP_map[s, r] = self.calculate_rsp(im, reconstruction)
                        if self.doFRCMapping:
                            if use_decorr:
                                decorr = DecorrAnalysis()
                                decorr.run_analysis(calculate_eSRRF_temporal_correlations(rgc_map, temporal_correlation))
                                FRC_map[s, r] = decorr.resolution
                            else:
                                rgc_map_odd = rgc_map[1::2, :, :]
                                rgc_map_even = rgc_map[::2, :, :]
                                reconstruction_odd = calculate_eSRRF_temporal_correlations(rgc_map_odd, temporal_correlation)
                                reconstruction_even = calculate_eSRRF_temporal_correlations(rgc_map_even, temporal_correlation)
                                FRC_map[s, r] = self.calculate_frc(reconstruction_odd, reconstruction_even)
                    else:
                        if im.shape[0] % n_frames != 0:
                            n_slices = im.shape[0] // n_frames + 1
                        else:
                            n_slices = im.shape[0] // n_frames

                        RSP_map = np.zeros((n_slices, len(sensitivity_array), len(radius_array)))
                        for i in range(n_slices):
                            if self.doErrorMapping:
                                reconstruction = calculate_eSRRF_temporal_correlations(rgc_map[i*n_frames:(i+1)*n_frames], temporal_correlation)
                                sliced_image = im[i*n_frames:(i+1)*n_frames]
                                print(sliced_image.shape, rgc_map[i*n_frames:(i+1)*n_frames].shape)
                                RSP_map[i, s, r] = self.calculate_rsp(sliced_image, reconstruction)
                                
                            # if self.doFRCMapping:
                            #     if use_decorr:
                            #         decorr = DecorrAnalysis()
                            #         decorr.run_analysis(calculate_eSRRF_temporal_correlations(rgc_map[i*n_frames:(i+1)*n_frames], temporal_correlation))
                            #         FRC_map[i, s, r] = decorr.resolution
                            #     else:
                            #         rgc_map_odd = rgc_map[i*n_frames+1:(i+1)*n_frames:2, :, :]
                            #         rgc_map_even = rgc_map[i*n_frames:(i+1)*n_frames:2, :, :]
                            #         reconstruction_odd = calculate_eSRRF_temporal_correlations(rgc_map_odd, temporal_correlation)
                            #         reconstruction_even = calculate_eSRRF_temporal_correlations(rgc_map_even, temporal_correlation)
                            #         FRC_map[i, s, r] = self.calculate_frc(reconstruction_odd, reconstruction_even)
                        RSP_map = np.mean(RSP_map, axis=0)
                    
                        if self.doFRCMapping:
                            if use_decorr:
                                decorr = DecorrAnalysis()
                                decorr.run_analysis(calculate_eSRRF_temporal_correlations(rgc_map, temporal_correlation))
                                FRC_map[s, r] = decorr.resolution
                            else:
                                rgc_map_odd = rgc_map[1::2, :, :]
                                rgc_map_even = rgc_map[::2, :, :]
                                reconstruction_odd = calculate_eSRRF_temporal_correlations(rgc_map_odd, temporal_correlation)
                                reconstruction_even = calculate_eSRRF_temporal_correlations(rgc_map_even, temporal_correlation)
                                FRC_map[s, r] = self.calculate_frc(reconstruction_odd, reconstruction_even)
                    progress_bar.update()

        print(RSP_map)
        print(FRC_map)
        QnR = self.calculate_qnr_score(RSP_map, FRC_map)

        return QnR

    def calculate_rsp(self, im, reconstruction):
        error_map = ErrorMap()
        error_map.optimise(np.mean(im, axis=0), reconstruction)
        return error_map.getRSP()

    def calculate_frc(self, im_odd, im_even):
        frc_calculator = FIRECalculator()
        fire_nb = frc_calculator.calculate_fire_number(np.asarray(im_odd), np.asarray(im_even))
        return fire_nb

    def logistic_image_conversion(self, im, min_val=None, max_val=None):  # max and min in nm
        # trying change to min and max vals to be taken from "im" instead of user defined

        if min_val is None:
            min_val = np.min(im)
            print(min_val)
        if max_val is None:
            max_val = np.max(im)
            print(max_val)

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
        print(RSP.shape, FRC.shape)
        assert RSP.shape == FRC.shape
        nFRC = self.logistic_image_conversion(FRC)
        print(nFRC)
        QnR = (2 * np.asarray(RSP) * np.asarray(nFRC)) / (np.asarray(RSP) + np.asarray(nFRC))
        return QnR
