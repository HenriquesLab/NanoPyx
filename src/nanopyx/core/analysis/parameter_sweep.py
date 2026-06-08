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
        if n_frames is not None:
            if n_frames <= 0:
                raise ValueError("n_frames must be a positive integer or None")
            n_frames = int(n_frames)

        RSP_map = np.zeros((len(sensitivity_array), len(radius_array)))
        FRC_map = np.zeros((len(sensitivity_array), len(radius_array)))
        s_size = len(sensitivity_array)
        r_size = len(radius_array)

        with tqdm(total=s_size*r_size, desc="Parameters pairs", unit="pairs") as progress_bar:
            for s in range(s_size):
                for r in range(r_size):
                    if n_frames is None:
                        rgc_map = eSRRF(verbose=True).run(
                            im, magnification=magnification, radius=radius_array[r], sensitivity=sensitivity_array[s]
                        )
                        rgc_map = self._as_frame_stack(rgc_map)
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
                        RSP_values = []
                        FRC_values = []

                        for start in range(0, im.shape[0], n_frames):
                            stop = start + n_frames
                            sliced_image = im[start:stop]
                            rgc_map = eSRRF(verbose=True).run(
                                sliced_image,
                                magnification=magnification,
                                radius=radius_array[r],
                                sensitivity=sensitivity_array[s],
                            )
                            rgc_map = self._as_frame_stack(rgc_map)

                            if self.doErrorMapping:
                                reconstruction = calculate_eSRRF_temporal_correlations(rgc_map, temporal_correlation)
                                RSP_values.append(self.calculate_rsp(sliced_image, reconstruction))

                            if self.doFRCMapping:
                                if use_decorr:
                                    decorr = DecorrAnalysis()
                                    decorr.run_analysis(calculate_eSRRF_temporal_correlations(rgc_map, temporal_correlation))
                                    FRC_values.append(decorr.resolution)
                                elif rgc_map.shape[0] >= 2:
                                    rgc_map_odd = rgc_map[1::2, :, :]
                                    rgc_map_even = rgc_map[::2, :, :]
                                    reconstruction_odd = calculate_eSRRF_temporal_correlations(rgc_map_odd, temporal_correlation)
                                    reconstruction_even = calculate_eSRRF_temporal_correlations(rgc_map_even, temporal_correlation)
                                    FRC_values.append(self.calculate_frc(reconstruction_odd, reconstruction_even))

                        if RSP_values:
                            RSP_map[s, r] = np.nanmean(RSP_values)
                        if FRC_values:
                            FRC_map[s, r] = np.nanmean(FRC_values)
                    progress_bar.update()

        QnR = self.calculate_qnr_score(RSP_map, FRC_map)

        return QnR

    def _as_frame_stack(self, im):
        im = np.asarray(im)
        if im.ndim == 2:
            return np.expand_dims(im, axis=0)
        return im

    def calculate_rsp(self, im, reconstruction):
        error_map = ErrorMap()
        error_map.optimise(np.mean(im, axis=0), reconstruction)
        return error_map.getRSP()

    def calculate_frc(self, im_odd, im_even):
        frc_calculator = FIRECalculator()
        fire_nb = frc_calculator.calculate_fire_number(np.asarray(im_odd), np.asarray(im_even))
        return fire_nb

    def logistic_image_conversion(self, im, min_val=None, max_val=None):  # max and min in nm
        im = np.asarray(im, dtype=np.float32)
        finite_mask = np.isfinite(im)
        if not np.any(finite_mask):
            return np.zeros_like(im, dtype=np.float32)

        if min_val is None:
            min_val = np.min(im[finite_mask])
        if max_val is None:
            max_val = np.max(im[finite_mask])

        if not np.isfinite(min_val) or not np.isfinite(max_val):
            return np.zeros_like(im, dtype=np.float32)

        if min_val == max_val:
            return np.where(finite_mask, 0.5, 0).astype(np.float32)

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

        return np.nan_to_num(np.asarray(im_out), nan=0.0, posinf=1.0, neginf=0.0)

    def calculate_qnr_score(self, RSP: np.array, FRC: np.array):
        assert RSP.shape == FRC.shape
        nFRC = self.logistic_image_conversion(FRC)
        denominator = np.asarray(RSP) + np.asarray(nFRC)
        QnR = np.divide(
            2 * np.asarray(RSP) * np.asarray(nFRC),
            denominator,
            out=np.zeros_like(denominator, dtype=np.float32),
            where=denominator != 0,
        )
        return np.nan_to_num(QnR, nan=0.0, posinf=0.0, neginf=0.0)
