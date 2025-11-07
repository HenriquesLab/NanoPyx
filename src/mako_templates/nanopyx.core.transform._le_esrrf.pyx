<%!
schedulers = ['threaded','threaded_guided','threaded_dynamic','threaded_static']
%># cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

import numpy as np

cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport cos, sin

from .__interpolation_tools__ import check_image, value2array
from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array, _fastest_device

from ._le_interpolation_catmull_rom import ShiftAndMagnify
from ._le_roberts_cross_gradients import GradientRobertsCross
from ._le_radial_gradient_convergence import RadialGradientConvergence
from ._interpolation import fht_space_interpolation as dht_interpolation


class eSRRF(LiquidEngine):
    """
    eSRRF using the NanoPyx Liquid Engine and running as a single task.
    """

    def __init__(self, clear_benchmarks=False, testing=False, verbose=True):
        self._designation = "eSRRF_ST"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, verbose=verbose)

    def run(self, image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True, pad_edges: bool = False, run_type=None):
        image = check_image(image)
        return self._run(image, magnification=magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting,
        pad_edges=pad_edges, run_type=run_type)

    def benchmark(self, image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True, pad_edges: bool = False):
        image = check_image(image)
        return super().benchmark(image, magnification=magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, pad_edges=pad_edges)

    def _run_opencl(self, image, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True, pad_edges=False, device=None, mem_div=1):
        """
        @gpu
        """
        runtype = "OpenCL"
        crsm = ShiftAndMagnify(verbose=False)
        rbc = GradientRobertsCross(verbose=False)
        rgc = RadialGradientConvergence(verbose=False)

        magnified_image = dht_interpolation(image, magnification)
        gradient_col, gradient_row = rbc.run(image, run_type=runtype)
        gradient_col_interp = dht_interpolation(gradient_col, magnification*2)
        gradient_row_interp = dht_interpolation(gradient_row, magnification*2)
        radial_gradients = rgc.run(gradient_col_interp, gradient_row_interp, magnified_image, magnification=magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, pad_edges=pad_edges, run_type=runtype)

        return radial_gradients

    % for sch in schedulers:
    def _run_${sch}(self, image, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True, pad_edges=False):
        """
        @cpu
        @threaded
        @cython
        """
        runtype = "${sch}".capitalize()
        crsm = ShiftAndMagnify(verbose=False)
        rbc = GradientRobertsCross(verbose=False)
        rgc = RadialGradientConvergence(verbose=False)

        magnified_image = dht_interpolation(image, magnification)
        gradient_col, gradient_row = rbc.run(image, run_type=runtype)
        gradient_col_interp = dht_interpolation(gradient_col, magnification*2)
        gradient_row_interp = dht_interpolation(gradient_row, magnification*2)
        radial_gradients = rgc.run(gradient_col_interp, gradient_row_interp, magnified_image, magnification=magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, pad_edges=pad_edges, run_type=runtype)

        return radial_gradients
    % endfor

    def _run_unthreaded(self, image, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True, pad_edges=False):
        """
        @cpu
        @cython
        """
        runtype = "Unthreaded"
        rbc = GradientRobertsCross(verbose=False)
        rgc = RadialGradientConvergence(verbose=False)

        magnified_image = dht_interpolation(image, magnification)
        gradient_col, gradient_row = rbc.run(image, run_type=runtype)
        gradient_col_interp = dht_interpolation(gradient_col, magnification*2)
        gradient_row_interp = dht_interpolation(gradient_row, magnification*2)
        radial_gradients = rgc.run(gradient_col_interp, gradient_row_interp, magnified_image, magnification=magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, pad_edges=pad_edges, run_type=runtype)

        return radial_gradients