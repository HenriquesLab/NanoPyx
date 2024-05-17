<%!
schedulers = ['threaded','threaded_guided','threaded_dynamic','threaded_static']
%># cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

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


class eSRRF(LiquidEngine):
    """
    eSRRF using the NanoPyx Liquid Engine and running as a single task.
    """

    def __init__(self, clear_benchmarks=False, testing=False, verbose=True):
        self._designation = "eSRRF_ST"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, verbose=verbose)

    def run(self, image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True, run_type=None):
        image = check_image(image)
        return self._run(image, magnification=magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, run_type=run_type)

    def benchmark(self, image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True):
        image = check_image(image)
        return super().benchmark(image, magnification=magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting)

    def _run_opencl(self, image, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True, device=None, mem_div=1):
        """
        @gpu
        """
        if device is None:
            device = _fastest_device

        # TODO doIntensityWeighting is irrelevant on gpu2
        cl_ctx = cl.Context([device['device']])
        dc = device['device']
        cl_queue = cl.CommandQueue(cl_ctx)

        output_shape = (image.shape[0], int(image.shape[1]*magnification), int(image.shape[2]*magnification))

        # needs input image, first interpolation output, roberts cross output, magnified gradients and output image
        
        total_memory = (3*image[0, :, :].nbytes) + (2*np.zeros((1, output_shape[1], output_shape[2]), dtype=np.float32).nbytes) + (2*np.zeros((1, output_shape[1]*2, output_shape[2]*2), dtype=np.float32).nbytes)
        output_image = np.zeros(output_shape, dtype=np.float32)
        
        max_slices = int((dc.global_mem_size // total_memory)/mem_div)
        max_slices = self._check_max_slices(image, max_slices)

        mf = cl.mem_flags
        input_cl = cl.Buffer(cl_ctx, mf.READ_ONLY, self._check_max_buffer_size(image[0:max_slices, :, :].nbytes, dc, max_slices))
        output_cl = cl.Buffer(cl_ctx, mf.WRITE_ONLY, self._check_max_buffer_size(np.empty((max_slices, output_shape[1], output_shape[2]), dtype=np.float32).nbytes, dc, max_slices))
        magnified_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(np.empty((max_slices, output_shape[1], output_shape[2]), dtype=np.float32).nbytes, dc, max_slices))
        col_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(np.empty((max_slices, image.shape[1], image.shape[2]), dtype=np.float32).nbytes, dc, max_slices))
        row_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(np.empty((max_slices, image.shape[1], image.shape[2]), dtype=np.float32).nbytes, dc, max_slices))
        col_magnified_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(np.empty((max_slices, image.shape[1]*magnification*2, image.shape[2]*magnification*2), dtype=np.float32).nbytes, dc, max_slices))
        row_magnified_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(np.empty((max_slices, image.shape[1]*magnification*2, image.shape[2]*magnification*2), dtype=np.float32).nbytes, dc, max_slices))
        cl.enqueue_copy(cl_queue, input_cl, image[0:max_slices,:,:]).wait()

        cr_code = self._get_cl_code("_le_interpolation_catmull_rom_.cl", device['DP'])
        cr_prg = cl.Program(cl_ctx, cr_code).build(options=["-cl-mad-enable -cl-fast-relaxed-math"])
        cr_knl = cr_prg.shiftAndMagnify

        rc_code = self._get_cl_code("_le_roberts_cross_gradients.cl", device['DP'])
        rc_prg = cl.Program(cl_ctx, rc_code).build(options=["-cl-mad-enable -cl-fast-relaxed-math"])
        rc_knl = rc_prg.gradient_roberts_cross

        rgc_code = self._get_cl_code("_le_radial_gradient_convergence.cl", device['DP'])
        rgc_prg = cl.Program(cl_ctx, rgc_code).build(options=["-cl-mad-enable -cl-fast-relaxed-math"])
        rgc_knl = rgc_prg.calculate_rgc

        for i in range(0, image.shape[0], max_slices):
            if image.shape[0] - i >= max_slices:
                n_slices = max_slices
            else:
                n_slices = image.shape[0] - i

            cr_knl(cl_queue,
                (n_slices, int(image.shape[1]*magnification), int(image.shape[2]*magnification)),
                self.get_work_group(dc, (n_slices, image.shape[1]*magnification, image.shape[2]*magnification)), 
                input_cl,
                magnified_cl,
                np.float32(0),
                np.float32(0),
                np.float32(magnification),
                np.float32(magnification)).wait()

            rc_knl(cl_queue,
                (n_slices,),
                None,
                input_cl,
                col_gradients_cl,
                row_gradients_cl,
                np.int32(image.shape[1]),
                np.int32(image.shape[2])).wait()

            cr_knl(cl_queue,
                (n_slices, int(image.shape[1]*magnification*2), int(image.shape[2]*magnification*2)), 
                self.get_work_group(dc, (n_slices, image.shape[1]*magnification*2, image.shape[2]*magnification*2)), 
                col_gradients_cl,
                col_magnified_gradients_cl,
                np.float32(0),
                np.float32(0),
                np.float32(magnification*2),
                np.float32(magnification*2)).wait()

            cr_knl(cl_queue,
                (n_slices, int(image.shape[1]*magnification*2), int(image.shape[2]*magnification*2)), 
                self.get_work_group(dc, (n_slices, image.shape[1]*magnification*2, image.shape[2]*magnification*2)), 
                row_gradients_cl,
                row_magnified_gradients_cl,
                np.float32(0),
                np.float32(0),
                np.float32(magnification*2),
                np.float32(magnification*2)).wait()

            rgc_knl(cl_queue,
                (n_slices, (image.shape[1]*magnification-magnification*2) - magnification*2, image.shape[2]*magnification-magnification*2 - magnification*2),
                self.get_work_group(dc, (n_slices, (image.shape[1]*magnification-magnification*2) - magnification*2, image.shape[2]*magnification-magnification*2 - magnification*2)),
                col_magnified_gradients_cl,
                row_magnified_gradients_cl,
                magnified_cl,
                output_cl,
                np.int32(image.shape[2]*magnification),
                np.int32(image.shape[1]*magnification),
                np.int32(magnification),
                np.float32(2),
                np.float32(radius),
                np.float32(2 * (radius / 2.355) + 1),
                np.float32(2 * (radius / 2.355) * (radius / 2.355)),
                np.float32(sensitivity),
                np.int32(doIntensityWeighting)).wait()
            
            cl.enqueue_copy(cl_queue, output_image[i:i+n_slices,:,:], output_cl).wait()

            if i+n_slices<image.shape[0]:
                cl.enqueue_copy(cl_queue, input_cl, image[i+n_slices:i+2*n_slices,:,:]).wait()

            cl_queue.finish()


        return output_image

    % for sch in schedulers:
    def _run_${sch}(self, image, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True):
        """
        @cpu
        @threaded
        @cython
        """
        runtype = "${sch}".capitalize()
        crsm = ShiftAndMagnify(verbose=False)
        rbc = GradientRobertsCross(verbose=False)
        rgc = RadialGradientConvergence(verbose=False)
        
        magnified_image = crsm.run(image, 0, 0, magnification, magnification, run_type=runtype)
        gradient_col, gradient_row = rbc.run(image, run_type=runtype)
        gradient_col_interp = crsm.run(gradient_col, 0, 0, magnification*2, magnification*2, run_type=runtype)
        gradient_row_interp = crsm.run(gradient_row, 0, 0, magnification*2, magnification*2, run_type=runtype)
        radial_gradients = rgc.run(gradient_col_interp, gradient_row_interp, magnified_image, magnification=magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, run_type=runtype)

        return radial_gradients
    % endfor

    def _run_unthreaded(self, image, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True):
        """
        @cpu
        @cython
        """
        runtype = "Unthreaded"
        crsm = ShiftAndMagnify(verbose=False)
        rbc = GradientRobertsCross(verbose=False)
        rgc = RadialGradientConvergence(verbose=False)
        
        magnified_image = crsm.run(image, 0, 0, magnification, magnification, run_type=runtype)
        gradient_col, gradient_row = rbc.run(image, run_type=runtype)
        gradient_col_interp = crsm.run(gradient_col, 0, 0, magnification*2, magnification*2, run_type=runtype)
        gradient_row_interp = crsm.run(gradient_row, 0, 0, magnification*2, magnification*2, run_type=runtype)
        radial_gradients = rgc.run(gradient_col_interp, gradient_row_interp, magnified_image, magnification=magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, run_type=runtype)

        return radial_gradients