import numpy as np

cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport cos, sin

from .__interpolation_tools__ import check_image, value2array
from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array

from ._le_interpolation_catmull_rom import ShiftAndMagnify
from ._le_roberts_cross_gradients import GradientRobertsCross
from ._le_radial_gradient_convergence import RadialGradientConvergence


class eSRRF(LiquidEngine):
    """
    eSRRF using the NanoPyx Liquid Engine and running as a single task.
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "eSRRF_ST"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, 
                        opencl_=True, unthreaded_=True, threaded_=True, threaded_static_=True, 
                        threaded_dynamic_=True, threaded_guided_=True)
        self._default_benchmarks = {'OpenCL': {"(['shape(100, 150, 150)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [16875000.0, 0.6182407079995755, 0.6122367919997487, 0.6093217500001629], "(['shape(50, 100, 100)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [3750000.0, 0.3028541249987029, 0.2802749999991647, 0.2996904169995105]}, 'Threaded': {"(['shape(100, 150, 150)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [16875000.0, 11.06756008299999, 11.480632083001183, 11.368770667000717], "(['shape(50, 100, 100)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [3750000.0, 2.490110000000641, 2.6822768330002873, 2.5428189579997706]}, 'Threaded_dynamic': {"(['shape(100, 150, 150)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [16875000.0, 9.366981125000166, 9.478710332999981, 10.355995709000126], "(['shape(50, 100, 100)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [3750000.0, 2.020473291999224, 2.0464198749996285, 2.1126812500006054]}, 'Threaded_guided': {"(['shape(100, 150, 150)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [16875000.0, 9.59399004199986, 9.394610874998762, 10.42694429200128], "(['shape(50, 100, 100)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [3750000.0, 2.1240765410002496, 2.1175940839984833, 2.127043415999651]}, 'Threaded_static': {"(['shape(100, 150, 150)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [16875000.0, 11.325379959000202, 11.439641291000953, 12.059574375000011], "(['shape(50, 100, 100)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [3750000.0, 2.4486127919990395, 2.5282963329991617, 2.7061019579996355]}, 'Unthreaded': {"(['shape(100, 150, 150)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [16875000.0, 50.56610608399933, 50.84249891699983, 51.47774041699995], "(['shape(50, 100, 100)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [3750000.0, 10.916643833001217, 11.070652250000421, 11.099240584000654]}}

    def run(self, image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True, run_type=None):
        image = check_image(image)
        return self._run(image, magnification=magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, run_type=run_type)

    def benchmark(self, image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True):
        image = check_image(image)
        return super().benchmark(image, magnification=magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting)

    def _run_opencl(self, image, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True, device=None, mem_div=1):
        # TODO doIntensityWeighting is irrelevant on gpu
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

        input_cl = cl.Buffer(cl_ctx, mf.READ_ONLY, image[0:max_slices, :, :].nbytes)
        output_cl = cl.Buffer(cl_ctx, mf.WRITE_ONLY, np.empty(output_shape, dtype=np.float32).nbytes)
        magnified_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, np.empty((max_slices, output_shape[1], output_shape[2]), dtype=np.float32).nbytes)
        col_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, np.empty((max_slices, image.shape[1], image.shape[2]), dtype=np.float32).nbytes)
        row_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, np.empty((max_slices, image.shape[1], image.shape[2]), dtype=np.float32).nbytes)
        col_magnified_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, np.empty((max_slices, image.shape[1]*magnification*2, image.shape[2]*magnification*2), dtype=np.float32).nbytes)
        row_magnified_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, np.empty((max_slices, image.shape[1]*magnification*2, image.shape[2]*magnification*2), dtype=np.float32).nbytes)
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

    def _run_threaded(self, image, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True):
        runtype = "Threaded"
        crsm = ShiftAndMagnify()
        rbc = GradientRobertsCross()
        rgc = RadialGradientConvergence()
        
        magnified_image = crsm.run(image, 0, 0, magnification, magnification, run_type=runtype)
        gradient_col, gradient_row = rbc.run(image, run_type=runtype)
        gradient_col_interp = crsm.run(gradient_col, 0, 0, magnification*2, magnification*2, run_type=runtype)
        gradient_row_interp = crsm.run(gradient_row, 0, 0, magnification*2, magnification*2, run_type=runtype)
        radial_gradients = rgc.run(gradient_col_interp, gradient_row_interp, magnified_image, magnification=magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, run_type=runtype)

        return radial_gradients

    def _run_threaded_dynamic(self, image, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True):
        runtype = "Threaded_dynamic"
        crsm = ShiftAndMagnify()
        rbc = GradientRobertsCross()
        rgc = RadialGradientConvergence()
        
        magnified_image = crsm.run(image, 0, 0, magnification, magnification, run_type=runtype)
        gradient_col, gradient_row = rbc.run(image, run_type=runtype)
        gradient_col_interp = crsm.run(gradient_col, 0, 0, magnification*2, magnification*2, run_type=runtype)
        gradient_row_interp = crsm.run(gradient_row, 0, 0, magnification*2, magnification*2, run_type=runtype)
        radial_gradients = rgc.run(gradient_col_interp, gradient_row_interp, magnified_image, magnification=magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, run_type=runtype)

        return radial_gradients

    def _run_threaded_static(self, image,  magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True):
        runtype = "Threaded_static"
        crsm = ShiftAndMagnify()
        rbc = GradientRobertsCross()
        rgc = RadialGradientConvergence()
        
        magnified_image = crsm.run(image, 0, 0, magnification, magnification, run_type=runtype)
        gradient_col, gradient_row = rbc.run(image, run_type=runtype)
        gradient_col_interp = crsm.run(gradient_col, 0, 0, magnification*2, magnification*2, run_type=runtype)
        gradient_row_interp = crsm.run(gradient_row, 0, 0, magnification*2, magnification*2, run_type=runtype)
        radial_gradients = rgc.run(gradient_col_interp, gradient_row_interp, magnified_image, magnification=magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, run_type=runtype)

        return radial_gradients

    def _run_threaded_guided(self, image, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True):
        runtype = "Threaded_guided"
        crsm = ShiftAndMagnify()
        rbc = GradientRobertsCross()
        rgc = RadialGradientConvergence()
        
        magnified_image = crsm.run(image, 0, 0, magnification, magnification, run_type=runtype)
        gradient_col, gradient_row = rbc.run(image, run_type=runtype)
        gradient_col_interp = crsm.run(gradient_col, 0, 0, magnification*2, magnification*2, run_type=runtype)
        gradient_row_interp = crsm.run(gradient_row, 0, 0, magnification*2, magnification*2, run_type=runtype)
        radial_gradients = rgc.run(gradient_col_interp, gradient_row_interp, magnified_image, magnification=magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, run_type=runtype)

        return radial_gradients

    def _run_unthreaded(self, image, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True):
        runtype = "Unthreaded"
        crsm = ShiftAndMagnify()
        rbc = GradientRobertsCross()
        rgc = RadialGradientConvergence()
        
        magnified_image = crsm.run(image, 0, 0, magnification, magnification, run_type=runtype)
        gradient_col, gradient_row = rbc.run(image, run_type=runtype)
        gradient_col_interp = crsm.run(gradient_col, 0, 0, magnification*2, magnification*2, run_type=runtype)
        gradient_row_interp = crsm.run(gradient_row, 0, 0, magnification*2, magnification*2, run_type=runtype)
        radial_gradients = rgc.run(gradient_col_interp, gradient_row_interp, magnified_image, magnification=magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, run_type=runtype)

        return radial_gradients
