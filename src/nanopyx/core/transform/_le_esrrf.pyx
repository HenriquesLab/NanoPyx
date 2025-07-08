# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np

cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport cos, sin

from .__interpolation_tools__ import check_image, value2array
from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array, _fastest_device

from ._le_convolution import Convolution
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

    def run(self, image, magnification: int = 5, grad_magnification: int = 2, radius: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True, run_type=None):
        image = check_image(image)
        return self._run(image, magnification=magnification, grad_magnification=grad_magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, run_type=run_type)

    def benchmark(self, image, magnification: int = 5, grad_magnification: int = 2, radius: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True):
        image = check_image(image)
        return super().benchmark(image, magnification=magnification, grad_magnification=grad_magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting)

    def _run_opencl(self, image, magnification=5, grad_magnification=2, radius=1.5, sensitivity=1, doIntensityWeighting=True, device=None, mem_div=1):
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

        # needs input image, 2 conv kernels,  first interpolation output, roberts cross output, magnified gradients and output image
        
        total_memory = (3*image[0, :, :].nbytes) + (2*np.zeros((2,2), dtype=np.float32).nbytes) + (2*np.zeros((1, output_shape[1], output_shape[2]), dtype=np.float32).nbytes) + (2*np.zeros((1, output_shape[1]*grad_magnification, output_shape[2]*grad_magnification), dtype=np.float32).nbytes)
        output_image = np.zeros(output_shape, dtype=np.float32)
        
        max_slices = int((dc.global_mem_size // total_memory)/mem_div)
        max_slices = self._check_max_slices(image, max_slices)

        kernelx = np.array([[0, -1], [1, 0]]).astype(np.float32)
        kernely = np.array([[-1, 0], [0, 1]]).astype(np.float32)

        offset = 0
        xyoffset = -0.5
        angle = np.pi/4

        mf = cl.mem_flags
        input_cl = cl.Buffer(cl_ctx, mf.READ_ONLY, self._check_max_buffer_size(image[0:max_slices, :, :].nbytes, dc, max_slices))
        output_cl = cl.Buffer(cl_ctx, mf.WRITE_ONLY, self._check_max_buffer_size(np.empty((max_slices, output_shape[1], output_shape[2]), dtype=np.float32).nbytes, dc, max_slices))
        magnified_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(np.empty((max_slices, output_shape[1], output_shape[2]), dtype=np.float32).nbytes, dc, max_slices))
        col_kernel_cl = cl.Buffer(cl_ctx, mf.READ_ONLY, self._check_max_buffer_size(np.empty_like(kernely).nbytes, dc, max_slices))
        row_kernel_cl = cl.Buffer(cl_ctx, mf.READ_ONLY, self._check_max_buffer_size(np.empty_like(kernelx).nbytes, dc, max_slices))
        cl.enqueue_copy(cl_queue, col_kernel_cl, kernely).wait()
        cl.enqueue_copy(cl_queue, row_kernel_cl, kernelx).wait()
        col_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(np.empty((max_slices, image.shape[1], image.shape[2]), dtype=np.float32).nbytes, dc, max_slices))
        row_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(np.empty((max_slices, image.shape[1], image.shape[2]), dtype=np.float32).nbytes, dc, max_slices))
        col_magnified_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(np.empty((max_slices, image.shape[1]*magnification*grad_magnification, image.shape[2]*magnification*grad_magnification), dtype=np.float32).nbytes, dc, max_slices))
        row_magnified_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(np.empty((max_slices, image.shape[1]*magnification*grad_magnification, image.shape[2]*magnification*grad_magnification), dtype=np.float32).nbytes, dc, max_slices))
        cl.enqueue_copy(cl_queue, input_cl, image[0:max_slices,:,:]).wait()

        cr_code = self._get_cl_code("_le_interpolation_catmull_rom_.cl", device['DP'])
        cr_prg = cl.Program(cl_ctx, cr_code).build(options=["-cl-mad-enable -cl-fast-relaxed-math"])
        cr_knl = cr_prg.shiftAndMagnify

        conv_code = self._get_cl_code("_le_convolution.cl", device['DP'])
        conv_prg = cl.Program(cl_ctx, conv_code).build(options=["-cl-mad-enable -cl-fast-relaxed-math"])
        conv_knl = conv_prg.conv2d_2

        rgc_code = self._get_cl_code("_le_radial_gradient_convergence.cl", device['DP'])
        rgc_prg = cl.Program(cl_ctx, rgc_code).build(options=["-cl-mad-enable -cl-fast-relaxed-math"])
        rgc_knl = rgc_prg.calculate_rgc

        margin = int(radius*magnification)
        lowest_row = margin # TODO discuss edges calculation
        highest_row = output_shape[1] - margin
        lowest_col = margin
        highest_col =  output_shape[2] - margin

        for i in range(0, image.shape[0], max_slices):
            if image.shape[0] - i >= max_slices:
                n_slices = max_slices
            else:
                n_slices = image.shape[0] - i

            cr_knl(cl_queue,
                (n_slices, int(image.shape[1]*magnification), int(image.shape[2]*magnification)),
                None, 
                input_cl,
                magnified_cl,
                np.float32(0),
                np.float32(0),
                np.float32(magnification),
                np.float32(magnification)).wait()

            conv_knl(cl_queue,
                (n_slices, image.shape[1], image.shape[2]), 
                None, 
                input_cl,
                col_gradients_cl,
                col_kernel_cl,
                np.int32(2)).wait()

            conv_knl(cl_queue,
                (n_slices, image.shape[1], image.shape[2]), 
                None, 
                input_cl,
                row_gradients_cl,
                row_kernel_cl,
                np.int32(2)).wait()

            cr_knl(cl_queue,
                (n_slices, int(image.shape[1]*magnification*grad_magnification), int(image.shape[2]*magnification*grad_magnification)), 
                None, 
                col_gradients_cl,
                col_magnified_gradients_cl,
                np.float32(0),
                np.float32(0),
                np.float32(magnification*grad_magnification),
                np.float32(magnification*grad_magnification)).wait()

            cr_knl(cl_queue,
                (n_slices, int(image.shape[1]*magnification*grad_magnification), int(image.shape[2]*magnification*grad_magnification)), 
                None, 
                row_gradients_cl,
                row_magnified_gradients_cl,
                np.float32(0),
                np.float32(0),
                np.float32(magnification*grad_magnification),
                np.float32(magnification*grad_magnification)).wait()

            rgc_knl(cl_queue,
                (n_slices, highest_row - lowest_row, highest_col - lowest_col),
                None,
                col_magnified_gradients_cl,
                row_magnified_gradients_cl,
                magnified_cl,
                output_cl,
                np.int32(output_shape[2]),  # Ensure correct dimensions
                np.int32(output_shape[1]),
                np.int32(magnification),
                np.float32(grad_magnification),
                np.float32(radius),
                np.float32(2 * (radius / 2.355) + 1),  # Match sigma calculation
                np.float32(2 * (radius / 2.355) ** 2),
                np.float32(sensitivity),
                np.int32(doIntensityWeighting),
                np.float32(offset),
                np.float32(xyoffset),
                np.float32(angle)).wait()
            
            cl.enqueue_copy(cl_queue, output_image[i:i+n_slices,:,:], output_cl).wait()

            if i+n_slices<image.shape[0]:
                cl.enqueue_copy(cl_queue, input_cl, image[i+n_slices:i+2*n_slices,:,:]).wait()

            cl_queue.finish()


        return output_image

    def _run_threaded(self, image, magnification=5, grad_magnification=2, radius=1.5, sensitivity=1, doIntensityWeighting=True):
        """
        @cpu
        @threaded
        @cython
        """
        runtype = "threaded".capitalize()
        crsm = ShiftAndMagnify(verbose=False)
        conv = Convolution(verbose=False)
        rgc = RadialGradientConvergence(verbose=False)

        kernelx = np.array([[0, -1], [1, 0]]).astype(np.float32)
        kernely = np.array([[-1, 0], [0, 1]]).astype(np.float32)
        
        magnified_image = crsm.run(image, 0, 0, magnification, magnification, run_type=runtype)
        gradient_col = conv.run(image, kernely, run_type=runtype)
        gradient_row = conv.run(image, kernelx, run_type=runtype)
        gradient_col_interp = crsm.run(gradient_col, 0, 0, magnification*grad_magnification, magnification*grad_magnification, run_type=runtype)
        gradient_row_interp = crsm.run(gradient_row, 0, 0, magnification*grad_magnification, magnification*grad_magnification, run_type=runtype)
        radial_gradients = rgc.run(gradient_col_interp, gradient_row_interp, magnified_image, magnification=magnification, grad_magnification=grad_magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, offset=0, xyoffset=-0.5, angle=np.pi/4, run_type=runtype)

        return radial_gradients
    def _run_threaded_guided(self, image, magnification=5, grad_magnification=2, radius=1.5, sensitivity=1, doIntensityWeighting=True):
        """
        @cpu
        @threaded
        @cython
        """
        runtype = "threaded_guided".capitalize()
        crsm = ShiftAndMagnify(verbose=False)
        conv = Convolution(verbose=False)
        rgc = RadialGradientConvergence(verbose=False)

        kernelx = np.array([[0, -1], [1, 0]]).astype(np.float32)
        kernely = np.array([[-1, 0], [0, 1]]).astype(np.float32)
        
        magnified_image = crsm.run(image, 0, 0, magnification, magnification, run_type=runtype)
        gradient_col = conv.run(image, kernely, run_type=runtype)
        gradient_row = conv.run(image, kernelx, run_type=runtype)
        gradient_col_interp = crsm.run(gradient_col, 0, 0, magnification*grad_magnification, magnification*grad_magnification, run_type=runtype)
        gradient_row_interp = crsm.run(gradient_row, 0, 0, magnification*grad_magnification, magnification*grad_magnification, run_type=runtype)
        radial_gradients = rgc.run(gradient_col_interp, gradient_row_interp, magnified_image, magnification=magnification, grad_magnification=grad_magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, offset=0, xyoffset=-0.5, angle=np.pi/4, run_type=runtype)

        return radial_gradients
    def _run_threaded_dynamic(self, image, magnification=5, grad_magnification=2, radius=1.5, sensitivity=1, doIntensityWeighting=True):
        """
        @cpu
        @threaded
        @cython
        """
        runtype = "threaded_dynamic".capitalize()
        crsm = ShiftAndMagnify(verbose=False)
        conv = Convolution(verbose=False)
        rgc = RadialGradientConvergence(verbose=False)

        kernelx = np.array([[0, -1], [1, 0]]).astype(np.float32)
        kernely = np.array([[-1, 0], [0, 1]]).astype(np.float32)
        
        magnified_image = crsm.run(image, 0, 0, magnification, magnification, run_type=runtype)
        gradient_col = conv.run(image, kernely, run_type=runtype)
        gradient_row = conv.run(image, kernelx, run_type=runtype)
        gradient_col_interp = crsm.run(gradient_col, 0, 0, magnification*grad_magnification, magnification*grad_magnification, run_type=runtype)
        gradient_row_interp = crsm.run(gradient_row, 0, 0, magnification*grad_magnification, magnification*grad_magnification, run_type=runtype)
        radial_gradients = rgc.run(gradient_col_interp, gradient_row_interp, magnified_image, magnification=magnification, grad_magnification=grad_magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, offset=0, xyoffset=-0.5, angle=np.pi/4, run_type=runtype)

        return radial_gradients
    def _run_threaded_static(self, image, magnification=5, grad_magnification=2, radius=1.5, sensitivity=1, doIntensityWeighting=True):
        """
        @cpu
        @threaded
        @cython
        """
        runtype = "threaded_static".capitalize()
        crsm = ShiftAndMagnify(verbose=False)
        conv = Convolution(verbose=False)
        rgc = RadialGradientConvergence(verbose=False)

        kernelx = np.array([[0, -1], [1, 0]]).astype(np.float32)
        kernely = np.array([[-1, 0], [0, 1]]).astype(np.float32)
        
        magnified_image = crsm.run(image, 0, 0, magnification, magnification, run_type=runtype)
        gradient_col = conv.run(image, kernely, run_type=runtype)
        gradient_row = conv.run(image, kernelx, run_type=runtype)
        gradient_col_interp = crsm.run(gradient_col, 0, 0, magnification*grad_magnification, magnification*grad_magnification, run_type=runtype)
        gradient_row_interp = crsm.run(gradient_row, 0, 0, magnification*grad_magnification, magnification*grad_magnification, run_type=runtype)
        radial_gradients = rgc.run(gradient_col_interp, gradient_row_interp, magnified_image, magnification=magnification, grad_magnification=grad_magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, offset=0, xyoffset=-0.5, angle=np.pi/4, run_type=runtype)

        return radial_gradients

    def _run_unthreaded(self, image, magnification=5, grad_magnification=2, radius=1.5, sensitivity=1, doIntensityWeighting=True):
        """
        @cpu
        @cython
        """
        runtype = "Unthreaded"
        crsm = ShiftAndMagnify(verbose=False)
        conv = Convolution(verbose=False)
        rgc = RadialGradientConvergence(verbose=False)

        kernelx = np.array([[0, -1], [1, 0]]).astype(np.float32)
        kernely = np.array([[-1, 0], [0, 1]]).astype(np.float32)
        
        magnified_image = crsm.run(image, 0, 0, magnification, magnification, run_type=runtype)
        gradient_col = conv.run(image, kernely, run_type=runtype)
        gradient_row = conv.run(image, kernelx, run_type=runtype)
        gradient_col_interp = crsm.run(gradient_col, 0, 0, magnification*grad_magnification, magnification*grad_magnification, run_type=runtype)
        gradient_row_interp = crsm.run(gradient_row, 0, 0, magnification*grad_magnification, magnification*grad_magnification, run_type=runtype)
        radial_gradients = rgc.run(gradient_col_interp, gradient_row_interp, magnified_image, magnification=magnification, grad_magnification=grad_magnification, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, offset=0, xyoffset=-0.5, angle=np.pi/4, run_type=runtype)

        return radial_gradients