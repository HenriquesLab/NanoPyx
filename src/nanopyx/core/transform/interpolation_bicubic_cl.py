import numpy as np
import pyopencl as cl


def magnify(image: np.ndarray, magnification: int) -> np.ndarray:
    """
    Magnify an image using bicubic interpolation.
    :param image: 2D array of shape (rows, cols)
    :param magnification: magnification factor
    :return: 2D array of shape (rows * magnification, cols * magnification)

    >>> image = np.array(np.arange(32*32), dtype=np.float32).reshape(32, 32)
    >>> image_magnified = magnify(image, 2)
    >>> image_magnified.shape
    (64, 64)
    """
    assert image.ndim == 2
    assert magnification > 0

    rows, cols = image.shape
    rows_magnified = rows * magnification
    cols_magnified = cols * magnification

    image = image.astype(np.float32)
    image_magnified = np.zeros((rows_magnified, cols_magnified), dtype=np.float32)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags

    image_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
    image_magnified_buf = cl.Buffer(ctx, mf.WRITE_ONLY, image_magnified.nbytes)

    prg = cl.Program(
        ctx,
        """
    __kernel void magnify(
        __global const float *image,
        __global float *image_magnified,
        const int magnification
    ) {
        int rM = get_global_id(0);
        int cM = get_global_id(1);
        int rows_magnified = get_global_size(0);
        int cols_magnified = get_global_size(1);

        float r = rM / magnification;
        float c = cM / magnification;

        int r0 = (int) r;
        int c0 = (int) c;
        int r1 = r0 + 1;
        int c1 = c0 + 1;

        float dr = r - r0;
        float dc = c - c0;

        // bicubic interpolation
        float a0 = image[r0 * cols_magnified + c0];
        float a1 = image[r0 * cols_magnified + c1];
        float a2 = image[r1 * cols_magnified + c0];
        float a3 = image[r1 * cols_magnified + c1];

        float b0 = a1 - a0;
        float b1 = a2 - a0;
        float b2 = a3 - a1 - a2 + a0;

        float c0 = a0;
        float c1 = b0;
        float c2 = b1;
        float c3 = b2;

        float d0 = c0;
        float d1 = c1 + c2 + c3;
        float d2 = 2 * c2 + 3 * c3;
        float d3 = 3 * c3;

        float value = d0 + d1 * dc + d2 * dc * dc + d3 * dc * dc * dc;

        image_magnified[rM * cols_magnified + cM] = value;
    }
    """,
    ).build()

    prg.magnify(
        queue,
        image_magnified.shape,
        None,
        image_buf,
        image_magnified_buf,
        np.int32(magnification),
    )

    cl.enqueue_copy(queue, image_magnified, image_magnified_buf)

    return image_magnified
