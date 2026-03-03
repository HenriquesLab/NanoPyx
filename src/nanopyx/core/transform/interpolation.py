"""
Pure-Python / NumPy implementation of the FHT (Fourier) space interpolation
and its tiled variant.

This module mirrors the logic in ``_interpolation.pyx`` without any Cython
dependency, making it easy to test, profile and iterate on the algorithm.
The Cython extension can delegate to these helpers or be kept in sync manually.
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.fft as fft


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mirror_padding_even_square(img: np.ndarray) -> np.ndarray:
    """
    Pad *img* with mirror reflections so the result is a power-of-2 square.

    Equivalent to ``_mirror_padding_even_square_c`` in _interpolation.pyx.
    """
    h, w = img.shape

    bit_w = int(2 ** math.ceil(math.log2(w))) if w > 1 else 1
    bit_h = int(2 ** math.ceil(math.log2(h))) if h > 1 else 1
    output_dim = max(bit_h, bit_w)

    scale_w = math.ceil(output_dim / w)
    if scale_w % 2 == 0:
        scale_w += 1
    scale_h = math.ceil(output_dim / h)
    if scale_h % 2 == 0:
        scale_h += 1

    mid_w = (scale_w - 1) // 2
    mid_h = (scale_h - 1) // 2

    padded = np.zeros((scale_h * h, scale_w * w), dtype=np.float32)

    # Build column and row index arrays for each tile position
    for sw in range(scale_w):
        col_idx = np.arange(w) if (sw - mid_w) % 2 == 0 else np.arange(w - 1, -1, -1)
        for sh in range(scale_h):
            row_idx = np.arange(h) if (sh - mid_h) % 2 == 0 else np.arange(h - 1, -1, -1)
            padded[sh * h:(sh + 1) * h, sw * w:(sw + 1) * w] = img[np.ix_(row_idx, col_idx)]

    x_roi = (scale_w * w - output_dim) // 2
    y_roi = (scale_h * h - output_dim) // 2
    return padded[y_roi:y_roi + output_dim, x_roi:x_roi + output_dim]


def _fht_interpolate_2d(img: np.ndarray, mag: int, do_mirror_padding: bool = True) -> np.ndarray:
    """
    FHT (zero-padded FFT) Fourier interpolation of a single 2-D float32 frame.

    Equivalent to ``_fht_space_interpolation_c`` in _interpolation.pyx.

    Parameters
    ----------
    img : 2-D float32 ndarray
    mag : integer upscaling factor
    do_mirror_padding : if True, apply mirror padding to reduce edge ringing
    """
    img = np.asarray(img, dtype=np.float32)
    orig_min = float(img.min())
    orig_max = float(img.max())

    working = _mirror_padding_even_square(img) if do_mirror_padding else img

    h, w = working.shape
    h_int = h * mag
    w_int = w * mag

    F = fft.rfft2(working)
    del working

    h2 = (h + 1) // 2
    F_int = np.zeros((h_int, w_int // 2 + 1), dtype=np.complex128)
    F_int[0:h2, 0:F.shape[1]] = F[0:h2, :]
    F_int[h_int - (h - h2):h_int, 0:F.shape[1]] = F[h2:h, :]
    del F

    output = fft.irfft2(F_int, s=(h_int, w_int)).astype(np.float32)
    del F_int

    out_min = float(output.min())
    out_max = float(output.max())
    if out_max != out_min:
        scale = (orig_max - orig_min) / (out_max - out_min)
        output -= out_min
        output *= scale
        output += orig_min

    if do_mirror_padding:
        x_roi = (w_int - mag * img.shape[1]) // 2
        y_roi = (h_int - mag * img.shape[0]) // 2
        return output[y_roi:y_roi + mag * img.shape[0], x_roi:x_roi + mag * img.shape[1]]

    return output


# ---------------------------------------------------------------------------
# Tiled interpolation (memory-efficient)
# ---------------------------------------------------------------------------

def fht_interpolate_2d_tiled(
    img: np.ndarray,
    magnification: int,
    tile_size: int = 256,
    overlap: int = 16,
) -> np.ndarray:
    """
    Memory-efficient tiled FHT interpolation of a single 2-D frame.

    The image is split into overlapping ``tile_size × tile_size`` input-pixel
    tiles.  Each tile is interpolated independently (with mirror padding) then
    blended back using a Hann window, giving smooth seams.

    Peak memory per tile: ``O(tile_size² × magnification²)``
    Full-frame peak memory: ``O(H² × W² × magnification²)``

    Parameters
    ----------
    img : 2-D float32 ndarray
    magnification : integer upscaling factor
    tile_size : input-pixel side length for each tile (power of 2 recommended)
    overlap : input-pixel border shared between adjacent tiles (default 16)
    """
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape
    M = int(magnification)
    ts = int(tile_size)
    ov = int(overlap)

    # Trivially small — no tiling benefit
    if h <= ts and w <= ts:
        return _fht_interpolate_2d(img, M, do_mirror_padding=True)

    out_h = h * M
    out_w = w * M
    output = np.zeros((out_h, out_w), dtype=np.float32)
    weight = np.zeros((out_h, out_w), dtype=np.float32)

    step = max(1, ts - 2 * ov)

    # Build tile start positions for one axis
    def starts(dim):
        s = list(range(0, dim, step))
        if s and s[-1] + ts < dim:
            s.append(max(0, dim - ts))
        return s

    row_starts = starts(h)
    col_starts = starts(w)

    for r0 in row_starts:
        r1 = min(r0 + ts, h)
        for c0 in col_starts:
            c1 = min(c0 + ts, w)

            tile = np.ascontiguousarray(img[r0:r1, c0:c1])
            tile_out = _fht_interpolate_2d(tile, M, do_mirror_padding=True)

            th, tw = tile_out.shape

            # Hann window — peak 1 at centre, near-zero at edges
            win_r = np.hanning(th + 2)[1:-1].astype(np.float32)
            win_c = np.hanning(tw + 2)[1:-1].astype(np.float32)
            win = np.outer(win_r, win_c)

            out_r0, out_r1 = r0 * M, r1 * M
            out_c0, out_c1 = c0 * M, c1 * M

            output[out_r0:out_r1, out_c0:out_c1] += tile_out * win
            weight[out_r0:out_r1, out_c0:out_c1] += win

    nz = weight > 0
    output[nz] /= weight[nz]
    return output


# ---------------------------------------------------------------------------
# Public API — mirrors fht_space_interpolation / fht_space_interpolation_tiled
# ---------------------------------------------------------------------------

def fht_space_interpolation_tiled(
    image: np.ndarray,
    magnification: int = 2,
    tile_size: int = 256,
    overlap: int = 16,
    doMirrorPadding: bool = True,   # kept for API compatibility
) -> np.ndarray:
    """
    Memory-efficient tiled Fourier interpolation of 2-D images or 3-D stacks.

    3-D stacks are processed frame-by-frame with a thread pool.

    Parameters
    ----------
    image : 2-D or 3-D ndarray
    magnification : integer upscaling factor
    tile_size : input-pixel tile side length (default 256)
    overlap : overlap border between tiles in input pixels (default 16)
    doMirrorPadding : API compatibility flag (ignored — always True per tile)
    """
    magnification = int(magnification)
    tile_size = int(tile_size)
    overlap = int(overlap)

    if image.ndim == 2:
        img_f32 = np.ascontiguousarray(image, dtype=np.float32)
        return np.ascontiguousarray(
            fht_interpolate_2d_tiled(img_f32, magnification, tile_size, overlap))

    elif image.ndim == 3:
        n_frames = int(image.shape[0])
        out_h = int(image.shape[1]) * magnification
        out_w = int(image.shape[2]) * magnification
        result = np.empty((n_frames, out_h, out_w), dtype=np.float32)

        max_workers = min(4, os.cpu_count() or 1)

        def _process(i):
            frame = np.ascontiguousarray(image[i], dtype=np.float32)
            result[i] = fht_interpolate_2d_tiled(
                frame, magnification, tile_size, overlap)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_process, range(n_frames)))

        return result

    else:
        raise ValueError("Input must be 2-D or 3-D array")


def fht_space_interpolation(
    image: np.ndarray,
    magnification: int = 2,
    doMirrorPadding: bool = True,
    tile_size: int = 0,
    overlap: int = 16,
) -> np.ndarray:
    """
    Fourier interpolation of 2-D images or 3-D stacks.

    Automatically delegates to :func:`fht_space_interpolation_tiled` when
    either spatial dimension exceeds ``tile_size`` (default 256), keeping peak
    memory bounded regardless of input size.

    Parameters
    ----------
    image : 2-D or 3-D ndarray
    magnification : integer upscaling factor
    doMirrorPadding : apply mirror padding before FFT
    tile_size : tile side (input pixels).  0 = auto (256 for large images).
    overlap : overlap border between tiles in input pixels (default 16)
    """
    magnification = int(magnification)
    overlap = int(overlap)

    if image.ndim == 2:
        h, w = int(image.shape[0]), int(image.shape[1])
    elif image.ndim == 3:
        h, w = int(image.shape[1]), int(image.shape[2])
    else:
        raise ValueError("Input must be 2-D or 3-D array")

    effective_tile = int(tile_size) if int(tile_size) > 0 else 256

    if h > effective_tile or w > effective_tile:
        return fht_space_interpolation_tiled(
            image, magnification, effective_tile, overlap, doMirrorPadding)

    # Full-frame path for small images
    if image.ndim == 2:
        img_f32 = np.ascontiguousarray(image, dtype=np.float32)
        return np.ascontiguousarray(
            _fht_interpolate_2d(img_f32, magnification, doMirrorPadding))
    else:
        return np.ascontiguousarray(np.stack([
            _fht_interpolate_2d(
                np.ascontiguousarray(frame, dtype=np.float32),
                magnification, doMirrorPadding)
            for frame in image]))

 