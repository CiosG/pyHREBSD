"""Optional flat-field correction for unprocessed EBSD patterns."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter, uniform_filter, zoom


def correct_pattern_background(
    image: ArrayLike,
    mode: str = "divide_gaussian",
    sigma_pixels: float = 64.0,
    downsample: int = 8,
) -> NDArray[np.float64]:
    """Remove broad detector shading while retaining raw-pattern detail.

    The broad background is estimated on a reduced grid, then interpolated
    back to the original resolution. Division normalizes local contrast;
    subtraction only removes the broad additive component.
    """
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("pattern must be a finite 2-D image")
    if mode == "none":
        return values
    if mode not in ("divide_gaussian", "subtract_gaussian"):
        raise ValueError("background mode must be none, divide_gaussian, or subtract_gaussian")
    if sigma_pixels <= 0 or downsample < 1 or downsample > min(values.shape):
        raise ValueError("invalid background sigma or downsample factor")
    reduced = values[::downsample, ::downsample]
    background = gaussian_filter(reduced, sigma=sigma_pixels/downsample,
                                 mode="reflect")
    factors = (values.shape[0]/background.shape[0],
               values.shape[1]/background.shape[1])
    background = zoom(background, factors, order=1, mode="nearest")
    if background.shape != values.shape:
        raise ValueError("background interpolation did not match pattern shape")
    if mode == "subtract_gaussian":
        return values-background
    return values/np.maximum(background, 1)-1


def local_mean_std_deviation(image: ArrayLike, radius: int) -> NDArray[np.float64]:
    """LMSD with edge replication, equivalent to the supplied pixelwise loops."""
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or radius < 0:
        raise ValueError("LMSD needs a finite 2-D image and nonnegative radius")
    size = 2*radius+1
    mean = uniform_filter(values, size=size, mode="nearest")
    second_moment = uniform_filter(values*values, size=size, mode="nearest")
    std = np.sqrt(np.maximum(second_moment-mean*mean, 0))
    result = np.zeros_like(values)
    np.divide(values-mean, std, out=result, where=std >= 1e-8)
    return result


def correct_pattern_static_lmsd(
    image: ArrayLike,
    static_background: ArrayLike,
    sigma_factor: float = 0.02,
    lmsd_factor: float = 0.183,
) -> NDArray[np.uint16]:
    """Apply the supplied static-background, Gaussian, LMSD, 16-bit workflow.

    This operates on the original rectangular detector image. Any square
    analysis crop must be taken after this operation so width-dependent
    parameters and percentile normalization match the supplied script.
    """
    pattern = np.asarray(image, dtype=np.float32)
    static = np.asarray(static_background, dtype=np.float32)
    if pattern.ndim != 2 or pattern.shape != static.shape:
        raise ValueError("static background must match the raw detector image")
    if not np.isfinite(pattern).all() or not np.isfinite(static).all():
        raise ValueError("pattern and static background must be finite")
    if sigma_factor <= 0 or lmsd_factor < 0:
        raise ValueError("sigma factor must be positive and LMSD factor nonnegative")
    static = np.where(static == 0, np.float32(1e-6), static)
    divided = pattern/static
    smooth = gaussian_filter(divided, sigma=pattern.shape[1]*sigma_factor)
    residual = divided-smooth
    filtered = local_mean_std_deviation(
        residual, int(pattern.shape[1]*lmsd_factor))
    low, high = np.percentile(filtered, (1.0, 99.0))
    if high <= low:
        raise ValueError("background-corrected pattern has no contrast")
    return (np.clip((filtered-low)/(high-low), 0, 1)*65535).astype(np.uint16)
