"""CUDA implementation of the static-background/LMSD EBSD correction."""

from __future__ import annotations

import numpy as np


def correct_pattern_static_lmsd_gpu(
    image,
    static_background,
    sigma_factor=0.02,
    lmsd_factor=0.183,
    gpu_device_id=0,
):
    """Return a CPU uint16 pattern matching the supplied correction method."""
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter, uniform_filter

    pattern_cpu = np.asarray(image)
    if pattern_cpu.ndim != 2 or sigma_factor <= 0 or lmsd_factor < 0:
        raise ValueError("invalid raw pattern or correction parameters")
    cp.cuda.Device(gpu_device_id).use()
    pattern = cp.asarray(pattern_cpu, dtype=cp.float32)
    static = cp.asarray(static_background, dtype=cp.float32)
    if pattern.shape != static.shape:
        raise ValueError("static background must match the raw detector image")
    static = cp.where(static == 0, cp.float32(1e-6), static)
    divided = pattern/static
    residual = divided-gaussian_filter(divided, sigma=pattern.shape[1]*sigma_factor)
    radius = int(pattern.shape[1]*lmsd_factor)
    size = 2*radius+1
    # Float64 moments avoid cancellation in the wide LMSD window.
    values = residual.astype(cp.float64)
    mean = uniform_filter(values, size=size, mode="nearest")
    second_moment = uniform_filter(values*values, size=size, mode="nearest")
    std = cp.sqrt(cp.maximum(second_moment-mean*mean, 0))
    filtered = cp.where(std >= 1e-8, (values-mean)/cp.where(std >= 1e-8, std, 1), 0)
    low, high = cp.percentile(filtered, (1.0, 99.0))
    if float(high-low) <= 0:
        raise ValueError("background-corrected pattern has no contrast")
    normalized = (cp.clip((filtered-low)/(high-low), 0, 1)*65535).astype(cp.uint16)
    return cp.asnumpy(normalized)
