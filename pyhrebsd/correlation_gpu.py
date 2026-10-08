"""Batched CUDA ROI FFT correlation matching the CPU OpenXY path.

Reference ROI FFTs stay on the selected GPU across pattern pairs. Each scan
pattern is transferred once, and all valid ROIs are correlated in one batch.
Only the six measured values per ROI return to the CPU deformation fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .correlation import CorrelationResult, _roi_slice


@dataclass(frozen=True)
class GPUPreparedReference:
    image_shape: tuple[int, int]
    roi_size: int
    origins: tuple[tuple[int, int], ...]
    a_fft: object
    a_centered: object
    a_norm: object
    frequency_filter: object | None
    window: object | None
    gpu_device_id: int


def _cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "GPU ROI correlation requires CuPy with CUDA support; select "
            "roi_device='cpu' or run in a CUDA-enabled Python environment") from exc
    return cp


def _extract_rois(image, origins, size, cp):
    tops = cp.asarray([row for row, _ in origins], dtype=cp.int64)
    lefts = cp.asarray([col for _, col in origins], dtype=cp.int64)
    offsets = cp.arange(size, dtype=cp.int64)
    return image[tops[:, None, None] + offsets[None, :, None],
                 lefts[:, None, None] + offsets[None, None, :]]


def prepare_reference_gpu(reference, centers, roi_size, frequency_filter=None,
                          window=None, gpu_device_id=0):
    """Cache batched reference ROIs, FFTs, and statistics on one GPU."""
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim != 2:
        raise ValueError("reference must be a 2-D image")
    if (frequency_filter is None) != (window is None):
        raise ValueError("frequency_filter and window must be provided together")
    if frequency_filter is not None and (
        np.shape(frequency_filter) != (roi_size, roi_size) or
        np.shape(window) != (roi_size, roi_size)
    ):
        raise ValueError("filter and window must match the ROI size")
    origins = []
    for x, y in np.asarray(centers, dtype=np.float64):
        region = _roi_slice(x, y, roi_size, ref.shape)
        origins.append((region[0].start, region[1].start))
    cp = _cupy()
    cp.cuda.Device(gpu_device_id).use()
    rois = _extract_rois(cp.asarray(ref), origins, roi_size, cp)
    centered = rois - rois.mean(axis=(1, 2), keepdims=True)
    a_norm = cp.sqrt(cp.sum(centered**2, axis=(1, 2)))
    filter_gpu = (None if frequency_filter is None else
                  cp.asarray(frequency_filter, dtype=cp.float64))
    window_gpu = (None if window is None else cp.asarray(window, dtype=cp.float64))
    transformed = centered*window_gpu[None] if window_gpu is not None else rois
    a_fft = cp.fft.fftn(transformed, axes=(-2, -1))
    if filter_gpu is not None:
        a_fft *= filter_gpu[None]
    return GPUPreparedReference(ref.shape, roi_size, tuple(origins), a_fft,
                                centered, a_norm, filter_gpu, window_gpu,
                                gpu_device_id)


def measure_pattern_shifts_gpu(reference, scan, centers, roi_size, *,
                               frequency_filter=None, window=None,
                               scan_centers=None, prepared_reference=None,
                               gpu_device_id=0,
                               subpixel_method="parabolic_1d",
                               return_gpu=False):
    """Return CPU-compatible ROI measurements from batched CUDA FFTs."""
    ref = np.asarray(reference, dtype=np.float64)
    cp = _cupy()
    target = scan if isinstance(scan, cp.ndarray) else np.asarray(scan, dtype=np.float64)
    points = np.asarray(centers, dtype=np.float64)
    if ref.ndim != 2 or ref.shape != target.shape:
        raise ValueError("reference and scan must be 2-D images of equal shape")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("centers must have shape (n, 2)")
    target_points = points if scan_centers is None else np.asarray(scan_centers, dtype=np.float64)
    if target_points.shape != points.shape:
        raise ValueError("scan_centers must match centers")
    if roi_size < 5 or roi_size > min(ref.shape):
        raise ValueError("invalid ROI size")
    if subpixel_method not in ("parabolic_1d", "quadratic_2d"):
        raise ValueError("subpixel method must be 'parabolic_1d' or 'quadratic_2d'")
    if (frequency_filter is None) != (window is None):
        raise ValueError("frequency_filter and window must be provided together")
    if prepared_reference is not None and (
        prepared_reference.image_shape != ref.shape or
        prepared_reference.roi_size != roi_size or
        prepared_reference.gpu_device_id != gpu_device_id
    ):
        raise ValueError("prepared GPU reference does not match shape, ROI size, or device")
    origins = []
    target_origins = []
    for (x, y), (target_x, target_y) in zip(points, target_points):
        region = _roi_slice(x, y, roi_size, ref.shape)
        scan_region = _roi_slice(target_x, target_y, roi_size, target.shape)
        origins.append((region[0].start, region[1].start))
        target_origins.append((scan_region[0].start, scan_region[1].start))
    if not origins:
        return []
    cp.cuda.Device(gpu_device_id).use()
    prepared = (prepared_reference if prepared_reference is not None else
                prepare_reference_gpu(ref, points, roi_size, frequency_filter,
                                      window, gpu_device_id))
    if prepared_reference is not None:
        lookup = {origin: index for index, origin in enumerate(prepared.origins)}
        indices = cp.asarray([lookup[origin] for origin in origins], dtype=cp.int64)
        a_fft = prepared.a_fft[indices]
        a_centered = prepared.a_centered[indices]
        a_norm = prepared.a_norm[indices]
    else:
        a_fft, a_centered, a_norm = (prepared.a_fft, prepared.a_centered,
                                     prepared.a_norm)
    b = _extract_rois(cp.asarray(target), target_origins, roi_size, cp)
    b_centered = b - b.mean(axis=(1, 2), keepdims=True)
    transformed = (b_centered*prepared.window[None]
                   if prepared.window is not None else b)
    b_fft = cp.fft.fftn(transformed, axes=(-2, -1))
    if prepared.frequency_filter is not None:
        b_fft *= prepared.frequency_filter[None]
    correlation = cp.fft.fftshift(
        cp.fft.ifftn(a_fft*cp.conj(b_fft), axes=(-2, -1)).real,
        axes=(-2, -1))

    count, size, _ = correlation.shape
    index = cp.arange(count)
    peak_flat = cp.argmax(correlation.reshape(count, -1), axis=1)
    row, col = peak_flat//size, peak_flat%size
    safe_row, safe_col = cp.clip(row, 1, size-2), cp.clip(col, 1, size-2)
    center_values = correlation[index, safe_row, safe_col]
    x_before = correlation[index, safe_row, safe_col-1]
    x_after = correlation[index, safe_row, safe_col+1]
    y_before = correlation[index, safe_row-1, safe_col]
    y_after = correlation[index, safe_row+1, safe_col]
    x_den = x_before-2*center_values+x_after
    y_den = y_before-2*center_values+y_after
    x_offset = cp.clip(cp.where(cp.abs(x_den) >= np.finfo(float).eps,
                                0.5*(x_before-x_after)/x_den, 0), -1, 1)
    y_offset = cp.clip(cp.where(cp.abs(y_den) >= np.finfo(float).eps,
                                0.5*(y_before-y_after)/y_den, 0), -1, 1)
    if subpixel_method == "quadratic_2d":
        z00 = correlation[index, safe_row-1, safe_col-1]
        z01 = correlation[index, safe_row-1, safe_col]
        z02 = correlation[index, safe_row-1, safe_col+1]
        z10 = x_before
        z11 = center_values
        z12 = x_after
        z20 = correlation[index, safe_row+1, safe_col-1]
        z21 = correlation[index, safe_row+1, safe_col]
        z22 = correlation[index, safe_row+1, safe_col+1]
        a = (z00+z10+z20+z02+z12+z22-2*(z01+z11+z21))/6
        b = (z00+z01+z02+z20+z21+z22-2*(z10+z11+z12))/6
        c = (z00-z02-z20+z22)/4
        d = (z02+z12+z22-z00-z10-z20)/6
        e = (z20+z21+z22-z00-z01-z02)/6
        determinant = 4*a*b-c*c
        valid_peak = (a < 0) & (b < 0) & (determinant > np.finfo(float).eps)
        safe_determinant = cp.where(valid_peak, determinant, 1)
        x_offset = cp.where(valid_peak,
                            cp.clip((c*e-2*b*d)/safe_determinant, -1, 1), x_offset)
        y_offset = cp.where(valid_peak,
                            cp.clip((c*d-2*a*e)/safe_determinant, -1, 1), y_offset)
    interior = ((row >= 2) & (col >= 2) & (row < size-2) & (col < size-2))
    x_offset = cp.where(interior, x_offset, 0)
    y_offset = cp.where(interior, y_offset, 0)
    origin_difference = np.asarray(target_origins)-np.asarray(origins)
    dx = size//2-col-x_offset+cp.asarray(origin_difference[:, 1])
    dy = size//2-row-y_offset+cp.asarray(origin_difference[:, 0])

    b_norm = cp.sqrt(cp.sum(b_centered**2, axis=(1, 2)))
    denominator = a_norm*b_norm
    coefficient = cp.where(denominator > 0,
                           cp.sum(a_centered*b_centered, axis=(1, 2))/denominator
                           *(size*size-1)/(size*size), 0)
    spread = correlation.std(axis=(1, 2), ddof=1)
    confidence = cp.where(spread > 0,
                           (correlation.max(axis=(1, 2))
                            -correlation.mean(axis=(1, 2)))/spread, 0)
    measured_gpu = cp.stack((dx, dy, coefficient, confidence), axis=1)
    if return_gpu:
        return measured_gpu
    measured = cp.asnumpy(measured_gpu)
    return [CorrelationResult(float(x), float(y), *map(float, values))
            for (x, y), values in zip(points, measured)]

def measure_pattern_shifts_gpu_batch(
    reference, scans, centers, roi_size, *, frequency_filter=None, window=None,
    scan_centers=None, prepared_reference=None, gpu_device_id=0,
    subpixel_method="parabolic_1d",
):
    """Measure many scan patterns with one CPU-to-GPU upload.

    ``scans`` has shape ``(batch, height, width)``.  The reference FFT cache
    stays resident on the GPU and each scan is processed there before only
    its compact ROI measurements are copied back.  ``scan_centers`` may be a
    common ``(roi, 2)`` array or one ``(batch, roi, 2)`` array.
    """
    cp = _cupy()
    values = np.asarray(scans)
    if values.ndim != 3:
        raise ValueError("scans must have shape (batch, height, width)")
    if values.shape[0] == 0:
        return []
    points = np.asarray(centers, dtype=np.float64)
    if scan_centers is None:
        target_points = points
    else:
        target_points = np.asarray(scan_centers, dtype=np.float64)
        if target_points.ndim == 2 and target_points.shape != points.shape:
            raise ValueError("scan_centers must match centers")
        if target_points.ndim == 3 and target_points.shape != (values.shape[0], *points.shape):
            raise ValueError("batched scan_centers must have shape (batch, roi, 2)")
    # Upload the complete chunk once.  Individual calls below operate on GPU
    # views, so no additional host transfer is made for each pattern.
    scans_gpu = cp.asarray(values, dtype=cp.float64)
    measured_gpu = []
    for index in range(values.shape[0]):
        current_centers = target_points if target_points.ndim == 2 else target_points[index]
        measured_gpu.append(measure_pattern_shifts_gpu(
            reference, scans_gpu[index], points, roi_size,
            frequency_filter=frequency_filter, window=window,
            scan_centers=current_centers, prepared_reference=prepared_reference,
            gpu_device_id=gpu_device_id, subpixel_method=subpixel_method,
            return_gpu=True))
    measured = cp.asnumpy(cp.stack(measured_gpu, axis=0))
    return [[CorrelationResult(float(x), float(y), *map(float, row))
             for (x, y), row in zip(points, measured[index])]
            for index in range(values.shape[0])]
