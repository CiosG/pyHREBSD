"""ROI cross-correlation based on the local HR-EBSD method of WMD2006.

Implementation references include OpenXY's GetROIs, custfftxc, and
subpixshift. Full citations and software provenance are recorded in
``docs/references.md``.

Coordinates are (x, y) pixel centers with a zero-based origin. Positive
shifts mean the scan pattern moved right or down relative to the reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class CorrelationResult:
    x: float
    y: float
    dx: float
    dy: float
    coefficient: float
    peak_confidence: float


@dataclass(frozen=True)
class PreparedReference:
    image_shape: tuple[int, int]
    roi_size: int
    rois: dict[tuple[int, int], tuple[NDArray[np.complex128], NDArray[np.float64], float]]


def prepare_reference(reference: ArrayLike, centers: ArrayLike, roi_size: int,
                      frequency_filter: ArrayLike | None = None,
                      window: ArrayLike | None = None) -> PreparedReference:
    """Cache reference ROI FFTs and intensity statistics for repeated scans."""
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim != 2:
        raise ValueError("reference must be a 2-D image")
    if (frequency_filter is None) != (window is None):
        raise ValueError("frequency_filter and window must be provided together")
    if frequency_filter is not None:
        frequency_filter = np.asarray(frequency_filter, dtype=np.float64)
        window = np.asarray(window, dtype=np.float64)
        if frequency_filter.shape != (roi_size, roi_size) or window.shape != (roi_size, roi_size):
            raise ValueError("filter and window must match the ROI size")
    rois = {}
    for x, y in np.asarray(centers, dtype=np.float64):
        region = _roi_slice(x, y, roi_size, ref.shape)
        a = ref[region]
        centered = a - a.mean()
        transformed = centered * window if window is not None else a
        a_fft = np.fft.fftn(transformed)
        if frequency_filter is not None:
            a_fft *= frequency_filter
        rois[(region[0].start, region[1].start)] = (
            a_fft, centered, float(np.linalg.norm(centered)))
    return PreparedReference(ref.shape, roi_size, rois)


def _matlab_round(value: ArrayLike) -> NDArray[np.int64]:
    """MATLAB's round for the nonnegative ROI coordinates used here."""
    return np.floor(np.asarray(value) + 0.5).astype(np.int64)


def grid_rois(image_shape: tuple[int, int], roi_size: int, count: int) -> NDArray[np.float64]:
    """Return the square grid used by OpenXY's GetROIs('Grid').

    ``count`` follows MATLAB's grid convention: the actual number of ROIs is
    ``round(sqrt(count)) ** 2``. Centers are returned as zero-based (x, y).
    """
    height, width = image_shape
    if height != width:
        raise ValueError("OpenXY grid ROIs require a square pattern")
    if roi_size < 3 or roi_size > width or count < 1:
        raise ValueError("invalid ROI size or count")
    edge_count = int(_matlab_round(np.sqrt(count)))
    edge_spacing = int(_matlab_round(roi_size / 2 + 0.1 * width))
    centers = np.linspace(edge_spacing, width - edge_spacing, edge_count) - 1
    xx, yy = np.meshgrid(centers, centers)
    points = np.column_stack((xx.ravel(order="F"), yy.ravel(order="F")))
    for x, y in points:
        _roi_slice(x, y, roi_size, image_shape)
    return points


def annular_rois(image_shape: tuple[int, int], roi_size: int,
                 count: int) -> NDArray[np.float64]:
    """Return one center ROI plus ``count - 1`` equally spaced on a ring."""
    height, width = image_shape
    if height != width:
        raise ValueError("annular ROIs require a square pattern")
    if roi_size < 3 or roi_size > width or count < 2:
        raise ValueError("invalid ROI size or count")
    center = width / 2 - 1
    radius = np.floor((width - roi_size) / 3)
    angles = np.arange(count - 1) * (2 * np.pi / (count - 1))
    points = np.empty((count, 2), dtype=np.float64)
    points[0] = (center, center)
    points[1:, 0] = center + radius * np.cos(angles)
    points[1:, 1] = center + radius * np.sin(angles)
    for x, y in points:
        _roi_slice(x, y, roi_size, image_shape)
    return points


def roi_size_from_percent(image_shape: tuple[int, int], percent: float) -> int:
    """Convert ROI width from percent of the shorter pattern dimension."""
    if len(image_shape) != 2 or not np.isfinite(percent) or percent <= 0:
        raise ValueError("ROI percentage must be positive and finite")
    size = int(_matlab_round(min(image_shape) * percent / 100))
    if size < 5 or size > min(image_shape):
        raise ValueError("ROI percentage produces an invalid ROI size")
    return size


def _roi_slice(x: float, y: float, size: int, shape: tuple[int, int]) -> tuple[slice, slice]:
    # MATLAB: round(center-size/2):round(center-size/2)+size-1, then 1-based indexing.
    left = int(_matlab_round(x + 1 - size / 2)) - 1
    top = int(_matlab_round(y + 1 - size / 2)) - 1
    if left < 0 or top < 0 or left + size > shape[1] or top + size > shape[0]:
        raise ValueError(f"ROI centered at ({x}, {y}) extends outside the pattern")
    return slice(top, top + size), slice(left, left + size)


def subpixel_shift(correlation: ArrayLike, method: str = "parabolic_1d") -> tuple[float, float]:
    """Locate an FFT peak with independent 1-D or coupled 2-D parabolas."""
    if method not in ("parabolic_1d", "quadratic_2d"):
        raise ValueError("subpixel method must be 'parabolic_1d' or 'quadratic_2d'")
    image = np.asarray(correlation, dtype=np.float64)
    if image.ndim != 2 or image.shape[0] != image.shape[1] or image.shape[0] < 5:
        raise ValueError("correlation must be a square array of at least 5 pixels")
    row, col = np.unravel_index(np.argmax(image), image.shape)
    center = image.shape[0] // 2
    if row < 2 or col < 2 or row >= image.shape[0] - 2 or col >= image.shape[1] - 2:
        return float(center - col), float(center - row)

    if method == "quadratic_2d":
        patch = image[row - 1:row + 2, col - 1:col + 2]
        # Least-squares fit of a*x*x + b*y*y + c*x*y + d*x + e*y + f
        # to the nine samples at x,y=-1,0,1.
        a = (patch[:, 0].sum() + patch[:, 2].sum() - 2*patch[:, 1].sum()) / 6
        b = (patch[0, :].sum() + patch[2, :].sum() - 2*patch[1, :].sum()) / 6
        c = (patch[0, 0] - patch[0, 2] - patch[2, 0] + patch[2, 2]) / 4
        d = (patch[:, 2].sum() - patch[:, 0].sum()) / 6
        e = (patch[2, :].sum() - patch[0, :].sum()) / 6
        determinant = 4*a*b - c*c
        if a < 0 and b < 0 and determinant > np.finfo(float).eps:
            x_offset = np.clip((c*e - 2*b*d) / determinant, -1, 1)
            y_offset = np.clip((c*d - 2*a*e) / determinant, -1, 1)
            return float(center - col - x_offset), float(center - row - y_offset)
        # A flat or saddle-shaped peak cannot be refined reliably in 2-D.

    def offset(before: float, peak: float, after: float) -> float:
        denominator = before - 2 * peak + after
        if abs(denominator) < np.finfo(float).eps:
            return 0.0
        return float(np.clip(0.5 * (before - after) / denominator, -1, 1))

    x_offset = offset(image[row, col - 1], image[row, col], image[row, col + 1])
    y_offset = offset(image[row - 1, col], image[row, col], image[row + 1, col])
    return center - col - x_offset, center - row - y_offset


def measure_pattern_shifts(
    reference: ArrayLike,
    scan: ArrayLike,
    centers: ArrayLike,
    roi_size: int,
    *,
    frequency_filter: ArrayLike | None = None,
    window: ArrayLike | None = None,
    scan_centers: ArrayLike | None = None,
    prepared_reference: PreparedReference | None = None,
    subpixel_method: str = "parabolic_1d",
) -> list[CorrelationResult]:
    """Measure local shifts from a reference EBSD image to a scan image.

    The optional frequency filter and spatial window reproduce the two
    inputs accepted by MATLAB's ``custfftxc``. Without a filter, raw FFT
    correlation is used, as in the MATLAB implementation.
    """
    ref = np.asarray(reference, dtype=np.float64)
    target = np.asarray(scan, dtype=np.float64)
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
    if prepared_reference is not None and (
        prepared_reference.image_shape != ref.shape or prepared_reference.roi_size != roi_size
    ):
        raise ValueError("prepared reference does not match image shape or ROI size")
    if (frequency_filter is None) != (window is None):
        raise ValueError("frequency_filter and window must be provided together")
    if frequency_filter is not None:
        frequency_filter = np.asarray(frequency_filter, dtype=np.float64)
        window = np.asarray(window, dtype=np.float64)
        if frequency_filter.shape != (roi_size, roi_size) or window.shape != (roi_size, roi_size):
            raise ValueError("filter and window must match the ROI size")

    results = []
    for (x, y), (target_x, target_y) in zip(points, target_points):
        region = _roi_slice(x, y, roi_size, ref.shape)
        scan_region = _roi_slice(target_x, target_y, roi_size, target.shape)
        a = ref[region]
        b = target[scan_region]
        if prepared_reference is not None:
            a_fft, a_centered, a_norm = prepared_reference.rois[
                (region[0].start, region[1].start)]
        else:
            a_centered = a - a.mean()
            a_norm = np.linalg.norm(a_centered)
            a_fft = (np.fft.fftn(a_centered * window) * frequency_filter
                     if frequency_filter is not None else np.fft.fftn(a))
        if frequency_filter is not None:
            b_fft = np.fft.fftn((b - b.mean()) * window) * frequency_filter
        else:
            b_fft = np.fft.fftn(b)
        correlation = np.fft.fftshift(np.fft.ifftn(a_fft * np.conj(b_fft)).real)
        dx, dy = subpixel_shift(correlation, subpixel_method)
        dx += scan_region[1].start - region[1].start
        dy += scan_region[0].start - region[0].start
        b_centered = b - b.mean()
        denominator = a_norm * np.linalg.norm(b_centered)
        # CalcCrossCorrelationCoef.m uses MATLAB std (sample denominator N-1)
        # and then divides the summed products by N.
        coefficient = (
            float(np.sum(a_centered * b_centered) / denominator * (a.size - 1) / a.size)
            if denominator else 0.0
        )
        spread = correlation.std(ddof=1)
        confidence = float((correlation.max() - correlation.mean()) / spread) if spread else 0.0
        results.append(CorrelationResult(float(x), float(y), dx, dy, coefficient, confidence))
    return results
