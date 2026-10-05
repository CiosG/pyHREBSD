"""Projective back-rotation for two-pass ROI correlation after BW2012.

See ``docs/references.md`` for the scientific citation and implementation
provenance.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def remap_pattern(scan, reference_pc, scan_pc, rotation_phosphor,
                  device="cpu", gpu_device_id=0):
    """Sample a scan pattern at the rays of a back-rotated reference detector.

    ``rotation_phosphor`` maps reference rays to scan rays. The output is on
    the reference detector grid, including the change of projection centre.
    Return both the resampled image and pixels with valid cubic support.
    """
    image = np.asarray(scan)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("remapping requires a square scan pattern")
    if device not in ("cpu", "gpu"):
        raise ValueError("remapping device must be 'cpu' or 'gpu'")
    if reference_pc[2] <= 0 or scan_pc[2] <= 0:
        raise ValueError("pattern centre depth must be positive")
    rotation = np.asarray(rotation_phosphor, dtype=np.float64)
    if (rotation.shape != (3, 3) or
            not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or
            np.linalg.det(rotation) < 0):
        raise ValueError("rotation_phosphor must be a proper 3-D rotation")

    if device == "gpu":
        try:
            import cupy as xp
            from cupyx.scipy.ndimage import map_coordinates as interpolate
        except ImportError as exc:
            raise RuntimeError("GPU remapping requires CuPy and cupyx") from exc
        xp.cuda.Device(gpu_device_id).use()
    else:
        xp = np
        interpolate = map_coordinates

    side = image.shape[0]
    y, x = xp.indices(image.shape, dtype=xp.float64)
    ray_x = reference_pc[0] - (x + 1) / side
    ray_y = 1 - reference_pc[1] - (y + 1) / side
    ray_z = -reference_pc[2]
    rotated_x = rotation[0, 0]*ray_x + rotation[0, 1]*ray_y + rotation[0, 2]*ray_z
    rotated_y = rotation[1, 0]*ray_x + rotation[1, 1]*ray_y + rotation[1, 2]*ray_z
    rotated_z = rotation[2, 0]*ray_x + rotation[2, 1]*ray_y + rotation[2, 2]*ray_z
    scale = -scan_pc[2] / rotated_z
    source_x = (scan_pc[0] - rotated_x*scale)*side - 1
    source_y = (1 - scan_pc[1] - rotated_y*scale)*side - 1
    # Cubic spline prefiltering also has a short boundary transient; six
    # pixels exclude it from the ROI fit. ROIs containing invalid pixels are
    # discarded by the analysis caller.
    valid = (xp.isfinite(source_x) & xp.isfinite(source_y) &
             (source_x >= 6) & (source_x < side - 7) &
             (source_y >= 6) & (source_y < side - 7))
    safe_x = xp.where(valid, source_x, 0)
    safe_y = xp.where(valid, source_y, 0)
    remapped = interpolate(xp.asarray(image, dtype=xp.float64),
                           xp.stack((safe_y, safe_x)), order=3,
                           mode="constant", cval=0)
    if device == "gpu":
        return xp.asnumpy(remapped), xp.asnumpy(valid)
    return remapped, valid


def valid_roi_centers(centers, roi_size, valid_pixels):
    """Keep ROIs whose entire resampled area lies inside the scan pattern."""
    from .correlation import _roi_slice

    invalid = (~np.asarray(valid_pixels, dtype=bool)).astype(np.int64)
    summed = np.pad(invalid, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    keep = []
    for x, y in centers:
        rows, cols = _roi_slice(x, y, roi_size, invalid.shape)
        top, bottom = rows.start, rows.stop
        left, right = cols.start, cols.stop
        count = (summed[bottom, right] - summed[top, right] -
                 summed[bottom, left] + summed[top, left])
        keep.append(count == 0)
    return np.asarray(keep, dtype=bool)
