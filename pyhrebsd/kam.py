"""Local misorientation of HR-EBSD lattice rotations on a square scan grid."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def hr_kam(rotation_vectors_mrad: np.ndarray, valid: np.ndarray,
           grain_ids: np.ndarray, threshold_degrees: float | None = 5.0
           ) -> tuple[np.ndarray, np.ndarray]:
    """Mean disorientation to four direct neighbours, in degrees.

    Rotation vectors are the sample-frame polar rotations measured by HR-EBSD
    relative to one reference pattern. The angle of ``R_i^{-1} R_j`` is used
    for each pair; crystal symmetry is not reapplied to these small, relative
    rotations. Pairs crossing a grain boundary or exceeding the optional
    misorientation threshold are excluded. Points without eligible neighbours
    have NaN KAM and zero count.
    """
    vectors = np.asarray(rotation_vectors_mrad, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    grains = np.asarray(grain_ids, dtype=np.int32)
    if vectors.ndim != 3 or vectors.shape[-1] != 3:
        raise ValueError("rotation_vectors_mrad must have shape (rows, columns, 3)")
    shape = vectors.shape[:2]
    if mask.shape != shape or grains.shape != shape:
        raise ValueError("valid and grain_ids must match the scan grid")
    if threshold_degrees is not None and not 0 < threshold_degrees <= 180:
        raise ValueError("threshold_degrees must be in (0, 180] or None")

    height, width = shape
    flat_vectors = vectors.reshape(-1, 3)
    flat_valid = mask.ravel() & (grains.ravel() > 0) & np.all(np.isfinite(flat_vectors), axis=1)
    safe_vectors = np.where(flat_valid[:, None], flat_vectors, 0.0)
    rotations = Rotation.from_rotvec(safe_vectors / 1000.0)
    grid = np.arange(height * width).reshape(shape)
    a = np.concatenate((grid[:, :-1].ravel(), grid[:-1, :].ravel()))
    b = np.concatenate((grid[:, 1:].ravel(), grid[1:, :].ravel()))
    same_grain = grains.ravel()[a] == grains.ravel()[b]
    eligible = flat_valid[a] & flat_valid[b] & same_grain
    a, b = a[eligible], b[eligible]
    angles = (rotations[a].inv() * rotations[b]).magnitude() * (180.0 / np.pi)
    if threshold_degrees is not None:
        keep = angles <= threshold_degrees
        a, b, angles = a[keep], b[keep], angles[keep]
    sums = np.zeros(height * width, dtype=np.float64)
    counts = np.zeros(height * width, dtype=np.int32)
    np.add.at(sums, a, angles)
    np.add.at(sums, b, angles)
    np.add.at(counts, a, 1)
    np.add.at(counts, b, 1)
    kam = np.full(height * width, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=kam, where=counts > 0)
    return kam.reshape(shape), counts.reshape(shape)
