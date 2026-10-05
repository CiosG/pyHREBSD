"""Elastic lattice rotations from the fitted deformation gradient."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation


def sample_rotation_vector_mrad(
    deformation_crystal: ArrayLike,
    sample_to_crystal: ArrayLike,
) -> NDArray[np.float64]:
    """Return the right-handed polar rotation vector in sample axes, mrad.

    The fitted F maps the reference pattern to the scan pattern. The rotation
    is the R in the right polar decomposition F = R U, after changing from
    crystal coordinates into sample coordinates. An improper F has no proper
    polar rotation and returns NaNs.
    """
    f = np.asarray(deformation_crystal, dtype=np.float64)
    g = np.asarray(sample_to_crystal, dtype=np.float64)
    if f.shape != (3, 3) or g.shape != (3, 3):
        raise ValueError("deformation and orientation must be 3x3 matrices")
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(g)):
        return np.full(3, np.nan)
    sample_f = g.T @ f @ g
    if np.linalg.det(sample_f) <= 0:
        return np.full(3, np.nan)
    left, _, right = np.linalg.svd(sample_f)
    polar_rotation = left @ right
    return Rotation.from_matrix(polar_rotation).as_rotvec() * 1000.0
