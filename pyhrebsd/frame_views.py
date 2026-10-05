"""Change the displayed in-plane sample basis without changing the map grid."""

from __future__ import annotations

import numpy as np


def in_plane_rotation(degrees: float) -> np.ndarray:
    """Proper rotation of sample components about axis 3."""
    if not np.isfinite(degrees):
        raise ValueError("sample axis rotation must be finite")
    radians = np.deg2rad(degrees)
    c, s = np.cos(radians), np.sin(radians)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotate_tensor_maps(maps: dict[str, np.ndarray], prefix: str,
                       degrees: float) -> dict[str, np.ndarray]:
    """Rotate six symmetric sample-frame component maps by R T R-transpose."""
    if degrees == 0:
        return maps
    tensor = np.empty((3, 3) + maps[f"{prefix}_11"].shape, dtype=float)
    for i in range(3):
        for j in range(i, 3):
            values = maps[f"{prefix}_{i+1}{j+1}"]
            tensor[i, j] = values
            tensor[j, i] = values
    rotation = in_plane_rotation(degrees)
    transformed = np.einsum("ia,jb,abhw->ijhw", rotation, rotation, tensor,
                            optimize=True)
    result = dict(maps)
    for i in range(3):
        for j in range(i, 3):
            result[f"{prefix}_{i+1}{j+1}"] = transformed[i, j]
    return result


def rotate_vector_maps(maps: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate three sample-frame vector component maps by R v."""
    values = np.asarray(maps, dtype=float)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError("vector maps must have shape (3, height, width)")
    return np.einsum("ij,jhw->ihw", in_plane_rotation(degrees), values,
                     optimize=True)
