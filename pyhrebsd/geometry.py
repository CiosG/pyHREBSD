"""Pattern geometry from euler2gmat.m and Theoretical_Pixel_Shift.m."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def euler_to_matrix(phi1: float, phi: float, phi2: float) -> NDArray[np.float64]:
    """Bunge Euler angles in radians to OpenXY's sample-to-crystal matrix."""
    cp1, sp1 = np.cos(phi1), np.sin(phi1)
    cp2, sp2 = np.cos(phi2), np.sin(phi2)
    cphi, sphi = np.cos(phi), np.sin(phi)
    return np.array([
        [cp1 * cp2 - sp1 * sp2 * cphi, sp1 * cp2 + cp1 * sp2 * cphi, sp2 * sphi],
        [-cp1 * sp2 - sp1 * cp2 * cphi, -sp1 * sp2 + cp1 * cp2 * cphi, cp2 * sphi],
        [sp1 * sphi, -cp1 * sphi, cphi],
    ], dtype=np.float64)


def phosphor_to_sample_from_oxford(
    sample_tilt: float, detector_euler: ArrayLike,
) -> NDArray[np.float64]:
    """Full Oxford detector orientation in OpenXY's phosphor-to-sample frame.

    Port of ``frameTransforms.phosphorToSample.m`` when ``camphi1`` exists.
    Angles are radians; detector_euler contains all three Bunge angles.
    """
    angles = np.asarray(detector_euler, dtype=np.float64)
    if angles.shape != (3,) or not np.all(np.isfinite(angles)):
        raise ValueError("detector_euler must contain three finite radians")
    c, s = np.cos(sample_tilt), np.sin(sample_tilt)
    qio = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    qmi = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    return qio @ qmi @ euler_to_matrix(*angles).T @ np.diag([-1, 1, -1])


def theoretical_pixel_shift(
    orientation: ArrayLike,
    pattern_center: tuple[float, float, float],
    centers: ArrayLike,
    deformation: ArrayLike,
    pixel_size: int,
    alpha: float,
    phosphor_to_sample: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Project a crystal-frame deformation into ROI shifts in image pixels.

    ``orientation`` is either a 3x3 matrix or Bunge Euler angles in radians.
    Pattern center is normalized (xstar, ystar, zstar); ``alpha`` is
    pi/2 - sample tilt + camera elevation, all in radians.
    """
    g = np.asarray(orientation, dtype=np.float64)
    if g.shape == (3,):
        g = euler_to_matrix(*g)
    f = np.asarray(deformation, dtype=np.float64)
    points = np.asarray(centers, dtype=np.float64)
    if g.shape != (3, 3) or f.shape != (3, 3):
        raise ValueError("orientation and deformation must be 3x3 matrices")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("centers must have shape (n, 2)")
    if pixel_size <= 0 or pattern_center[2] <= 0:
        raise ValueError("pixel size and zstar must be positive")
    xstar, ystar, zstar = pattern_center
    cosine, sine = np.cos(alpha), np.sin(alpha)
    qps = (np.array([[0, -cosine, -sine], [-1, 0, 0], [0, sine, -cosine]])
           if phosphor_to_sample is None else np.asarray(phosphor_to_sample, dtype=np.float64))
    if qps.shape != (3, 3) or not np.allclose(qps @ qps.T, np.eye(3), atol=1e-5):
        raise ValueError("phosphor_to_sample must be a rotation matrix")
    qvp = np.diag([-1.0, -1.0, 1.0])
    dvp = np.array([xstar * pixel_size, (1 - ystar) * pixel_size, 0.0])
    # MATLAB ROI centers are 1-based; the public Python API is zero-based.
    matlab_points = points + 1
    screen = (qvp @ np.column_stack((matlab_points, np.zeros(len(points)))).T).T + dvp
    sample = (qps @ (screen + [0, 0, -zstar * pixel_size]).T).T
    crystal = (g @ sample.T).T
    deformed_sample = (g.T @ f @ crystal.T).T
    normal = qps @ np.array([0.0, 0.0, -1.0])
    plane_point = qps @ np.array([0.0, 0.0, -zstar * pixel_size])
    denominator = deformed_sample @ normal
    if np.any(np.isclose(denominator, 0)):
        raise ValueError("deformed ray is parallel to the detector")
    intersection = deformed_sample * ((normal @ plane_point) / denominator)[:, None]
    deformed_screen = (qps.T @ intersection.T).T + [0, 0, zstar * pixel_size]
    deformed_view = (qvp @ deformed_screen.T).T + dvp
    return deformed_view[:, :2] - matlab_points
