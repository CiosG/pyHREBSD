"""Surface HR-EBSD GND estimates from partial Nye curvature measurements.

The six observables are (alpha12, alpha13, alpha21, alpha23, alpha33,
alpha11-alpha22) in sample axes.  They are inferred from in-plane derivatives
of the elastic distortion.  Unmeasured depth derivatives of elastic strain are
set to zero; this is an explicit approximation, not a full 3-D Nye tensor.

Wilkinson & Randman, Philosophical Magazine 90 (2010), Eqs. 6-9 and 11.
Arsenlis & Parks, Acta Materialia 47 (1999), Nye tensor decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog


OBSERVABLES = ("alpha12", "alpha13", "alpha21", "alpha23", "alpha33",
               "alpha11_minus_alpha22")


@dataclass(frozen=True)
class DislocationType:
    name: str
    kind: str
    burgers_crystal_m: NDArray[np.float64]
    line_crystal: NDArray[np.float64]
    plane_crystal: NDArray[np.float64] | None


def fcc_perfect_dislocations(lattice_parameter_m: float
                             ) -> tuple[DislocationType, ...]:
    """12 edge and 6 unique screw a/2<110>{111} perfect dislocations.

    A screw direction lies in two {111} planes, so a screw density cannot be
    assigned uniquely to one of those planes from curvature alone.
    """
    if not np.isfinite(lattice_parameter_m) or lattice_parameter_m <= 0:
        raise ValueError("lattice_parameter_m must be positive")
    planes = ((1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1))
    directions = set()
    for zero in range(3):
        others = [i for i in range(3) if i != zero]
        for sign in (-1, 1):
            v = np.zeros(3, dtype=int)
            v[others[0]], v[others[1]] = 1, sign
            directions.add(tuple(v))
    directions = sorted(directions)
    result: list[DislocationType] = []
    for plane in planes:
        n = np.asarray(plane, dtype=float)
        for direction in directions:
            d = np.asarray(direction, dtype=float)
            if np.dot(n, d) != 0:
                continue
            line = np.cross(n, d)
            line /= np.linalg.norm(line)
            result.append(DislocationType(
                f"({plane[0]}{plane[1]}{plane[2]})[{direction[0]}{direction[1]}{direction[2]}] edge",
                "edge", lattice_parameter_m * d / 2, line, n / np.linalg.norm(n)))
    for direction in directions:
        d = np.asarray(direction, dtype=float)
        result.append(DislocationType(
            f"<{direction[0]}{direction[1]}{direction[2]}> screw",
            "screw", lattice_parameter_m * d / 2, d / np.linalg.norm(d), None))
    if len(result) != 18:
        raise AssertionError("FCC basis must have 12 edge and 6 screw types")
    return tuple(result)


def silicon_perfect_dislocations(lattice_parameter_m: float = 0.5431e-9
                                 ) -> tuple[DislocationType, ...]:
    """Silicon diamond-cubic perfect-dislocation subset with FCC slip geometry."""
    return fcc_perfect_dislocations(lattice_parameter_m)


def observable_nye(alpha: ArrayLike) -> NDArray[np.float64]:
    """Extract the five partial Nye components and one diagonal difference."""
    a = np.asarray(alpha, dtype=float)
    if a.shape[-2:] != (3, 3):
        raise ValueError("alpha must have trailing 3x3 shape")
    return np.stack((a[..., 0, 1], a[..., 0, 2], a[..., 1, 0],
                     a[..., 1, 2], a[..., 2, 2], a[..., 0, 0] - a[..., 1, 1]),
                    axis=-1)


def partial_nye_from_maps(rotation_sample_mrad: ArrayLike, x_step_m: float,
                          y_step_m: float, strain_sample: ArrayLike | None = None,
                          gradient_span: int = 1
                          ) -> NDArray[np.float64]:
    """Return six surface observables in 1/m, using forward x/y differences.

    Input rotation has shape (rows, columns, 3); optional symmetric strain has
    shape (rows, columns, 3, 3).  Last row/column and any stencil touching a NaN
    remain NaN.  Positive rotation follows the right-handed rotvec convention.
    gradient_span=N uses points N map steps apart without smoothing the fields.
    """
    w = np.asarray(rotation_sample_mrad, dtype=float) / 1000.0
    if w.ndim != 3 or w.shape[-1] != 3 or min(w.shape[:2]) < 2:
        raise ValueError("rotation map must have shape (rows>=2, columns>=2, 3)")
    if not all(np.isfinite(v) and v > 0 for v in (x_step_m, y_step_m)):
        raise ValueError("map steps must be positive metres")
    if not isinstance(gradient_span, int) or gradient_span < 1 or gradient_span >= min(w.shape[:2]):
        raise ValueError("gradient_span must fit within both map dimensions")
    beta = np.zeros(w.shape[:2] + (3, 3), dtype=float)
    beta[..., 0, 1], beta[..., 0, 2] = -w[..., 2], w[..., 1]
    beta[..., 1, 0], beta[..., 1, 2] = w[..., 2], -w[..., 0]
    beta[..., 2, 0], beta[..., 2, 1] = -w[..., 1], w[..., 0]
    if strain_sample is not None:
        strain = np.asarray(strain_sample, dtype=float)
        if strain.shape != beta.shape or not np.allclose(
                strain, np.swapaxes(strain, -1, -2), atol=1e-8, equal_nan=True):
            raise ValueError("strain map must be symmetric and match rotation map")
        beta += strain
    beta[~np.all(np.isfinite(w), axis=-1)] = np.nan
    s = gradient_span
    dx = (beta[:-s, s:] - beta[:-s, :-s]) / (s * x_step_m)
    dy = (beta[s:, :-s] - beta[:-s, :-s]) / (s * y_step_m)
    core = np.stack((
        -dx[..., 0, 2],
        dx[..., 0, 1] - dy[..., 0, 0],
        dy[..., 1, 2],
        dx[..., 1, 1] - dy[..., 1, 0],
        dx[..., 2, 1] - dy[..., 2, 0],
        dy[..., 0, 2] + dx[..., 1, 2],
    ), axis=-1)
    out = np.full(w.shape[:2] + (6,), np.nan)
    out[:-s, :-s] = core
    return out


def dislocation_matrix(types: tuple[DislocationType, ...], sample_to_crystal: ArrayLike
                       ) -> NDArray[np.float64]:
    """Nye observable matrix A, with columns in metres per unit density."""
    g = np.asarray(sample_to_crystal, dtype=float)
    if g.shape != (3, 3) or not np.allclose(g @ g.T, np.eye(3), atol=1e-5):
        raise ValueError("sample_to_crystal must be an orthogonal 3x3 matrix")
    columns = [observable_nye(np.outer(g.T @ kind.burgers_crystal_m,
                                       g.T @ kind.line_crystal)) for kind in types]
    matrix = np.column_stack(columns)
    if np.linalg.matrix_rank(matrix) < 6:
        raise ValueError("dislocation types cannot span all six observables")
    return matrix


def solve_gnd_l1(observables_per_m: ArrayLike, matrix_m: ArrayLike,
                 types: tuple[DislocationType, ...], poisson_ratio: float = 0.28
                 ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Minimum-line-energy signed densities (m^-2) and residuals (1/m).

    The L1 objective weights pure edge lines by 1/(1-nu) and pure screw
    lines by 1.  LP densities are scaled to units of 1e12 m^-2.
    """
    k = np.asarray(observables_per_m, dtype=float)
    a = np.asarray(matrix_m, dtype=float)
    if k.shape[-1] != 6 or a.shape != (6, len(types)):
        raise ValueError("incompatible observables or dislocation matrix")
    if not 0 <= poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must be between 0 and 0.5")
    weights = np.array([1 / (1 - poisson_ratio) if kind.kind == "edge" else 1
                        for kind in types])
    objective = np.r_[weights, weights]
    equality = np.column_stack((a, -a)) * 1e12
    flat = k.reshape(-1, 6)
    density = np.full((len(flat), len(types)), np.nan)
    residual = np.full((len(flat), 6), np.nan)
    for index, values in enumerate(flat):
        if not np.all(np.isfinite(values)):
            continue
        if np.max(np.abs(values)) < 1e-10:
            density[index] = 0
            residual[index] = 0
            continue
        fitted = linprog(objective, A_eq=equality, b_eq=values,
                         bounds=(0, None), method="highs")
        if not fitted.success:
            continue
        density[index] = (fitted.x[:len(types)] - fitted.x[len(types):]) * 1e12
        residual[index] = a @ density[index] - values
    return density.reshape(k.shape[:-1] + (len(types),)), residual.reshape(k.shape)
