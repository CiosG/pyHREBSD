"""Real-pattern deformation analysis based on OpenXY's CalcFShift.m."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import re
import unicodedata

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erf

from .correlation import (CorrelationResult, PreparedReference, _roi_slice, annular_rois, grid_rois,
                          measure_pattern_shifts, prepare_reference)
from .geometry import euler_to_matrix, theoretical_pixel_shift


@dataclass(frozen=True)
class Material:
    name: str
    lattice: str
    c11: float
    c12: float
    c44: float
    c13: float | None = None
    c33: float | None = None
    c66: float | None = None
    stiffness_voigt_gpa: NDArray[np.float64] | None = field(
        default=None, repr=False, compare=False)
    verification_status: str = "unspecified"
    reference: str = ""

    @classmethod
    def from_openxy_file(cls, path: str | Path) -> Material:
        """Read the elastic constants (GPa) from an OpenXY Materials/*.txt file."""
        values = {}
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                words = line.split()
                if len(words) >= 2:
                    values[words[0]] = words[1]
        required = ("C11", "C12", "C44", "lattice")
        if any(key not in values for key in required):
            raise ValueError(f"{path} is missing elastic constants or lattice")
        optional = ("C13", "C33", "C66")
        return cls(
            name=values.get("Material", Path(path).stem),
            lattice=values["lattice"].lower(),
            c11=float(values["C11"]), c12=float(values["C12"]), c44=float(values["C44"]),
            **{key.lower(): float(values[key]) if values.get(key) else None for key in optional},
        )

    @staticmethod
    def _normalise_name(value: str) -> str:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    @classmethod
    def from_hdf5(cls, path: str | Path, material: str) -> Material:
        """Read a material and its complete 6 x 6 Voigt stiffness matrix.

        ``material`` may be the database key (for example ``silicon``) or the
        human-readable material name stored in the group metadata.
        """
        requested = cls._normalise_name(material)
        with h5py.File(path, "r") as file:
            if "materials" not in file:
                raise ValueError(f"{path} does not contain /materials")
            parent = file["materials"]
            key = requested if requested in parent else None
            if key is None:
                matches = [candidate for candidate, group in parent.items()
                           if cls._normalise_name(str(group.attrs.get("name", ""))) == requested]
                if len(matches) != 1:
                    available = ", ".join(sorted(parent.keys()))
                    raise KeyError(f"material {material!r} not found in {path}; available: {available}")
                key = matches[0]
            group = parent[key]
            matrix = np.asarray(group["stiffness_voigt_gpa"], dtype=np.float64)
            if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"invalid stiffness matrix for {material!r}")
            if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-8):
                raise ValueError(f"stiffness matrix for {material!r} is not symmetric")
            name = str(group.attrs.get("name", key))
            lattice = str(group.attrs.get("crystal_system", "unknown")).lower()
            title = str(group.attrs.get("reference_title", ""))
            doi = str(group.attrs.get("reference_doi", ""))
            reference = title + (f"; doi:{doi}" if doi else "")
            return cls(
                name=name, lattice=lattice,
                c11=float(matrix[0, 0]), c12=float(matrix[0, 1]),
                c44=float(matrix[3, 3]), c13=float(matrix[0, 2]),
                c33=float(matrix[2, 2]), c66=float(matrix[5, 5]),
                stiffness_voigt_gpa=matrix,
                verification_status=str(group.attrs.get("verification_status", "unspecified")),
                reference=reference,
            )

    @classmethod
    def list_hdf5(cls, path: str | Path) -> tuple[str, ...]:
        """Return the stable database keys available under ``/materials``."""
        with h5py.File(path, "r") as file:
            return tuple(sorted(file["materials"].keys()))

    def stiffness(self) -> NDArray[np.float64]:
        """Fourth-order crystal-frame elastic tensor in GPa."""
        if self.stiffness_voigt_gpa is not None:
            voigt = np.asarray(self.stiffness_voigt_gpa, dtype=np.float64)
            pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
            tensor = np.zeros((3, 3, 3, 3), dtype=np.float64)
            for row, (i, j) in enumerate(pairs):
                for column, (k, ell) in enumerate(pairs):
                    value = voigt[row, column]
                    for a, b in ((i, j), (j, i)):
                        for c, d in ((k, ell), (ell, k)):
                            tensor[a, b, c, d] = value
            return tensor
        c = np.zeros((3, 3, 3, 3), dtype=np.float64)
        if self.lattice in ("cubic", "tetragonal"):
            normals = (self.c11, self.c11, self.c11)
            pairs = {(0, 1): self.c12, (0, 2): self.c12, (1, 2): self.c12}
            shears = {(0, 1): self.c44, (0, 2): self.c44, (1, 2): self.c44}
        elif self.lattice == "hexagonal":
            if self.c13 is None or self.c33 is None or self.c66 is None:
                raise ValueError("hexagonal material requires C13, C33, and C66")
            normals = (self.c11, self.c11, self.c33)
            pairs = {(0, 1): self.c12, (0, 2): self.c13, (1, 2): self.c13}
            shears = {(0, 1): self.c66, (0, 2): self.c44, (1, 2): self.c44}
        else:
            raise ValueError(f"unsupported lattice: {self.lattice}")
        for i in range(3):
            c[i, i, i, i] = normals[i]
        for (i, j), value in pairs.items():
            c[i, i, j, j] = c[j, j, i, i] = value
        for (i, j), value in shears.items():
            c[i, j, i, j] = c[j, i, i, j] = c[i, j, j, i] = c[j, i, j, i] = value
        return c


@dataclass(frozen=True)
class DeformationResult:
    deformation: NDArray[np.float64]
    strain: NDArray[np.float64]
    stress_gpa: NDArray[np.float64]
    orientation: NDArray[np.float64]
    rms_shift_error: float
    used_rois: int
    total_rois: int
    shifts: tuple[CorrelationResult, ...]
    keep_mask: NDArray[np.bool_]
    first_pass_shifts: tuple[CorrelationResult, ...] | None = None
    first_pass_rms_shift_error: float | None = None
    remapping_rotation_mrad: float | None = None
    residual_rotation_mrad: float | None = None


@dataclass(frozen=True)
class AnalysisPlan:
    centers: NDArray[np.float64]
    frequency_filter: NDArray[np.float64] | None
    window: NDArray[np.float64] | None
    reference: PreparedReference
    device: str = "cpu"
    gpu_device_id: int = 0


def _roi_centers(shape: tuple[int, int], roi_size: int, roi_count: int,
                 roi_layout: str) -> NDArray[np.float64]:
    if roi_layout == "annular":
        return annular_rois(shape, roi_size, roi_count)
    if roi_layout == "grid":
        return grid_rois(shape, roi_size, roi_count)
    raise ValueError("roi_layout must be 'annular' or 'grid'")


def prepare_analysis(reference: ArrayLike, roi_size: int, roi_count: int,
                     roi_filter: tuple[float, float, bool, bool] | None = None,
                     roi_layout: str = "grid", device: str = "cpu",
                     gpu_device_id: int = 0) -> AnalysisPlan:
    """Prepare a fixed reference pattern once for a scan of many patterns."""
    ref = np.asarray(reference)
    centers = _roi_centers(ref.shape, roi_size, roi_count, roi_layout)
    frequency_filter, window = (roi_frequency_filter(roi_size, *roi_filter)
                                if roi_filter is not None else (None, None))
    if device == "cpu":
        cache = prepare_reference(ref, centers, roi_size, frequency_filter, window)
    elif device == "gpu":
        from .correlation_gpu import prepare_reference_gpu
        cache = prepare_reference_gpu(ref, centers, roi_size, frequency_filter,
                                      window, gpu_device_id)
    else:
        raise ValueError("ROI device must be 'cpu' or 'gpu'")
    return AnalysisPlan(centers, frequency_filter, window, cache,
                        device, gpu_device_id)


def roi_frequency_filter(size: int, low: float, high: float, soften_low: bool = True,
                         soften_high: bool = True) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reproduce CalcFShift's radial FFT mask and cosine ROI window."""
    if size < 5 or low < 0 or high <= low:
        raise ValueError("invalid ROI frequency filter")
    i, j = np.meshgrid(np.arange(1, size + 1), np.arange(1, size + 1))
    center = np.floor((size + 1) / 2 + 0.5)
    distance = np.hypot(i - center, j - center)
    rejected = np.where((distance < low) | (distance > high), 1.0, 0.0)
    if soften_high:
        band = (distance > high) & (distance < high + 13)
        rejected[band] = erf((distance[band] - high) / 13 * np.pi)
    if soften_low:
        band = (distance < low) & (distance > low - 13)
        rejected[band] = erf((low - distance[band]) / 13 * np.pi)
    frequency_filter = np.fft.fftshift(1 - rejected)
    window_center = (size + 1) / 2
    window = np.cos((i - window_center) * np.pi / size) * np.cos((j - window_center) * np.pi / size)
    return frequency_filter, window


def _rotation_phosphor_to_sample(alpha: float) -> NDArray[np.float64]:
    return np.array([[0, -np.cos(alpha), -np.sin(alpha)],
                     [-1, 0, 0], [0, np.sin(alpha), -np.cos(alpha)]], dtype=np.float64)


def _r_squared(measured: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    residual = np.sum((measured - predicted) ** 2)
    total = np.sum((measured - measured.mean()) ** 2)
    return float(1 - residual / total) if total > 0 else float("nan")


def fit_deformation(
    shifts: list[CorrelationResult],
    pixel_size: int,
    reference_pc: tuple[float, float, float],
    scan_pc: tuple[float, float, float],
    orientation: ArrayLike,
    sample_tilt: float,
    camera_elevation: float,
    material: Material,
    standard_deviation: float = 2.0,
    phosphor_to_sample: ArrayLike | None = None,
) -> DeformationResult:
    """Fit CalcFShift's real-reference, free-surface deformation solution.

    Pattern centers are normalized to pattern width. Angles are radians.
    The resulting F and strain are in the crystal frame; stress is in GPa.
    """
    if len(shifts) < 4:
        raise ValueError("at least four ROI shifts are required")
    if pixel_size <= 0 or reference_pc[2] <= 0 or scan_pc[2] <= 0:
        raise ValueError("pixel size and pattern center depths must be positive")
    g = np.asarray(orientation, dtype=np.float64)
    if g.shape == (3,):
        g = euler_to_matrix(*g)
    if g.shape != (3, 3) or not np.allclose(g @ g.T, np.eye(3), atol=1e-5):
        raise ValueError("orientation must be a rotation matrix or three Euler angles")
    points = np.array([(row.x, row.y) for row in shifts], dtype=np.float64)
    measured = np.array([(row.dx, row.dy) for row in shifts], dtype=np.float64)
    xstar, ystar, zstar = reference_pc
    r0 = np.column_stack((
        xstar - (points[:, 0] + 1) / pixel_size,
        1 - ystar - (points[:, 1] + 1) / pixel_size,
        np.full(len(points), -zstar),
    ))
    q0 = np.column_stack((-measured[:, 0], -measured[:, 1], np.zeros(len(points)))) / pixel_size
    ratio = reference_pc[2] / scan_pc[2]
    translation = np.array([
        reference_pc[0] - scan_pc[0], scan_pc[1] - reference_pc[1],
        scan_pc[2] - reference_pc[2],
    ])
    q0 = (q0 - translation) * ratio - r0 * (1 - ratio)
    corrected = -q0[:, :2] * pixel_size
    lengths = np.linalg.norm(r0, axis=1)
    r = r0 / lengths[:, None]
    q = q0 / lengths[:, None]

    keep = np.ones(len(shifts), dtype=bool)
    for axis in range(2):
        spread = np.std(corrected[:, axis], ddof=1)
        if spread > 0:
            keep &= np.abs(corrected[:, axis]) < 129
            keep &= np.abs(corrected[:, axis] - corrected[:, axis].mean()) < standard_deviation * spread
    if keep.sum() < 4:
        raise ValueError("fewer than four ROIs remain after shift outlier filtering")
    r, q = r[keep], q[keep]
    a1 = np.column_stack((r[:, 0] * r[:, 2], r[:, 1] * r[:, 2], r[:, 2] ** 2,
                          np.zeros(len(r)), np.zeros(len(r)), np.zeros(len(r)),
                          -(r[:, 0] + q[:, 0]) * r[:, 0],
                          -(r[:, 0] + q[:, 0]) * r[:, 1],
                          -(r[:, 0] + q[:, 0]) * r[:, 2]))
    a2 = np.column_stack((np.zeros(len(r)), np.zeros(len(r)), np.zeros(len(r)),
                          r[:, 0] * r[:, 2], r[:, 1] * r[:, 2], r[:, 2] ** 2,
                          -(r[:, 1] + q[:, 1]) * r[:, 0],
                          -(r[:, 1] + q[:, 1]) * r[:, 1],
                          -(r[:, 1] + q[:, 1]) * r[:, 2]))
    alpha = np.pi / 2 - sample_tilt + camera_elevation
    qps = (_rotation_phosphor_to_sample(alpha) if phosphor_to_sample is None
           else np.asarray(phosphor_to_sample, dtype=np.float64))
    if qps.shape != (3, 3) or not np.allclose(qps @ qps.T, np.eye(3), atol=1e-5):
        raise ValueError("phosphor_to_sample must be a rotation matrix")
    qpc = g @ qps
    crystal_stiffness = material.stiffness()
    crystal_to_phosphor = qpc.T
    phosphor_stiffness = np.einsum(
        "ia,jb,kc,ld,abcd->ijkl", crystal_to_phosphor, crystal_to_phosphor,
        crystal_to_phosphor, crystal_to_phosphor, crystal_stiffness,
    )
    normal = qps.T @ np.array([0.0, 0.0, 1.0])
    boundary = np.array([
        sum(phosphor_stiffness[2, j, k, l] * normal[j] for j in range(3))
        for k in range(3) for l in range(3)
    ]) / 100
    design = np.vstack((a1, a2, boundary))
    response = np.concatenate((q[:, 0] * r[:, 2], q[:, 1] * r[:, 2], [0.0]))
    solution, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    if rank < 9:
        raise ValueError("ROI layout cannot determine all deformation components")
    phosphor_deformation = np.eye(3) + solution.reshape(3, 3)
    f = qpc @ phosphor_deformation @ qpc.T
    _, singular_values, vh = np.linalg.svd(f)
    stretch = (vh.T * singular_values) @ vh
    strain = stretch - np.eye(3)
    stress = np.einsum("ijkl,kl->ij", crystal_stiffness, strain)
    predicted = theoretical_pixel_shift(g, reference_pc, points, f, pixel_size, alpha, qps)
    error = np.sqrt(np.mean(np.sum((predicted[keep] - corrected[keep]) ** 2, axis=1)))
    return DeformationResult(f, strain, stress, g, float(error), int(keep.sum()),
                             len(shifts), tuple(shifts), keep)


def analyze_pair(
    reference: ArrayLike,
    scan: ArrayLike,
    *,
    roi_size: int,
    roi_count: int,
    reference_pc: tuple[float, float, float],
    scan_pc: tuple[float, float, float],
    orientation: ArrayLike,
    sample_tilt: float,
    camera_elevation: float,
    material: Material,
    standard_deviation: float = 2.0,
    roi_filter: tuple[float, float, bool, bool] | None = None,
    prepared: AnalysisPlan | None = None,
    phosphor_to_sample: ArrayLike | None = None,
    roi_layout: str = "grid",
    device: str = "cpu",
    gpu_device_id: int = 0,
    subpixel_method: str = "parabolic_1d",
    remapping: bool = False,
    initial_shifts: list[CorrelationResult] | tuple[CorrelationResult, ...] | None = None,
) -> DeformationResult:
    """Correlate a pattern pair, optionally back-rotating for a second pass."""
    ref = np.asarray(reference)
    target = np.asarray(scan)
    if ref.shape != target.shape or ref.ndim != 2 or ref.shape[0] != ref.shape[1]:
        raise ValueError("reference and scan must be square images of equal size")
    centers = (prepared.centers if prepared is not None else
               _roi_centers(ref.shape, roi_size, roi_count, roi_layout))
    if device not in ("cpu", "gpu"):
        raise ValueError("ROI device must be 'cpu' or 'gpu'")
    if prepared is not None and (prepared.device != device or
                                 prepared.gpu_device_id != gpu_device_id):
        raise ValueError("prepared ROI reference uses a different device")
    ratio = reference_pc[2] / scan_pc[2]
    translation = np.array([reference_pc[0] - scan_pc[0],
                            scan_pc[1] - reference_pc[1], scan_pc[2] - reference_pc[2]])
    r0 = np.column_stack((reference_pc[0] - (centers[:, 0] + 1) / ref.shape[0],
                          1 - reference_pc[1] - (centers[:, 1] + 1) / ref.shape[0],
                          np.full(len(centers), -reference_pc[2])))
    shifted = -np.array([reference_pc[0], 1 - reference_pc[1], -reference_pc[2]]) + translation + r0 / ratio
    scan_centers = -shifted[:, :2] * ref.shape[0] - 1
    valid = []
    for target_x, target_y in scan_centers:
        try:
            _roi_slice(target_x, target_y, roi_size, ref.shape)
            valid.append(True)
        except ValueError:
            valid.append(False)
    centers, scan_centers = centers[valid], scan_centers[valid]
    if len(centers) < 4:
        raise ValueError("pattern-center correction places too many ROIs outside the scan")
    options = {}
    if prepared is not None:
        options = {"frequency_filter": prepared.frequency_filter,
                   "window": prepared.window,
                   "prepared_reference": prepared.reference}
    elif roi_filter is not None:
        frequency_filter, window = roi_frequency_filter(roi_size, *roi_filter)
        options = {"frequency_filter": frequency_filter, "window": window}
    if initial_shifts is not None:
        shifts = list(initial_shifts)
    elif device == "gpu":
        from .correlation_gpu import measure_pattern_shifts_gpu
        shifts = measure_pattern_shifts_gpu(
            ref, target, centers, roi_size, scan_centers=scan_centers,
            gpu_device_id=gpu_device_id, subpixel_method=subpixel_method, **options)
    else:
        shifts = measure_pattern_shifts(ref, target, centers, roi_size,
                                        scan_centers=scan_centers,
                                        subpixel_method=subpixel_method, **options)
    first = fit_deformation(shifts, ref.shape[0], reference_pc, scan_pc, orientation,
                            sample_tilt, camera_elevation, material, standard_deviation,
                            phosphor_to_sample)
    if not remapping:
        return first

    # The first-pass right polar factor is a finite rotation from reference
    # to scan. Undo it in image space before measuring the small residual.
    u, _, vh = np.linalg.svd(first.deformation)
    rotation_crystal = u @ vh
    if np.linalg.det(rotation_crystal) <= 0:
        raise ValueError("first-pass deformation has no proper polar rotation")
    g = first.orientation
    alpha = np.pi / 2 - sample_tilt + camera_elevation
    qps = (_rotation_phosphor_to_sample(alpha) if phosphor_to_sample is None
           else np.asarray(phosphor_to_sample, dtype=np.float64))
    qpc = g @ qps
    rotation_phosphor = qpc.T @ rotation_crystal @ qpc

    from .remapping import remap_pattern, valid_roi_centers
    remapped, valid_pixels = remap_pattern(
        target, reference_pc, scan_pc, rotation_phosphor, device, gpu_device_id)
    second_centers = centers[valid_roi_centers(centers, roi_size, valid_pixels)]
    if len(second_centers) < 4:
        raise ValueError("remapping places too many ROIs outside the scan")
    if device == "gpu":
        residual_shifts = measure_pattern_shifts_gpu(
            ref, remapped, second_centers, roi_size, gpu_device_id=gpu_device_id,
            subpixel_method=subpixel_method, **options)
    else:
        residual_shifts = measure_pattern_shifts(
            ref, remapped, second_centers, roi_size,
            subpixel_method=subpixel_method, **options)
    residual = fit_deformation(
        residual_shifts, ref.shape[0], reference_pc, reference_pc, orientation,
        sample_tilt, camera_elevation, material, standard_deviation,
        phosphor_to_sample)
    deformation = rotation_crystal @ residual.deformation
    from .rotations import sample_rotation_vector_mrad
    return replace(
        residual, deformation=deformation,
        first_pass_shifts=tuple(shifts),
        first_pass_rms_shift_error=first.rms_shift_error,
        remapping_rotation_mrad=float(np.linalg.norm(
            sample_rotation_vector_mrad(rotation_crystal, g))),
        residual_rotation_mrad=float(np.linalg.norm(
            sample_rotation_vector_mrad(residual.deformation, g))),
    )
