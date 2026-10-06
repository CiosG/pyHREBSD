"""Fit a scan-position PC plane, corresponding to OpenXY's PCPlaneFit option."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .correlation import measure_pattern_shifts


@dataclass(frozen=True)
class PCPlane:
    # Rows are x*, y*, z*; columns are intercept, map column, map row.
    coefficients: np.ndarray
    points_used: int
    points_total: int
    rms_residual_pixels: tuple[float, float, float]
    max_residual_pixels: tuple[float, float, float]
    pattern_width: int
    source_path: str
    method: str = "affine_plane_from_dataset_pattern_centers"
    diagnostics: dict = field(default_factory=dict)

    def at(self, index: int, columns: int) -> tuple[float, float, float]:
        column, row = index % columns, index // columns
        values = self.coefficients @ np.array([1.0, column, row])
        return tuple(float(value) for value in values)

    def report(self) -> dict:
        return {
            "method": self.method,
            "source_path": self.source_path,
            "coordinates": "zero-based map column and row",
            "pattern_center_units": "fraction of cropped square pattern width",
            "coefficients": {
                component: dict(zip(("intercept", "per_column", "per_row"),
                                    map(float, row)))
                for component, row in zip(("x", "y", "z"), self.coefficients)
            },
            "points_used": self.points_used,
            "points_total": self.points_total,
            "cropped_pattern_width_pixels": self.pattern_width,
            "rms_residual_pixels": dict(zip(("x", "y", "z"), self.rms_residual_pixels)),
            "max_residual_pixels": dict(zip(("x", "y", "z"), self.max_residual_pixels)),
            "residual_definition": (
                "difference from the dataset affine PC plane over the map"
                if self.method == "external_effective_pixel_size_beam_shift" else
                "measured centre-ROI X shift fit and H5 Y/Z plane fit"
                if self.method == "center_roi_beam_shift_calibration" else
                "H5 per-point PC residual after affine plane fit"),
            "absolute_pc_calibrated_independently": False,
            **({"beam_shift_calibration": self.diagnostics}
               if self.diagnostics else {}),
        }


def fit_pc_plane(reader) -> PCPlane | None:
    """Robustly smooth the stored per-point PC over a regular map.

    The fit estimates scan-position drift; it cannot establish absolute PC
    without an independent physical or diffraction-based calibration.
    """
    if reader.x_cells is None or reader.y_cells is None:
        raise ValueError("PC plane calibration needs X Cells and Y Cells")
    if reader.count != reader.x_cells * reader.y_cells:
        raise ValueError("PC plane calibration needs a complete rectangular scan")
    centers = reader.pattern_centers()
    if centers is None:
        return None
    rows, columns = np.divmod(np.arange(reader.count), reader.x_cells)
    design = np.column_stack((np.ones(reader.count), columns, rows))
    valid = np.all(np.isfinite(centers), axis=1) & (centers[:, 2] > 0)
    if valid.sum() < 4:
        raise ValueError("at least four valid pattern centers are needed for PC calibration")
    for _ in range(5):
        coefficients = np.linalg.lstsq(design[valid], centers[valid], rcond=None)[0]
        residual = centers - design @ coefficients
        distance = np.linalg.norm(residual, axis=1)
        median = np.median(distance[valid])
        mad = np.median(np.abs(distance[valid] - median))
        threshold = median + max(6 * 1.4826 * mad, 1e-5 / min(reader.width, reader.height))
        new_valid = valid & (distance <= threshold)
        if new_valid.sum() < 4 or np.array_equal(new_valid, valid):
            break
        valid = new_valid
    coefficients = np.linalg.lstsq(design[valid], centers[valid], rcond=None)[0]
    residual_pixels = (centers[valid] - design[valid] @ coefficients) * min(reader.width, reader.height)
    rms = np.sqrt(np.mean(residual_pixels ** 2, axis=0))
    maximum = np.max(np.abs(residual_pixels), axis=0)
    return PCPlane(coefficients.T, int(valid.sum()), reader.count,
                   tuple(float(v) for v in rms), tuple(float(v) for v in maximum),
                   min(reader.width, reader.height), reader.pc_source_path)


def _fit_shift_line(reference, pattern_reader, reference_index, points,
                    center, roi_size, coordinate_axis):
    """Fit the displacement of one PC-centred ROI along a scan line."""
    coordinates = np.array([point[coordinate_axis] -
                            (reference_index % pattern_reader.x_cells if coordinate_axis == 0
                             else reference_index // pattern_reader.x_cells)
                            for point in points], dtype=float)
    shifts = []
    for column, row in points:
        target = pattern_reader.pattern(pattern_reader.map_index(column, row))
        result, = measure_pattern_shifts(reference, target, center, roi_size,
                                         subpixel_method="quadratic_2d")
        shifts.append((result.dx, result.dy))
    shifts = np.asarray(shifts)
    design = np.column_stack((np.ones(len(points)), coordinates))
    valid = np.ones(len(points), dtype=bool)
    for _ in range(5):
        coefficients = np.linalg.lstsq(design[valid], shifts[valid], rcond=None)[0]
        residual_norm = np.linalg.norm(shifts - design @ coefficients, axis=1)
        median = np.median(residual_norm[valid])
        mad = np.median(np.abs(residual_norm[valid] - median))
        next_valid = residual_norm <= median + max(4 * 1.4826 * mad, 0.05)
        if next_valid.sum() < 6 or np.array_equal(next_valid, valid):
            break
        valid = next_valid
    coefficients = np.linalg.lstsq(design[valid], shifts[valid], rcond=None)[0]
    residual = shifts[valid] - design[valid] @ coefficients
    all_residual = shifts - design @ coefficients
    return {"slope_pixels_per_map_step": coefficients[1].tolist(),
            "intercept_pixels": coefficients[0].tolist(),
            "rms_residual_pixels": np.sqrt(np.mean(residual ** 2, axis=0)).tolist(),
            "max_residual_pixels": np.max(np.abs(residual), axis=0).tolist(),
            "rms_residual_all_pixels": np.sqrt(np.mean(all_residual ** 2, axis=0)).tolist(),
            "max_residual_all_pixels": np.max(np.abs(all_residual), axis=0).tolist(),
            "coordinates_map_steps": coordinates.tolist(),
            "measured_shifts_pixels": shifts.tolist(),
            "inlier_mask": valid.tolist(),
            "points_used": int(valid.sum()), "points_total": len(points),
            "line_start": list(points[0]), "line_end": list(points[-1])}, residual


def _map_step_um(reader, axis: str) -> float:
    if reader._header is None or f"{axis} Step" not in reader._header:
        raise ValueError(f"beam-shift calibration needs dataset {axis} Step metadata")
    values = np.asarray(reader._header[f"{axis} Step"][()], dtype=float).reshape(-1)
    if len(values) != 1 or not np.isfinite(values[0]) or values[0] <= 0:
        raise ValueError(f"beam-shift calibration needs a positive {axis} Step")
    return float(values[0])


def _line_positions(anchor: int, cells: int, step_um: float,
                    extent_um: float | None, spacing: int) -> list[int]:
    if extent_um is None:
        positions = list(range(0, cells, spacing))
        if positions[-1] != cells - 1:
            positions.append(cells - 1)
        if len(positions) < 7:
            raise ValueError("calibration line needs at least seven map points")
        return positions
    direction = 1 if cells - 1 - anchor >= anchor else -1
    available = cells - 1 - anchor if direction == 1 else anchor
    span = min(available, int(extent_um / step_um))
    if span < 6 * spacing:
        raise ValueError("calibration line needs at least seven map points")
    positions = [anchor + direction * distance
                 for distance in range(0, span + 1, spacing)]
    endpoint = anchor + direction * span
    if positions[-1] != endpoint:
        positions.append(endpoint)
    return positions


def measure_effective_pixel_size(reader, reference_index: int,
                                 *, roi_size_percent: float = 25.0,
                                 spacing: int = 1,
                                 extent_um: float | None = None,
                                 pattern_center: tuple[float, float, float] | None = None) -> dict:
    """Measure effective pixel size along an X line on a strain-free scan.

    Returns a *positive* effective pixel size and the measured detector-X
    direction separately. Patterns must have the same pixel resolution and
    detector geometry as those to which the calibration will be applied.
    By default every point in the reference pattern's map row is used.
    """
    if reader.x_cells is None or reader.y_cells is None:
        raise ValueError("beam-shift calibration needs a rectangular H5OINA scan")
    if (spacing < 1 or (extent_um is not None and extent_um <= 0)
            or not 0 < roi_size_percent <= 50):
        raise ValueError("invalid beam-shift calibration sampling or ROI size")
    reference = reader.pattern(reference_index)
    size = reference.shape[0]
    if reference.ndim != 2 or reference.shape[1] != size:
        raise ValueError("beam-shift calibration needs square patterns")
    roi_size = max(32, int(round(size * roi_size_percent / 100)))
    if roi_size > size // 2:
        raise ValueError("calibration pattern is too small for the selected ROI")
    pc = (pattern_center if pattern_center is not None else
          reader.pattern_center(reference_index))
    if pc is None or len(pc) != 3 or not np.all(np.isfinite(pc)):
        raise ValueError("calibration needs a valid reference PC or pattern_center")
    center = np.array([[pc[0] * size - 1, (1 - pc[1]) * size - 1]])
    x_step = _map_step_um(reader, "X")
    column0 = reference_index % reader.x_cells
    row0 = reference_index // reader.x_cells
    columns = _line_positions(column0, reader.x_cells, x_step, extent_um, spacing)
    points = [(column, row0) for column in columns]
    fit, _ = _fit_shift_line(reference, reader, reference_index,
                            points, center, roi_size, 0)
    slope = float(fit["slope_pixels_per_map_step"][0])
    if not np.isfinite(slope) or abs(slope) < 1e-8:
        raise ValueError("calibration did not measure a nonzero X shift slope")
    y_shifts = np.array(fit["measured_shifts_pixels"])[:, 1]
    y_shift_max_abs = float(np.max(np.abs(y_shifts - y_shifts[0])))
    return {
        "method": "beam_shift_x_line_effective_pixel_size",
        "input_file": str(reader.path), "scan_group": reader.scan_group,
        "pattern_type": reader.pattern_type,
        "pattern_width_pixels": size,
        "reference_map_point": [column0, row0],
        "reference_pattern_center": list(map(float, pc)),
        "roi_center_pixels": center[0].tolist(), "roi_size_pixels": roi_size,
        "x_step_um": x_step,
        "line_extent_um": abs(columns[-1] - columns[0]) * x_step,
        "line_spacing_map_steps": spacing,
        "measured_x_shift_pixels_per_step": slope,
        "measured_x_shift_pixels_per_um": slope / x_step,
        "effective_pixel_size_um_per_pixel": abs(x_step / slope),
        "detector_x_shift_sign": 1 if slope > 0 else -1,
        "max_abs_y_shift_pixels": y_shift_max_abs,
        "y_alignment_within_1_pixel": y_shift_max_abs <= 1.0,
        "x_fit_intercept_pixels": fit["intercept_pixels"][0],
        "x_fit_rms_all_pixels": fit["rms_residual_all_pixels"][0],
        "x_fit_max_all_pixels": fit["max_residual_all_pixels"][0],
        "line_fit": fit,
    }


def measure_effective_pixel_size_homography(
    reader, reference_index: int, *, spacing: int = 1,
    extent_um: float | None = None,
    pattern_center: tuple[float, float, float] | None = None,
    margin_fraction: float = 0.08, max_iterations: int = 250,
    tolerance: float = 1e-5, device: str = "cpu", gpu_device_id: int = 0,
) -> dict:
    """Measure beam-shift EPS from a whole-pattern projective registration.

    Each target pattern is registered to the reference with the same global
    homography used by the analysis path. The fitted warp is evaluated at the
    reference pattern centre and regressed against scan-X displacement.
    """
    from .homography import (_warp_points, prepare_homography,
                             register_homography)

    if reader.x_cells is None or reader.y_cells is None:
        raise ValueError("beam-shift calibration needs a rectangular H5OINA scan")
    if spacing < 1 or (extent_um is not None and extent_um <= 0):
        raise ValueError("invalid beam-shift calibration sampling")
    reference = reader.pattern(reference_index)
    size = reference.shape[0]
    if reference.ndim != 2 or reference.shape[1] != size:
        raise ValueError("beam-shift calibration needs square patterns")
    pc = (pattern_center if pattern_center is not None else
          reader.pattern_center(reference_index))
    if pc is None or len(pc) != 3 or not np.all(np.isfinite(pc)):
        raise ValueError("calibration needs a valid reference PC or pattern_center")
    pc_pixel = np.array(
        [[pc[0] * size - 1, (1 - pc[1]) * size - 1]], dtype=float)
    x_step = _map_step_um(reader, "X")
    column0 = reference_index % reader.x_cells
    row0 = reference_index // reader.x_cells
    columns = _line_positions(column0, reader.x_cells, x_step, extent_um, spacing)
    coordinates = np.asarray(columns, dtype=float) - column0
    plan = prepare_homography(
        reference, margin_fraction=margin_fraction, device=device,
        gpu_device_id=gpu_device_id)
    shifts, znssd, iterations, converged = [], [], [], []
    for column in columns:
        target = reader.pattern(reader.map_index(column, row0))
        h, rms, count, good = register_homography(
            plan, target, max_iterations=max_iterations, tolerance=tolerance)
        shifts.append((_warp_points(h, pc_pixel)[0] - pc_pixel[0]).tolist())
        znssd.append(float(rms))
        iterations.append(int(count))
        converged.append(bool(good))
    shifts = np.asarray(shifts, dtype=float)
    design = np.column_stack((np.ones(len(columns)), coordinates))
    valid = np.asarray(converged, dtype=bool) & np.all(np.isfinite(shifts), axis=1)
    if valid.sum() < 7:
        raise ValueError("homography calibration needs at least seven converged points")
    for _ in range(5):
        coefficients = np.linalg.lstsq(design[valid], shifts[valid], rcond=None)[0]
        residual_norm = np.linalg.norm(shifts - design @ coefficients, axis=1)
        median = np.median(residual_norm[valid])
        mad = np.median(np.abs(residual_norm[valid] - median))
        next_valid = (np.asarray(converged, dtype=bool) &
                      (residual_norm <= median + max(4 * 1.4826 * mad, 0.05)))
        if next_valid.sum() < 7 or np.array_equal(next_valid, valid):
            break
        valid = next_valid
    coefficients = np.linalg.lstsq(design[valid], shifts[valid], rcond=None)[0]
    residual = shifts[valid] - design[valid] @ coefficients
    all_residual = shifts - design @ coefficients
    slope = float(coefficients[1, 0])
    if not np.isfinite(slope) or abs(slope) < 1e-8:
        raise ValueError("calibration did not measure a nonzero X shift slope")
    fit = {
        "slope_pixels_per_map_step": coefficients[1].tolist(),
        "intercept_pixels": coefficients[0].tolist(),
        "rms_residual_pixels": np.sqrt(np.mean(residual ** 2, axis=0)).tolist(),
        "max_residual_pixels": np.max(np.abs(residual), axis=0).tolist(),
        "rms_residual_all_pixels": np.sqrt(np.mean(all_residual ** 2, axis=0)).tolist(),
        "max_residual_all_pixels": np.max(np.abs(all_residual), axis=0).tolist(),
        "coordinates_map_steps": coordinates.tolist(),
        "measured_shifts_pixels": shifts.tolist(),
        "inlier_mask": valid.tolist(),
        "points_used": int(valid.sum()), "points_total": len(columns),
        "line_start": [int(columns[0]), row0],
        "line_end": [int(columns[-1]), row0],
    }
    y_shift_max_abs = float(np.max(np.abs(shifts[:, 1] - shifts[0, 1])))
    return {
        "method": "beam_shift_x_line_homography_effective_pixel_size",
        "input_file": str(reader.path), "scan_group": reader.scan_group,
        "pattern_type": reader.pattern_type,
        "pattern_width_pixels": size,
        "reference_map_point": [column0, row0],
        "reference_pattern_center": list(map(float, pc)),
        "evaluation_point_pixels": pc_pixel[0].tolist(),
        "x_step_um": x_step,
        "line_extent_um": abs(columns[-1] - columns[0]) * x_step,
        "line_spacing_map_steps": spacing,
        "measured_x_shift_pixels_per_step": slope,
        "measured_x_shift_pixels_per_um": slope / x_step,
        "effective_pixel_size_um_per_pixel": abs(x_step / slope),
        "detector_x_shift_sign": 1 if slope > 0 else -1,
        "max_abs_y_shift_pixels": y_shift_max_abs,
        "y_alignment_within_1_pixel": y_shift_max_abs <= 1.0,
        "x_fit_intercept_pixels": float(coefficients[0, 0]),
        "x_fit_rms_all_pixels": float(fit["rms_residual_all_pixels"][0]),
        "x_fit_max_all_pixels": float(fit["max_residual_all_pixels"][0]),
        "homography_margin_fraction": float(margin_fraction),
        "homography_max_iterations": int(max_iterations),
        "homography_tolerance": float(tolerance),
        "homography_device": device,
        "homography_converged": converged,
        "homography_iterations": iterations,
        "homography_znssd_rms": znssd,
        "line_fit": fit,
    }


def pc_plane_from_effective_pixel_size(reader,
                                       reference_index: int,
                                       effective_pixel_size_um: float,
                                       detector_x_shift_sign: int | str = "auto") -> PCPlane:
    """Apply a separately calibrated EPS to scan-X PC drift, once.

    The H5 PC plane supplies the absolute PC and all other gradients. The
    external EPS replaces only the detector-X displacement per map column;
    both ROI and homography solvers then use this PC plane directly.
    """
    if not np.isfinite(effective_pixel_size_um) or effective_pixel_size_um <= 0:
        raise ValueError("effective_pixel_size_um must be positive")
    baseline = fit_pc_plane(reader)
    if baseline is None:
        raise ValueError("external beam-shift EPS requires a dataset reference PC")
    if detector_x_shift_sign == "auto":
        h5_slope = float(baseline.coefficients[0, 1])
        if abs(h5_slope) < 1e-12:
            raise ValueError("H5 PC X slope is zero; set detector_x_shift_sign")
        detector_x_shift_sign = 1 if h5_slope > 0 else -1
    if detector_x_shift_sign not in (-1, 1):
        raise ValueError("detector_x_shift_sign must be 'auto', -1 or +1")
    x_step = _map_step_um(reader, "X")
    column0, row0 = reference_index % reader.x_cells, reference_index // reader.x_cells
    reference_pc = baseline.at(reference_index, reader.x_cells)
    coefficients = baseline.coefficients.copy()
    measured_slope = detector_x_shift_sign * x_step / effective_pixel_size_um
    coefficients[0, 1] = measured_slope / baseline.pattern_width
    coefficients[0, 0] = reference_pc[0] - coefficients[0, 1] * column0 - \
        coefficients[0, 2] * row0
    columns = np.arange(reader.count) % reader.x_cells
    rows = np.arange(reader.count) // reader.x_cells
    design = np.column_stack((np.ones(reader.count), columns, rows))
    difference_pixels = design @ (coefficients - baseline.coefficients).T * \
        baseline.pattern_width
    difference_rms = tuple(float(value) for value in
                           np.sqrt(np.mean(difference_pixels ** 2, axis=0)))
    difference_max = tuple(float(value) for value in
                           np.max(np.abs(difference_pixels), axis=0))
    diagnostics = {
        "description": "external effective pixel size replaces only "
                       "X-versus-map-column PC slope; reference PC and other "
                       "gradients remain from H5OINA",
        "effective_pixel_size_um_per_pixel": float(effective_pixel_size_um),
        "detector_x_shift_sign": detector_x_shift_sign,
        "x_step_um": x_step,
        "applied_x_shift_pixels_per_map_step": measured_slope,
        "h5_x_shift_pixels_per_map_step":
            float(baseline.coefficients[0, 1] * baseline.pattern_width),
        "difference_from_h5_plane_rms_pixels": dict(zip("xyz", difference_rms)),
        "difference_from_h5_plane_max_pixels": dict(zip("xyz", difference_max)),
        "h5_plane_fit_rms_residual_pixels": dict(
            zip("xyz", baseline.rms_residual_pixels)),
        "reference_map_point": [column0, row0],
        "reference_pc": list(reference_pc),
        "vertical_pc_gradient_source": "H5OINA PC plane",
        "absolute_pc_calibrated_independently": False,
    }
    return PCPlane(coefficients, baseline.points_used, baseline.points_total,
                   difference_rms, difference_max,
                   baseline.pattern_width, baseline.source_path,
                   "external_effective_pixel_size_beam_shift", diagnostics)


def fit_beam_shift_pc_plane(reader, pattern_reader,
                            reference_index: int, *, roi_size: int = 256,
                            spacing: int = 8, extent_um: float = 100.0) -> PCPlane:
    """Calibrate scan-induced PC translation from two strain-free lines.

    The H5 PC plane supplies the absolute PC and all gradients except
    X-versus-map-column. The vertical line is measured for diagnostics only;
    its slope is not used for the EPS calibration. The calibration line must
    be strain-free and use the analysis geometry.
    """
    if (reader.count != pattern_reader.count or
            reader.x_cells != pattern_reader.x_cells or
            reader.y_cells != pattern_reader.y_cells):
        raise ValueError("PC and pattern readers must describe the same scan")
    if reader.x_cells is None or reader.y_cells is None:
        raise ValueError("beam-shift calibration needs a rectangular scan")
    if spacing < 1 or extent_um <= 0:
        raise ValueError("spacing and extent_um must be positive")
    baseline = fit_pc_plane(reader)
    if baseline is None:
        raise ValueError("beam-shift calibration needs a reference PC in H5OINA")
    reference = pattern_reader.pattern(reference_index)
    size = reference.shape[0]
    if reference.ndim != 2 or reference.shape[1] != size:
        raise ValueError("beam-shift calibration needs square patterns")
    if roi_size < 32 or roi_size > size // 2:
        raise ValueError("beam-shift ROI must be at least 32 and at most half the pattern")
    pc = baseline.at(reference_index, reader.x_cells)
    center = np.array([[pc[0] * size - 1, (1 - pc[1]) * size - 1]])
    column0, row0 = reference_index % reader.x_cells, reference_index // reader.x_cells
    header = reader._header
    x_step = float(np.asarray(header["X Step"][()]).reshape(-1)[0])
    y_step = float(np.asarray(header["Y Step"][()]).reshape(-1)[0])
    if not np.all(np.isfinite((x_step, y_step))) or x_step <= 0 or y_step <= 0:
        raise ValueError("beam-shift calibration needs positive X/Y step sizes")

    def line(anchor, cells, step_um):
        direction = 1 if cells - 1 - anchor >= anchor else -1
        available = cells - 1 - anchor if direction == 1 else anchor
        span = min(available, int(extent_um / step_um))
        if span < 6 * spacing:
            raise ValueError("reference is too close to the scan edge for beam calibration")
        positions = [anchor + direction * distance
                     for distance in range(0, span + 1, spacing)]
        endpoint = anchor + direction * span
        if positions[-1] != endpoint:
            positions.append(endpoint)
        return positions

    x_points = [(x, row0) for x in line(column0, reader.x_cells, x_step)]
    y_points = [(column0, y) for y in line(row0, reader.y_cells, y_step)]
    x_fit, x_residual = _fit_shift_line(reference, pattern_reader, reference_index,
                                        x_points, center, roi_size, 0)
    y_fit, _ = _fit_shift_line(reference, pattern_reader, reference_index,
                               y_points, center, roi_size, 1)
    x_slope = float(x_fit["slope_pixels_per_map_step"][0])
    if not np.isfinite(x_slope) or abs(x_slope) < 1e-8:
        raise ValueError("beam-shift calibration did not measure a valid X slope")
    coefficients = baseline.coefficients.copy()
    coefficients[0, 1] = x_slope / size
    coefficients[0, 0] = pc[0] - coefficients[0, 1] * column0 - \
        coefficients[0, 2] * row0
    diagnostics = {
        "description": "PC-centred ROI shifts on two lines; only the X shift "
                       "per map column replaces the H5 PC slope. The vertical "
                       "line is diagnostic; all other gradients remain H5 anchored",
        "reference_map_point": [column0, row0],
        "reference_pc": list(pc), "calibration_pattern_type": pattern_reader.pattern_type,
        "roi_size_pixels": roi_size, "roi_center_pixels": center[0].tolist(),
        "line_spacing_map_steps": spacing, "line_extent_um": extent_um,
        "map_step_um": {"x": x_step, "y": y_step},
        "x_line": x_fit, "y_line": y_fit,
        "effective_pixel_size_um_per_pixel_x": abs(x_step / x_slope),
        "vertical_line_used_for_pc": False,
        "h5_pc_slope_pixels_per_map_step": (baseline.coefficients[:, 1:] * size).tolist(),
        "absolute_pc_calibrated_independently": False,
    }
    return PCPlane(coefficients, x_fit["points_used"], x_fit["points_total"],
                   (float(np.sqrt(np.mean(x_residual[:, 0] ** 2))),
                    baseline.rms_residual_pixels[1],
                    baseline.rms_residual_pixels[2]),
                   (float(np.max(np.abs(x_residual[:, 0]))),
                    baseline.max_residual_pixels[1],
                    baseline.max_residual_pixels[2]),
                   size, reader.pc_source_path,
                   "center_roi_beam_shift_calibration", diagnostics)
