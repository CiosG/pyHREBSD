"""Run the PyHREBSD real-pattern pipeline after editing CONFIG below."""

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pyhrebsd.analysis import Material, analyze_pair, prepare_analysis
from pyhrebsd.bcf import BCFReader, correct_pattern_dynamic_lmsd
from pyhrebsd.correlation import roi_size_from_percent
from pyhrebsd.h5oina import H5OINAReader
from pyhrebsd.edax import open_edax
from pyhrebsd.tfs import TFSReader
from pyhrebsd.homography import analyze_homography, prepare_homography
from pyhrebsd.io import read_pattern
from pyhrebsd.pc_calibration import (fit_pc_plane, measure_effective_pixel_size,
                                   measure_effective_pixel_size_homography,
                                   pc_plane_from_effective_pixel_size)
from pyhrebsd.preprocess import (correct_pattern_background,
                               correct_pattern_static_lmsd)
from pyhrebsd.rotations import sample_rotation_vector_mrad


# -------------------------- CONFIG: edit these values --------------------------
CONFIG = {
    # Required: select the run and its input/output.
    "run_mode": "analysis",  # "analysis" or "calibration" on a separate strain-free single-crystal scan
    "input_file": "scan.h5oina",  # .tfs.hdf5, .h5oina, .oh5, .ang, .up2, or .bcf; relative or absolute
    "output_dir": "results_roi_cpu",  # must be new; existing results are not overwritten

    # Required for run_mode="analysis".
    "analysis_method": "roi",  # "roi" or whole-pattern "homography"
    "pattern_type": "processed",  # H5OINA: processed=stored 8-bit, unprocessed=raw 16-bit; TFS: processed only
    # BCF: processed=dynamic-LMSD 8-bit in memory, unprocessed=raw intensity plus selected correction
    "h5_pc_source": "ebsd",  # H5OINA: "ebsd" or "data_processing" (including MapSweeper PC); other readers select their source
    "pc_mode": "array",  # "single" repeats one PC; "array" uses per-point/source PC
    "material_name": "silicon",  # database key or the full material name
    "reference_map_point": None,  # None uses first stored pattern; or zero-based (column, row)
    "pattern_binning": 1,  # use 2, 4, or 8 for block-averaged patterns
    "edax_ang_file": None,  # optional explicit ANG companion path for UP2
    "edax_preprocess_device": "cpu",  # CPU or GPU for processed EDAX patterns

    # Calibration settings; used only for run_mode="calibration".
    "calibration_method": "roi",  # "roi" or whole-pattern "homography"
    "calibration_line_extent_um": None,  # None uses the complete reference row
    "calibration_line_spacing": 1,

    # Optional dataset, material, and PC overrides.
    "material_database": "pyhrebsd/materials.h5",
    "pattern_center_fallback": (0.45, 0.53, 0.65),  # (PCX, PCY, DD), normalized by pattern width
    "beam_shift_effective_pixel_size_um": None,  # used by pc_mode="beam_shift_eps" to build an array from map steps
    "beam_shift_detector_x_sign": "auto",  # infer source PC slope; or copy -1/+1 from calibration

    # Optional geometry overrides; None reads the value from the selected input source.
    "detector_geometry": "full",  # H5OINA/TFS: full uses all 3 detector Euler angles;
    # BCF provides elevation geometry only, so it effectively uses "elevation".
    "reference_euler_degrees": None,
    "sample_tilt_degrees": None,  # None reads sample tilt from the selected input source
    "camera_elevation_degrees": None,  # None reads source-specific detector elevation:
    # H5OINA: detector Euler Phi - 90 deg; TFS: DCStoSCS geometry; BCF: CameraTilt.

    # Optional execution and output settings.
    "workers": max(1, (os.cpu_count() or 2) - 1),  # logical CPU cores minus one or set value by hand
    "save_per_pattern_files": False,  # full scan normally goes to one summary CSV

    # Optional ROI-method settings.
    "roi_device": "cpu",  # set "gpu" after installing matching CuPy
    "roi_gpu_device_id": 0,
    "roi_gpu_workers": 4,
    "roi_layout": "annular",  # center plus one circular ring; "grid" uses a square layout
    "roi_size": None,  # explicit pixel override; None uses roi_size_percent
    "roi_size_percent": 25.0,  # ROI width as a percentage of pattern width
    "roi_count": 48,  # center ROI plus 47 equally spaced ring ROIs
    "roi_remapping": True,  # two-pass projective back-rotation
    "roi_filter": (2.0, 50.0, True, True),  # OpenXY default: low/high FFT radius and softened edges
    # Reject an ROI when corrected dx or dy exceeds this many standard deviations from the mean.
    "outlier_standard_deviation": 2.0,

    # Optional homography-method settings.
    "homography_device": "cpu",  # set "gpu" after installing matching CuPy
    "homography_gpu_device_id": 0,
    "homography_gpu_workers": 4,
    "homography_margin_fraction": 0.08,
    "homography_max_iterations": 250,  # upper limit; fitting stops on convergence

    # Optional grain segmentation.
    "detect_grains": True,
    "grain_threshold_degrees": 5.0,
    "grain_min_size": 5,
    "grain_symmetry": "cubic",  # use "none" only if crystal symmetry is unknown

    # Optional raw-pattern correction; used only for pattern_type="unprocessed".
    # "static_lmsd" uses the H5OINA static background when present; BCF/TFS have
    # no standalone static background and use the documented dynamic fallback. Other choices:
    # "dynamic_lmsd", "divide_gaussian", "subtract_gaussian", or "none".
    "unprocessed_background_mode": "static_lmsd",
    # Used only by divide_gaussian/subtract_gaussian; ignored by LMSD modes.
    "unprocessed_background_sigma_pixels": 64.0,  # absolute raw-pattern pixels
    "unprocessed_background_downsample": 8,  # estimate background on every 8th pixel
    # Used only by static_lmsd; both are fractions of the original detector width.
    "unprocessed_static_sigma_factor": 0.02,  # Gaussian sigma = width * factor
    "unprocessed_lmsd_factor": 0.183,  # LMSD radius = int(width * factor)
    "unprocessed_preprocess_device": "cpu",  # set "gpu" after installing matching CuPy

    # Dynamic-LMSD settings used by processed BCF and raw H5OINA fallback.
    "bcf_preprocess_device": "cpu",  # "cpu" or "gpu" for processed BCF dynamic-LMSD correction
    "dynamic_lmsd_sigma_factor": 0.047,  # Gaussian sigma / detector width
    "dynamic_lmsd_radius_factor": 0.0375,  # LMSD radius / detector width
    "dynamic_lmsd_edge_mode": "truncate",  # "truncate", "reflect", or "nearest"
    "dynamic_lmsd_clip_percentile": 0.75,

}
# -------------------------------------------------------------------------------


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def _input_path(config):
    value = config.get("input_file", config.get("h5oina_file"))
    if value is None:
        raise ValueError("set input_file to a supported .h5oina, .oh5, .ang, .up2, .tfs.hdf5, or .bcf dataset")
    return _path(value)


def _open_reader(path, config):
    pattern_type = config.get("pattern_type", config.get("h5_pattern_type", "processed"))
    if path.suffix.lower() == ".bcf":
        device_id = (config.get("roi_gpu_device_id", 0)
                     if config.get("analysis_method", "roi") == "roi" else
                     config.get("homography_gpu_device_id", 0))
        return BCFReader(
            path, pattern_type,
            processing_device=config.get("bcf_preprocess_device", "cpu"),
            gpu_device_id=device_id,
            lmsd_sigma_factor=config.get(
                "dynamic_lmsd_sigma_factor", config.get("bcf_lmsd_sigma_factor", 0.047)),
            lmsd_radius_factor=config.get(
                "dynamic_lmsd_radius_factor", config.get("bcf_lmsd_radius_factor", 0.0375)),
            lmsd_edge_mode=config.get(
                "dynamic_lmsd_edge_mode", config.get("bcf_lmsd_edge_mode", "truncate")),
            lmsd_clip_percentile=config.get(
                "dynamic_lmsd_clip_percentile",
                config.get("bcf_lmsd_clip_percentile", 0.75)),
        )
    if path.suffix.lower() in (".oh5", ".ang", ".up2"):
        device_id = (config.get("roi_gpu_device_id", 0)
                     if config.get("analysis_method", "roi") == "roi" else
                     config.get("homography_gpu_device_id", 0))
        return open_edax(
            path, pattern_type,
            ang_path=config.get("edax_ang_file"),
            processing_device=config.get("edax_preprocess_device", "cpu"),
            gpu_device_id=device_id,
            lmsd_sigma_factor=config.get("dynamic_lmsd_sigma_factor", 0.047),
            lmsd_radius_factor=config.get("dynamic_lmsd_radius_factor", 0.0375),
            lmsd_edge_mode=config.get("dynamic_lmsd_edge_mode", "truncate"),
            lmsd_clip_percentile=config.get("dynamic_lmsd_clip_percentile", 0.75))
    if path.name.lower().endswith(".tfs.hdf5"):
        return TFSReader(path, pattern_type)
    if path.suffix.lower() in (".h5oina", ".h5", ".hdf5"):
        return H5OINAReader(path, config.get("h5_scan_group"), pattern_type,
                            pc_source=config.get("h5_pc_source", "ebsd"))
    raise ValueError("input_file must have a .h5oina, .h5, .hdf5, .oh5, .ang, .up2, .tfs.hdf5, or .bcf extension")


def _reference_index(reader, config):
    point = config.get("reference_map_point")
    if point is not None:
        index = reader.map_index(*point)
    elif config.get("reference_index") is not None:
        index = int(config["reference_index"])
    else:
        available = reader.available_indices()
        if not len(available):
            raise ValueError("input contains no stored patterns")
        index = int(available[0])
    if not reader.has_pattern(index):
        raise ValueError(
            f"reference map index {index} has no stored pattern; "
            "set reference_map_point to an acquired point")
    return index


def run_beam_shift_calibration(config):
    """Measure effective pixel size on a separate strain-free scan."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = _input_path(config)
    output = _path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite calibration: {output}")
    with _open_reader(path, config) as reader:
        reference_index = _reference_index(reader, config)
        column, row = reference_index % reader.x_cells, reference_index // reader.x_cells
        method = config.get("calibration_method", "roi")
        common = {
            "spacing": int(config.get("calibration_line_spacing", 1)),
            "extent_um": config.get("calibration_line_extent_um"),
        }
        if method == "roi":
            report = measure_effective_pixel_size(
                reader, reference_index,
                roi_size_percent=config.get("roi_size_percent", 25.0), **common)
        elif method == "homography":
            report = measure_effective_pixel_size_homography(
                reader, reference_index,
                margin_fraction=config.get("homography_margin_fraction", 0.08),
                max_iterations=config.get("homography_max_iterations", 250),
                device=config.get("homography_device", "cpu"),
                gpu_device_id=config.get("homography_gpu_device_id", 0), **common)
        else:
            raise ValueError("calibration_method must be 'roi' or 'homography'")
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "beam_shift_calibration.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    fit = report["line_fit"]
    relative = np.asarray(fit["coordinates_map_steps"], dtype=float)
    shifts = np.asarray(fit["measured_shifts_pixels"], dtype=float)
    inlier = np.asarray(fit["inlier_mask"], dtype=bool)
    csv_path = output / "beam_shift_line.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("map_column", "map_row", "distance_um",
                         "shift_x_pixels", "shift_y_pixels", "fit_inlier"))
        for distance, (dx, dy), good in zip(relative, shifts, inlier):
            writer.writerow((int(column + distance), row,
                             distance * report["x_step_um"], dx, dy, bool(good)))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    distance_um = relative * report["x_step_um"]
    fitted = (fit["intercept_pixels"][0] +
              fit["slope_pixels_per_map_step"][0] * relative)
    axes[0].plot(distance_um, shifts[:, 0], "o", ms=4, label="measured X shift")
    axes[0].plot(distance_um[inlier], shifts[inlier, 0], "o", ms=4,
                 label="fit inliers")
    axes[0].plot(distance_um, fitted, "-", label="linear fit")
    axes[0].set_ylabel("Detector X shift (pixels)")
    axes[0].legend()
    axes[1].plot(distance_um, shifts[:, 0] - fitted, "o-", ms=4, label="X residual")
    axes[1].plot(distance_um, shifts[:, 1], "o-", ms=4, label="Y shift")
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_xlabel("Beam displacement along scan X (µm)")
    axes[1].set_ylabel("Detector shift (pixels)")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.suptitle(f"Beam-shift calibration: {report['effective_pixel_size_um_per_pixel']:.3f} µm/pixel")
    fig.tight_layout()
    png_path = output / "beam_shift_calibration.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"Effective pixel size: {report['effective_pixel_size_um_per_pixel']:.6f} "
          f"µm/pixel; detector-X sign {report['detector_x_shift_sign']:+d}")
    print(f"X fit RMS (all points): {report['x_fit_rms_all_pixels']:.4f} px; "
          f"Y alignment maximum: {report['max_abs_y_shift_pixels']:.4f} px")
    print(f"Calibration: {json_path}; plot: {png_path}")
    return report


def _pattern_config(config, pattern_shape):
    config = dict(config)
    binning = int(config.get("pattern_binning", 1))
    if binning < 1 or pattern_shape[0] % binning or pattern_shape[1] % binning:
        raise ValueError("pattern_binning must be a positive divisor of both pattern dimensions")
    if config.get("roi_size") is None:
        config["roi_size"] = roi_size_from_percent(
            (pattern_shape[0] // binning, pattern_shape[1] // binning),
            config.get("roi_size_percent", 25.0))
    return config


def _bin_pattern(pattern, factor):
    if factor == 1:
        return pattern
    height, width = pattern.shape
    if height % factor or width % factor:
        raise ValueError("pattern_binning must divide the cropped pattern dimensions")
    return np.asarray(pattern, dtype=np.float64).reshape(
        height // factor, factor, width // factor, factor).mean(axis=(1, 3))


def _binned_pc(pc, factor, original_side):
    if factor == 1:
        return pc
    offset = (factor - 1) / (2 * original_side)
    return (pc[0] + offset, pc[1] - offset, pc[2])


def _pc_for_analysis(pc, original_side, binning, calibration_side=None):
    """Express an H5 PC at the analyzed pattern resolution.

    The H5 PC can have been calibrated on a pattern resolution different from
    the stored image. A half-pixel origin conversion is needed only when that
    calibration resolution differs from the analyzed resolution.
    """
    if calibration_side is None:
        calibration_side = original_side
    calibration_side = int(calibration_side)
    analyzed_side = original_side // binning
    if calibration_side < 1 or analyzed_side < 1:
        raise ValueError("PC calibration and analysis pattern sides must be positive")
    offset = 0.5 / analyzed_side - 0.5 / calibration_side
    return (pc[0] + offset, pc[1] - offset, pc[2])


def _save_result(result, output_dir, name, source, material):
    if hasattr(result, "shifts"):
        with (output_dir / f"{name}_rois.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("x", "y", "dx", "dy", "coefficient", "peak_confidence", "used"))
            for row, used in zip(result.shifts, result.keep_mask):
                writer.writerow((row.x, row.y, row.dx, row.dy, row.coefficient,
                                 row.peak_confidence, bool(used)))
        if result.first_pass_shifts is not None:
            with (output_dir / f"{name}_rois_first_pass.csv").open(
                    "w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("x", "y", "dx", "dy", "coefficient", "peak_confidence"))
                for row in result.first_pass_shifts:
                    writer.writerow((row.x, row.y, row.dx, row.dy, row.coefficient,
                                     row.peak_confidence))
    summary = {
        **source,
        "material": material.name,
        "deformation_gradient_crystal": result.deformation.tolist(),
        "elastic_strain_crystal": result.strain.tolist(),
        "stress_gpa_crystal": result.stress_gpa.tolist(),
        "rotation_sample_mrad": sample_rotation_vector_mrad(
            result.deformation, result.orientation).tolist(),
    }
    if hasattr(result, "homography"):
        summary.update(homography_reference_to_scan=result.homography.tolist(),
                       znssd_rms=result.znssd_rms, iterations=result.iterations,
                       converged=result.converged)
    else:
        summary.update(rms_shift_error_pixels=result.rms_shift_error,
                       used_rois=result.used_rois, total_rois=result.total_rois,
                       roi_remapping=result.first_pass_shifts is not None,
                       first_pass_rms_shift_error_pixels=result.first_pass_rms_shift_error,
                       remapping_rotation_mrad=result.remapping_rotation_mrad,
                       residual_rotation_mrad=result.residual_rotation_mrad)
    with (output_dir / f"{name}_analysis.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    if hasattr(result, "homography"):
        print(f"{name}: homography ZNSSD RMS {result.znssd_rms:.4g}, "
              f"{result.iterations} iterations, converged={result.converged}")
    else:
        print(f"{name}: {result.used_rois}/{result.total_rois} ROIs, "
              f"RMS shift error {result.rms_shift_error:.4g} px")


def _analyze(reference, scan, reference_pc, scan_pc, orientation, material, config,
             prepared=None):
    method = config.get("analysis_method", "roi")
    if method == "homography":
        return analyze_homography(
            reference, scan, reference_pc, scan_pc, orientation,
            np.deg2rad(config["sample_tilt_degrees"]),
            np.deg2rad(config["camera_elevation_degrees"]), material,
            prepared=prepared,
            margin_fraction=config.get("homography_margin_fraction", 0.08),
            max_iterations=config.get("homography_max_iterations", 40),
            phosphor_to_sample=config.get("phosphor_to_sample"),
            device=config.get("homography_device", "cpu"),
            gpu_device_id=config.get("homography_gpu_device_id", 0))
    if method != "roi":
        raise ValueError("analysis_method must be 'roi' or 'homography'")
    return analyze_pair(
        reference, scan,
        roi_size=config["roi_size"], roi_count=config["roi_count"],
        roi_layout=config.get("roi_layout", "grid"),
        reference_pc=reference_pc, scan_pc=scan_pc,
        orientation=orientation,
        sample_tilt=np.deg2rad(config["sample_tilt_degrees"]),
        camera_elevation=np.deg2rad(config["camera_elevation_degrees"]),
        material=material,
        standard_deviation=config["outlier_standard_deviation"],
        roi_filter=config["roi_filter"],
        prepared=prepared,
        phosphor_to_sample=config.get("phosphor_to_sample"),
        device=config.get("roi_device", "cpu"),
        gpu_device_id=config.get("roi_gpu_device_id", 0),
        subpixel_method="quadratic_2d",
        remapping=config.get("roi_remapping", False),
    )


def _summary_fields():
    fields = ["scan_index", "reference_index", "column", "row", "grain_id",
              "pc_source", "pc_mode", "analysis_method", "roi_device",
              "homography_device",
              "status", "error",
              "rms_shift_error_pixels", "used_rois", "total_rois",
              "roi_remapping", "first_pass_rms_shift_error_pixels",
              "remapping_rotation_mrad", "residual_rotation_mrad",
              "znssd_rms", "homography_iterations", "homography_converged",
              "pattern_center_x", "pattern_center_y", "pattern_center_z"]
    fields.extend(f"H_{i}{j}" for i in range(1, 4) for j in range(1, 4))
    for prefix in ("F", "strain", "stress_gpa", "strain_sample",
                   "stress_sample_gpa", "deviatoric_strain_sample"):
        fields.extend(f"{prefix}_{i}{j}" for i in range(1, 4) for j in range(1, 4))
    fields.extend(f"rotation_sample_{i}_mrad" for i in range(1, 4))
    return fields


_H5_WORKER = None


def _h5_pc(reader, index, config, pc_plane):
    if pc_plane is not None:
        pc = pc_plane.at(index, reader.x_cells)
    elif config.get("pc_mode", "array") == "single":
        pc = config.get("_single_pc") or reader.pattern_center(index)
    else:
        pc = reader.pattern_center(index)
    pc = pc or config.get("pattern_center_fallback")
    return (_pc_for_analysis(
        pc, min(reader.height, reader.width), int(config.get("pattern_binning", 1)),
        config.get("h5_pc_calibration_pattern_side")) if pc is not None else None)


def _h5_pattern(reader, index, config, static_background=None):
    mode = config.get("unprocessed_background_mode", "static_lmsd")
    if reader.pattern_type == "unprocessed" and mode in ("static_lmsd", "dynamic_lmsd"):
        device = config.get("unprocessed_preprocess_device", "cpu")
        gpu_device_id = (config.get("roi_gpu_device_id", 0)
                         if config.get("analysis_method", "roi") == "roi" else
                         config.get("homography_gpu_device_id", 0))
        if mode == "dynamic_lmsd":
            corrected = correct_pattern_dynamic_lmsd(
                reader.uncropped_pattern(index),
                config.get("dynamic_lmsd_sigma_factor",
                           config.get("bcf_lmsd_sigma_factor", 0.047)),
                config.get("dynamic_lmsd_radius_factor",
                           config.get("bcf_lmsd_radius_factor", 0.0375)),
                config.get("dynamic_lmsd_edge_mode",
                           config.get("bcf_lmsd_edge_mode", "truncate")),
                config.get("dynamic_lmsd_clip_percentile",
                           config.get("bcf_lmsd_clip_percentile", 0.75)),
                device, gpu_device_id)
        elif static_background is None:
            raise ValueError("static_lmsd requires a static background or resolved dynamic fallback")
        elif device == "gpu":
            from pyhrebsd.preprocess_gpu import correct_pattern_static_lmsd_gpu
            corrected = correct_pattern_static_lmsd_gpu(
                reader.uncropped_pattern(index), static_background,
                sigma_factor=config.get("unprocessed_static_sigma_factor", 0.02),
                lmsd_factor=config.get("unprocessed_lmsd_factor", 0.183),
                gpu_device_id=gpu_device_id)
        elif device == "cpu":
            corrected = correct_pattern_static_lmsd(
                reader.uncropped_pattern(index), static_background,
                sigma_factor=config.get("unprocessed_static_sigma_factor", 0.02),
                lmsd_factor=config.get("unprocessed_lmsd_factor", 0.183))
        else:
            raise ValueError("unprocessed_preprocess_device must be 'cpu' or 'gpu'")
        height, width = corrected.shape
        side = min(height, width)
        top, left = (height-side)//2, (width-side)//2
        return _bin_pattern(corrected[top:top+side, left:left+side].astype(np.float64),
                            int(config.get("pattern_binning", 1)))
    pattern = reader.pattern(index)
    if reader.pattern_type == "unprocessed":
        pattern = correct_pattern_background(
            pattern, mode=mode,
            sigma_pixels=config.get("unprocessed_background_sigma_pixels", 64.0),
            downsample=config.get("unprocessed_background_downsample", 8))
    return _bin_pattern(pattern, int(config.get("pattern_binning", 1)))


def _make_h5_context(reader, reference_index, material, config, pc_plane=None):
    static_background = None
    if (reader.pattern_type == "unprocessed" and
            config.get("unprocessed_background_mode", "static_lmsd") == "static_lmsd"):
        try:
            static_background = reader.unprocessed_static_background()
        except KeyError:
            config = dict(config)
            config["unprocessed_background_mode"] = "dynamic_lmsd"
    reference = _h5_pattern(reader, reference_index, config, static_background)
    config = _pattern_config(config, reference.shape)
    orientation = (reader.orientation(reference_index)
                   if config["reference_euler_degrees"] is None
                   else np.deg2rad(config["reference_euler_degrees"]))
    reference_pc = _h5_pc(reader, reference_index, config, pc_plane)
    if reference_pc is None:
        raise ValueError("reference pattern center missing; set pattern_center_fallback")
    prepared = (prepare_homography(reference,
                                  margin_fraction=config.get("homography_margin_fraction", 0.08),
                                  device=config.get("homography_device", "cpu"),
                                  gpu_device_id=config.get("homography_gpu_device_id", 0))
                if config.get("analysis_method", "roi") == "homography" else
                prepare_analysis(reference, config["roi_size"], config["roi_count"],
                                 config["roi_filter"], config.get("roi_layout", "grid"),
                                 config.get("roi_device", "cpu"),
                                 config.get("roi_gpu_device_id", 0)))
    return {"reader": reader, "reference": reference, "reference_pc": reference_pc,
            "static_background": static_background,
            "pc_plane": pc_plane,
            "orientation": orientation, "prepared": prepared, "material": material,
            "config": config}


def _init_h5_worker(path, reference_index, material, config, pc_plane):
    global _H5_WORKER
    reader = _open_reader(Path(path), config)
    _H5_WORKER = _make_h5_context(reader, reference_index, material, config, pc_plane)


def _analyze_h5_index(index):
    context = _H5_WORKER
    reader = context["reader"]
    try:
        scan = _h5_pattern(reader, index, context["config"],
                           context["static_background"])
        scan_pc = _h5_pc(reader, index, context["config"], context["pc_plane"])
        if scan_pc is None:
            raise ValueError("pattern center missing; set pattern_center_fallback")
        result = _analyze(context["reference"], scan, context["reference_pc"], scan_pc,
                          context["orientation"], context["material"], context["config"],
                          context["prepared"])
        return index, result, scan_pc, None
    except (ValueError, np.linalg.LinAlgError) as exc:
        return index, None, None, str(exc)


def run(config=CONFIG):
    run_mode = config.get("run_mode", "analysis")
    if run_mode == "calibration":
        return run_beam_shift_calibration(config)
    if run_mode != "analysis":
        raise ValueError("run_mode must be 'analysis' or 'calibration'")
    if config.get("material_database"):
        material = Material.from_hdf5(_path(config["material_database"]),
                                      config.get("material_name", "silicon"))
    elif config.get("material_file"):  # compatibility with older configurations
        material = Material.from_openxy_file(_path(config["material_file"]))
    else:
        raise ValueError("set material_database and material_name")
    configured_euler = config["reference_euler_degrees"]
    orientation = None if configured_euler is None else np.deg2rad(configured_euler)
    output_dir = _path(config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite analysis results: {output_dir}")
    mode = config.get("input_mode",
                      "dataset" if ("input_file" in config or "h5oina_file" in config)
                      else "images")
    if mode in ("dataset", "h5oina", "bcf"):
        path = _input_path(config)
        with _open_reader(path, config) as reader:
            if (reader.pattern_type == "unprocessed" and
                    config.get("unprocessed_background_mode", "static_lmsd") == "static_lmsd" and
                    not reader.has_unprocessed_static_background()):
                config = dict(config)
                config["unprocessed_background_mode"] = "dynamic_lmsd"
                print("Static background unavailable; using per-pattern dynamic LMSD")
            side = min(reader.height, reader.width)
            config = _pattern_config(config, (side, side))
            if config.get("analysis_method", "roi") == "homography":
                print(f"Homography DIC ({config.get('homography_device', 'cpu')}): "
                      f"{config.get('pattern_binning', 1)}x binning, every pixel "
                      f"on {side // config.get('pattern_binning', 1)} px square pattern")
            else:
                print(f"ROIs ({config.get('roi_device', 'cpu')}): "
                      f"{config.get('roi_layout', 'grid')}, "
                      f"{config['roi_count']} requested, {config['roi_size']} px "
                      f"on {side // config.get('pattern_binning', 1)} px square pattern")
            if config["sample_tilt_degrees"] is None:
                config["sample_tilt_degrees"] = reader.sample_tilt_degrees()
            if config["camera_elevation_degrees"] is None:
                config["camera_elevation_degrees"] = reader.camera_elevation_degrees()
            if config["sample_tilt_degrees"] is None or config["camera_elevation_degrees"] is None:
                raise ValueError("dataset geometry is missing; set sample_tilt_degrees and "
                                 "camera_elevation_degrees explicitly")
            print(f"Geometry: sample tilt {config['sample_tilt_degrees']:.6f} degrees, "
                  f"camera elevation {config['camera_elevation_degrees']:.6f} degrees")
            geometry_mode = config.get("detector_geometry", "elevation")
            if geometry_mode == "full":
                config["phosphor_to_sample"] = reader.phosphor_to_sample(
                    config["sample_tilt_degrees"])
                if config["phosphor_to_sample"] is None:
                    raise ValueError("full detector geometry needs sample tilt and "
                                     "detector orientation metadata")
            elif geometry_mode == "elevation":
                config["phosphor_to_sample"] = None
            else:
                raise ValueError("detector_geometry must be 'full' or 'elevation'")
            pc_mode = config.get("pc_mode", "h5")
            if pc_mode not in ("single", "array", "h5", "affine", "beam_shift_eps"):
                raise ValueError("pc_mode must be 'single', 'array', 'affine', or 'beam_shift_eps'")
            print(f"PC source: {reader.pc_source_path}; mode: {pc_mode}")
            reference_index = _reference_index(reader, config)
            if pc_mode == "single":
                config = dict(config)
                config["_single_pc"] = reader.pattern_center(reference_index)
            if pc_mode == "affine":
                pc_plane = fit_pc_plane(reader)
            elif pc_mode == "beam_shift_eps":
                effective_pixel_size = config.get("beam_shift_effective_pixel_size_um")
                if effective_pixel_size is None:
                    raise ValueError("set beam_shift_effective_pixel_size_um from "
                                     "the separate calibration result")
                pc_plane = pc_plane_from_effective_pixel_size(
                    reader, reference_index, effective_pixel_size,
                    config.get("beam_shift_detector_x_sign", "auto"))
            else:
                pc_plane = None
            reference_pc = _h5_pc(reader, reference_index, config, pc_plane)
            if reference_pc is None:
                raise ValueError("reference pattern center missing; set pattern_center_fallback")
            indices = config.get("scan_indices")
            if indices is None:
                indices = reader.available_indices()
            indices = list(indices)
            missing = [int(index) for index in indices if not reader.has_pattern(index)]
            if missing:
                preview = ", ".join(map(str, missing[:5]))
                suffix = "..." if len(missing) > 5 else ""
                raise ValueError(f"scan indices without stored patterns: {preview}{suffix}")
            method = config.get("analysis_method", "roi")
            if method == "homography" and config.get("homography_device", "cpu") == "gpu":
                workers = int(config.get("homography_gpu_workers", 1))
            elif method == "roi" and config.get("roi_device", "cpu") == "gpu":
                workers = int(config.get("roi_gpu_workers", 1))
            else:
                workers = int(config.get("workers", 1))
            if workers < 1:
                raise ValueError("workers must be at least 1")
            output_dir.mkdir(parents=True, exist_ok=True)
            grain_labels = None
            if config.get("detect_grains", False):
                from segment_grains import detect_from_reader
                grains = detect_from_reader(
                    reader, output_dir, config.get("grain_threshold_degrees", 5.0),
                    config.get("grain_min_size", 5), config.get("grain_symmetry", "cubic"))
                grain_labels = grains.labels.ravel()
            if pc_plane is not None:
                report_path = output_dir / "pc_calibration.json"
                with report_path.open("w", encoding="utf-8") as stream:
                    json.dump(pc_plane.report(), stream, indent=2)
                print(f"PC plane: {pc_plane.points_used}/{pc_plane.points_total} points; "
                      f"RMS residual {pc_plane.rms_residual_pixels} pixels; {report_path}")
            summary_path = output_dir / "scan_results.csv"
            if workers == 1:
                global _H5_WORKER
                _H5_WORKER = _make_h5_context(reader, reference_index, material, config, pc_plane)
                results = map(_analyze_h5_index, indices)
                executor = None
            else:
                executor = ProcessPoolExecutor(
                    max_workers=workers, initializer=_init_h5_worker,
                    initargs=(str(path), reference_index, material, config, pc_plane),
                )
                results = executor.map(_analyze_h5_index, indices, chunksize=4)
            try:
                with summary_path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=_summary_fields())
                    writer.writeheader()
                    failures = 0
                    for count, (index, result, scan_pc, error) in enumerate(results, start=1):
                        row = {"scan_index": int(index), "reference_index": reference_index,
                               "column": index % reader.x_cells if reader.x_cells else "",
                               "row": index // reader.x_cells if reader.x_cells else "",
                               "grain_id": int(grain_labels[index]) if grain_labels is not None else "",
                               "pc_source": reader.pc_source,
                               "pc_mode": pc_mode,
                               "analysis_method": config.get("analysis_method", "roi"),
                               "roi_device": (config.get("roi_device", "cpu")
                                              if config.get("analysis_method", "roi")
                                              == "roi" else ""),
                               "homography_device": (config.get("homography_device", "cpu")
                                                     if config.get("analysis_method", "roi")
                                                     == "homography" else ""),
                               "status": "error" if error else "ok", "error": error or ""}
                        if result is not None:
                            if hasattr(result, "homography"):
                                row.update({"znssd_rms": result.znssd_rms,
                                            "homography_iterations": result.iterations,
                                            "homography_converged": result.converged})
                                row.update({f"H_{i+1}{j+1}": float(result.homography[i, j])
                                            for i in range(3) for j in range(3)})
                            else:
                                row.update({"rms_shift_error_pixels": result.rms_shift_error,
                                            "used_rois": result.used_rois,
                                            "total_rois": result.total_rois,
                                            "roi_remapping": result.first_pass_shifts is not None,
                                            "first_pass_rms_shift_error_pixels": result.first_pass_rms_shift_error,
                                            "remapping_rotation_mrad": result.remapping_rotation_mrad,
                                            "residual_rotation_mrad": result.residual_rotation_mrad})
                            row.update({"pattern_center_x": scan_pc[0],
                                        "pattern_center_y": scan_pc[1],
                                        "pattern_center_z": scan_pc[2]})
                            for prefix, matrix in (("F", result.deformation),
                                                   ("strain", result.strain),
                                                   ("stress_gpa", result.stress_gpa),
                                                   ("strain_sample", result.orientation.T @ result.strain @ result.orientation),
                                                   ("stress_sample_gpa", result.orientation.T @ result.stress_gpa @ result.orientation),
                                                   ("deviatoric_strain_sample", 
                                                    (result.orientation.T @ result.strain @ result.orientation)
                                                    - np.eye(3) * np.trace(result.strain) / 3)):
                                row.update({f"{prefix}_{i+1}{j+1}": float(matrix[i, j])
                                            for i in range(3) for j in range(3)})
                            row.update({f"rotation_sample_{i+1}_mrad": float(value)
                                        for i, value in enumerate(sample_rotation_vector_mrad(
                                            result.deformation, result.orientation))})
                            if config.get("save_per_pattern_files", False):
                                name = f"scan_{index:06d}"
                                source = {"input_file": str(path), "scan_group": reader.scan_group,
                                          "pattern_type": reader.pattern_type,
                                          "pc_source": reader.pc_source,
                                          "pc_mode": pc_mode,
                                          "reference_index": reference_index, "scan_index": int(index),
                                          "reference_pattern_center": reference_pc,
                                          "scan_pattern_center": scan_pc}
                                _save_result(result, output_dir, name, source, material)
                        else:
                            failures += 1
                        writer.writerow(row)
                        if count % 100 == 0 or count == len(indices):
                            stream.flush()
                            print(f"Processed {count}/{len(indices)} patterns; "
                                  f"{failures} failed; summary: {summary_path}")
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)
    elif mode == "images":
        if int(config.get("pattern_binning", 1)) != 1:
            raise ValueError("pattern_binning currently requires a dataset input")
        if orientation is None:
            raise ValueError("reference_euler_degrees is required in images mode")
        reference_path = _path(config["reference_image"])
        reference = read_pattern(reference_path)
        config = _pattern_config(config, reference.shape)
        prepared = (prepare_homography(reference,
                                      margin_fraction=config.get("homography_margin_fraction", 0.08),
                                      device=config.get("homography_device", "cpu"),
                                      gpu_device_id=config.get("homography_gpu_device_id", 0))
                    if config.get("analysis_method", "roi") == "homography" else
                    prepare_analysis(reference, config["roi_size"], config["roi_count"],
                                     config["roi_filter"], config.get("roi_layout", "grid"),
                                     config.get("roi_device", "cpu"),
                                     config.get("roi_gpu_device_id", 0)))
        output_dir.mkdir(parents=True, exist_ok=True)
        for number, item in enumerate(config["scan_images"], start=1):
            scan_path = _path(item["path"])
            result = _analyze(reference, read_pattern(scan_path),
                              config["reference_pattern_center"],
                              item.get("pattern_center", config["reference_pattern_center"]),
                              orientation, material, config, prepared)
            name = f"{number:04d}_{scan_path.stem}"
            _save_result(result, output_dir, name,
                         {"reference_image": str(reference_path), "scan_image": str(scan_path)},
                         material)
    else:
        raise ValueError("input_mode must be 'dataset' or 'images'")


if __name__ == "__main__":
    run()
