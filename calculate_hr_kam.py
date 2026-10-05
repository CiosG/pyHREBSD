"""Calculate KAM from HR-EBSD lattice rotations in PyHREBSD scan CSVs."""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from pyhrebsd.kam import hr_kam


# -------------------------- CONFIG: edit these values --------------------------
CONFIG = {
    "scan_csv_files": [
        "validation_roi_cpu/scan_results.csv",
        "validation_roi_gpu/scan_results.csv",
        "validation_homography_cpu_serial/scan_results.csv",
        "validation_homography_gpu_serial/scan_results.csv",
    ],
    "output_dir": "hr_kam_maps_full",
    "neighbor_misorientation_limit_degrees": 5.0,
    "roi_rms_limit_pixels": 2.0,
    "homography_znssd_limit": 0.5,
    "scan_step_um": 1.0,  # /1/EBSD/Header/X Step and Y Step in this H5OINA file
    "color_percentile": 99.5,
    "log_color_floor_degrees": 0.002,
}
# -------------------------------------------------------------------------------

DISPLAY_NAMES = {
    "validation_roi_cpu": "Standard CPU",
    "validation_roi_gpu": "Standard GPU",
    "validation_homography_cpu_serial": "Homography CPU",
    "validation_homography_gpu_serial": "Homography GPU",
}


def load_rotations(path: Path, config: dict) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} has no rows")
    height = max(int(row["row"]) for row in rows) + 1
    width = max(int(row["column"]) for row in rows) + 1
    shape = (height, width)
    if len(rows) != height * width:
        raise ValueError(f"{path} is not a complete rectangular scan")
    vectors = np.full(shape + (3,), np.nan)
    grains = np.zeros(shape, dtype=np.int32)
    valid = np.zeros(shape, dtype=bool)
    indexed = np.zeros(shape, dtype=bool)
    homography = rows[0]["analysis_method"] == "homography"
    quality_field = "znssd_rms" if homography else "rms_shift_error_pixels"
    quality_limit = (config["homography_znssd_limit"] if homography else
                     config["roi_rms_limit_pixels"])
    for row in rows:
        y, x = int(row["row"]), int(row["column"])
        if indexed[y, x]:
            raise ValueError(f"{path} has duplicate map point ({x}, {y})")
        indexed[y, x] = True
        grains[y, x] = int(row["grain_id"] or 0)
        if row["status"] != "ok":
            continue
        vector = np.array([float(row[f"rotation_sample_{i}_mrad"])
                           for i in range(1, 4)])
        vectors[y, x] = vector
        quality = float(row[quality_field])
        valid[y, x] = (grains[y, x] > 0 and np.isfinite(vector).all() and
                       np.isfinite(quality) and quality <= quality_limit and
                       (not homography or row["homography_converged"] == "True"))
    if not indexed.all():
        raise ValueError(f"{path} has missing map points")
    raw_valid = (grains > 0) & np.isfinite(vectors).all(axis=2)
    return {"path": path, "vectors": vectors, "grains": grains,
            "valid": valid, "raw_valid": raw_valid,
            "quality_field": quality_field, "quality_limit": quality_limit}


def _statistics(kam: np.ndarray, counts: np.ndarray) -> dict:
    finite = kam[np.isfinite(kam)]
    return {"points_with_kam": int(finite.size),
            "points_without_neighbors": int(np.count_nonzero(counts == 0)),
            "median_degrees": float(np.median(finite)) if finite.size else None,
            "p99_degrees": float(np.percentile(finite, 99)) if finite.size else None,
            "maximum_degrees": float(np.max(finite)) if finite.size else None}


def _plot_map(kam: np.ndarray, title: str, output: Path, vmax: float,
              step_um: float, note: str) -> None:
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(10.5, 7.4), constrained_layout=True)
    image = ax.imshow(kam, cmap=cmap, vmin=0, vmax=vmax,
                      origin="upper", interpolation="nearest")
    ax.set(xlabel="Map column", ylabel="Map row",
           title=f"{title}\n{note}; color 0–{vmax:.3g}°; white = no eligible neighbours")
    fig.colorbar(image, ax=ax, label="HR-EBSD KAM (degrees)", shrink=0.85)
    bar_pixels = 20 / step_um
    x0, y0 = 8, kam.shape[0] - 7
    ax.plot([x0, x0 + bar_pixels], [y0, y0], color="white", linewidth=4)
    ax.plot([x0, x0 + bar_pixels], [y0, y0], color="black", linewidth=2)
    ax.text(x0 + bar_pixels / 2, y0 - 3, "20 µm", ha="center", va="bottom",
            color="black", fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_comparison(inputs: list[dict], field: str, vmax: float,
                     output: Path, threshold: float, percentile: float,
                     log_floor: float | None = None) -> None:
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, item in zip(axes.flat, inputs):
        if log_floor is None:
            image = ax.imshow(item[field], cmap=cmap, vmin=0, vmax=vmax,
                              origin="upper", interpolation="nearest")
        else:
            # Keep zero KAM on the darkest color; only missing points are white.
            values = np.where(np.isfinite(item[field]),
                              np.maximum(item[field], log_floor), np.nan)
            image = ax.imshow(values, cmap=cmap,
                              norm=LogNorm(vmin=log_floor, vmax=vmax, clip=True),
                              origin="upper", interpolation="nearest")
        ax.set_title(DISPLAY_NAMES.get(item["path"].parent.name,
                                       item["path"].parent.name))
        ax.set_xlabel("Map column")
        ax.set_ylabel("Map row")
    ticks = None
    if log_floor is not None:
        ticks = [value for value in (.002, .005, .01, .02, .05, .1, .2, .5, 1.0)
                 if log_floor <= value <= vmax]
    bar = fig.colorbar(image, ax=axes, label="HR-EBSD KAM (degrees)",
                       shrink=0.8, ticks=ticks)
    if ticks is not None:
        bar.ax.set_yticklabels([f"{value:g}" for value in ticks])
    quality = "fit-quality masked" if field == "kam" else "all fitted rotations; low-confidence values included"
    scale = (f"log color {log_floor:g}–{vmax:.3g}°" if log_floor is not None
             else f"color 0–{vmax:.3g}°")
    fig.suptitle("HR-EBSD KAM: local lattice-rotation misorientation\n"
                 f"4 direct neighbours, same grain, pair ≤ {threshold:g}°, "
                 f"{quality}; shared {scale} "
                 f"({percentile:g}th percentile)")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def run(config: dict = CONFIG, *, log_only: bool = False) -> None:
    output_root = Path(config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    inputs = [load_rotations(Path(path), config) for path in config["scan_csv_files"]]
    shapes = {item["grains"].shape for item in inputs}
    if len(shapes) != 1:
        raise ValueError("all scans must share one map grid")
    threshold = config["neighbor_misorientation_limit_degrees"]
    for item in inputs:
        item["kam"], item["neighbors"] = hr_kam(
            item["vectors"], item["valid"], item["grains"], threshold)
        item["kam_raw"], item["neighbors_raw"] = hr_kam(
            item["vectors"], item["raw_valid"], item["grains"], threshold)
    limits = [float(np.percentile(item["kam"][np.isfinite(item["kam"])],
                                  config["color_percentile"])) for item in inputs
              if np.isfinite(item["kam"]).any()]
    vmax = max(max(limits, default=0.0), 1e-12)
    raw_limits = [float(np.percentile(item["kam_raw"][np.isfinite(item["kam_raw"])],
                                      config["color_percentile"])) for item in inputs
                  if np.isfinite(item["kam_raw"]).any()]
    raw_vmax = max(max(raw_limits, default=0.0), 1e-12)
    log_floor = config["log_color_floor_degrees"]
    if not 0 < log_floor < min(vmax, raw_vmax):
        raise ValueError("log_color_floor_degrees must be positive and below both color maxima")
    if log_only:
        _plot_comparison(inputs, "kam_raw", raw_vmax,
                         output_root / "hr_kam_comparison_log.png", threshold,
                         config["color_percentile"], log_floor)
        _plot_comparison(inputs, "kam", vmax,
                         output_root / "hr_kam_comparison_quality_masked_log.png",
                         threshold, config["color_percentile"], log_floor)
        print(output_root / "hr_kam_comparison_log.png")
        print(output_root / "hr_kam_comparison_quality_masked_log.png")
        return
    for item in inputs:
        name = item["path"].parent.name
        folder = output_root / name
        folder.mkdir(parents=True, exist_ok=True)
        finite = item["kam"][np.isfinite(item["kam"])]
        local_vmax = max(float(np.percentile(finite, config["color_percentile"])),
                         1e-12) if finite.size else 1e-12
        raw_finite = item["kam_raw"][np.isfinite(item["kam_raw"])]
        raw_local_vmax = max(float(np.percentile(raw_finite, config["color_percentile"])),
                             1e-12) if raw_finite.size else 1e-12
        height, width = item["grains"].shape
        with (folder / "hr_kam.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("scan_index", "column", "row", "grain_id",
                             "quality_valid", "kam_degrees", "neighbor_count",
                             "kam_unmasked_degrees", "unmasked_neighbor_count"))
            for y in range(height):
                for x in range(width):
                    masked, raw = item["kam"][y, x], item["kam_raw"][y, x]
                    writer.writerow((y * width + x, x, y, int(item["grains"][y, x]),
                                     bool(item["valid"][y, x]),
                                     float(masked) if np.isfinite(masked) else "",
                                     int(item["neighbors"][y, x]),
                                     float(raw) if np.isfinite(raw) else "",
                                     int(item["neighbors_raw"][y, x])))
        note = (f"4 direct neighbours, same grain, pair ≤ {threshold:g}°; "
                f"{item['quality_field']} ≤ {item['quality_limit']:g}")
        display_name = DISPLAY_NAMES.get(name, name)
        _plot_map(item["kam"], f"HR-EBSD KAM — {display_name}", folder / "hr_kam.png",
                  local_vmax, config["scan_step_um"], note)
        _plot_map(item["kam_raw"], f"HR-EBSD KAM — {display_name} (unmasked)",
                  folder / "hr_kam_unmasked.png", raw_local_vmax, config["scan_step_um"],
                  f"4 direct neighbours, same grain, pair ≤ {threshold:g}°; no fit-quality mask")
        summary = {"source_csv": str(item["path"]), "rotation_source":
                   "sample-frame HR-EBSD polar rotation vectors relative to scan reference",
                   "neighbor_order": 1, "neighbor_connectivity": 4,
                   "same_grain_only": True, "pair_threshold_degrees": threshold,
                   "fit_quality": note, "scan_step_um": config["scan_step_um"],
                   "shared_color_max_degrees": vmax,
                   "individual_color_max_degrees": local_vmax,
                   "unmasked_shared_color_max_degrees": raw_vmax,
                   "unmasked_individual_color_max_degrees": raw_local_vmax,
                   "quality_masked": _statistics(item["kam"], item["neighbors"]),
                   "unmasked": _statistics(item["kam_raw"], item["neighbors_raw"])}
        with (folder / "hr_kam_summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        print(f"{name}: {summary['quality_masked']}; {folder / 'hr_kam.png'}")

    _plot_comparison(inputs, "kam_raw", raw_vmax,
                     output_root / "hr_kam_comparison.png", threshold,
                     config["color_percentile"])
    print(output_root / "hr_kam_comparison.png")
    _plot_comparison(inputs, "kam", vmax,
                     output_root / "hr_kam_comparison_quality_masked.png", threshold,
                     config["color_percentile"])
    print(output_root / "hr_kam_comparison_quality_masked.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-only", action="store_true",
                        help="render logarithmic comparisons without rewriting existing maps")
    run(log_only=parser.parse_args().log_only)
