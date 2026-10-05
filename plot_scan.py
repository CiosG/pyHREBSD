"""Plot PyHREBSD strain, stress, and fit-quality maps from scan_results.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, TwoSlopeNorm
import numpy as np

from pyhrebsd.frame_views import rotate_tensor_maps


STRAIN_FIELDS = (
    ("strain_11", r"$\epsilon_{xx}$"), ("strain_22", r"$\epsilon_{yy}$"),
    ("strain_33", r"$\epsilon_{zz}$"), ("strain_12", r"$\epsilon_{xy}$"),
    ("strain_13", r"$\epsilon_{xz}$"), ("strain_23", r"$\epsilon_{yz}$"),
)
STRESS_FIELDS = (
    ("stress_gpa_11", r"$\sigma_{xx}$"), ("stress_gpa_22", r"$\sigma_{yy}$"),
    ("stress_gpa_33", r"$\sigma_{zz}$"), ("stress_gpa_12", r"$\sigma_{xy}$"),
    ("stress_gpa_13", r"$\sigma_{xz}$"), ("stress_gpa_23", r"$\sigma_{yz}$"),
)
SAMPLE_STRAIN_FIELDS = (
    ("strain_sample_11", r"$\epsilon_{11}$"), ("strain_sample_22", r"$\epsilon_{22}$"),
    ("strain_sample_33", r"$\epsilon_{33}$"), ("strain_sample_12", r"$\epsilon_{12}$"),
    ("strain_sample_13", r"$\epsilon_{13}$"), ("strain_sample_23", r"$\epsilon_{23}$"),
)
SAMPLE_STRESS_FIELDS = (
    ("stress_sample_gpa_11", r"$\sigma_{11}$"),
    ("stress_sample_gpa_22", r"$\sigma_{22}$"),
    ("stress_sample_gpa_33", r"$\sigma_{33}$"),
    ("stress_sample_gpa_12", r"$\sigma_{12}$"),
    ("stress_sample_gpa_13", r"$\sigma_{13}$"),
    ("stress_sample_gpa_23", r"$\sigma_{23}$"),
)
SAMPLE_DEVIATORIC_FIELDS = (
    ("deviatoric_strain_sample_11", r"$\epsilon'_{11}$"),
    ("deviatoric_strain_sample_12", r"$\epsilon'_{12}$"),
    ("deviatoric_strain_sample_13", r"$\epsilon'_{13}$"),
    ("deviatoric_strain_sample_22", r"$\epsilon'_{22}$"),
    ("deviatoric_strain_sample_23", r"$\epsilon'_{23}$"),
    ("deviatoric_strain_sample_33", r"$\epsilon'_{33}$"),
)
SAMPLE_ROTATION_FIELDS = (
    ("rotation_sample_1_mrad", r"$\omega_1$"),
    ("rotation_sample_2_mrad", r"$\omega_2$"),
    ("rotation_sample_3_mrad", r"$\omega_3$"),
)


def load_maps(csv_path: Path, shape: tuple[int, int] | None = None):
    """Read complete CSV rows; leave unprocessed and failed points as NaN."""
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    parsed = []
    for row in rows:
        try:
            parsed.append((int(row["row"]), int(row["column"]), row))
        except (TypeError, ValueError):
            continue  # A scan may be writing its final CSV row.
    if not parsed:
        raise ValueError(f"{csv_path} has no complete map rows")
    if shape is None:
        shape = (max(y for y, _, _ in parsed) + 1,
                 max(x for _, x, _ in parsed) + 1)
    height, width = shape
    keys = [name for name, _ in STRAIN_FIELDS + STRESS_FIELDS]
    homography = rows[0].get("analysis_method") == "homography"
    keys += (["znssd_rms"] if homography else
             ["rms_shift_error_pixels", "used_rois"])
    if all(key in rows[0] for key, _ in SAMPLE_DEVIATORIC_FIELDS):
        keys += [key for key, _ in SAMPLE_DEVIATORIC_FIELDS]
    for fields in (SAMPLE_STRAIN_FIELDS, SAMPLE_STRESS_FIELDS,
                   SAMPLE_ROTATION_FIELDS):
        if all(key in rows[0] for key, _ in fields):
            keys += [key for key, _ in fields]
    maps = {key: np.full(shape, np.nan, dtype=np.float64) for key in keys}
    if homography:
        maps["homography_converged"] = np.full(shape, np.nan, dtype=np.float64)
    status = np.zeros(shape, dtype=np.uint8)  # 0 pending, 1 fit, 2 failed
    for y, x, row in parsed:
        if not (0 <= y < height and 0 <= x < width):
            raise ValueError(f"point ({x}, {y}) lies outside requested shape {shape}")
        if row["status"] != "ok":
            status[y, x] = 2
            continue
        try:
            values = {key: float(row[key]) for key in keys}
        except (TypeError, ValueError, KeyError):
            continue
        if not all(np.isfinite(value) for value in values.values()):
            continue
        for key, value in values.items():
            maps[key][y, x] = value
        if homography:
            maps["homography_converged"][y, x] = (
                1.0 if row.get("homography_converged") == "True" else 0.0)
        status[y, x] = 1
    return maps, status


def _cmap(name):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("#e6e6e6")
    return cmap


def _map_axis(ax, data, title, cmap, norm=None, unit=None):
    image = ax.imshow(data, origin="upper", interpolation="nearest", aspect="equal",
                      cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Map column")
    ax.set_ylabel("Map row")
    cbar = ax.figure.colorbar(image, ax=ax, shrink=0.80, pad=0.02)
    if unit:
        cbar.set_label(unit)


def _shared_signed_limit(maps, fields, factor, percentile=99.5):
    """One symmetric color limit, chosen from the widest component map."""
    if not 0 < percentile <= 100:
        raise ValueError("color percentile must be in (0, 100]")
    limits = []
    for key, _ in fields:
        values = np.abs(maps[key] * factor)
        finite = values[np.isfinite(values)]
        if len(finite):
            limits.append(float(np.percentile(finite, percentile)))
    return max(max(limits, default=0.0), 1e-12)


def _signed_figure(maps, fields, factor, unit, title, output_path, subtitle,
                   color_percentile=99.5):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    limit = _shared_signed_limit(maps, fields, factor, color_percentile)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    for ax, (key, label) in zip(axes.flat, fields):
        data = maps[key] * factor
        image = ax.imshow(data, origin="upper", interpolation="nearest", aspect="equal",
                          cmap=_cmap("RdBu_r"), norm=norm)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Map column")
        ax.set_ylabel("Map row")
    fig.colorbar(image, ax=axes, shrink=0.80, pad=0.02, label=unit)
    clipping = (f" · shared ±{limit:.3g} {unit} ({color_percentile:g}th percentile)"
                if color_percentile < 100 else f" · shared ±{limit:.3g} {unit}")
    fig.suptitle(f"{title}\n{subtitle}{clipping}", fontsize=14)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _rotation_figure(maps, output_path, subtitle, color_percentile=99.5):
    """Plot all sample rotation components on one signed mrad scale."""
    limit = _shared_signed_limit(maps, SAMPLE_ROTATION_FIELDS, 1,
                                 color_percentile)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, (key, label) in zip(axes, SAMPLE_ROTATION_FIELDS):
        image = ax.imshow(maps[key], origin="upper", interpolation="nearest",
                          aspect="equal", cmap=_cmap("RdBu_r"), norm=norm)
        ax.set_title(label)
        ax.set_xlabel("Map column")
        ax.set_ylabel("Map row")
    fig.colorbar(image, ax=axes, shrink=0.8, pad=0.02, label="mrad")
    fig.suptitle(f"Sample-frame rotation\n{subtitle} - shared +/-{limit:.3g} mrad")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_scan(csv_path: Path, output_dir: Path, shape: tuple[int, int] | None = None,
              color_percentile: float = 99.5,
              sample_axis_rotation_degrees: float = 0.0):
    maps, status = load_maps(csv_path, shape)
    complete = int(np.count_nonzero(status == 1))
    failed = int(np.count_nonzero(status == 2))
    total = int(status.size)
    subtitle = f"{complete:,}/{total:,} fitted · {failed:,} failed · gray = pending/failed"
    output_dir.mkdir(parents=True, exist_ok=True)
    strain_path = output_dir / "strain_maps.png"
    stress_path = output_dir / "stress_maps.png"
    quality_path = output_dir / "fit_quality.png"
    _signed_figure(maps, STRAIN_FIELDS, 100, "%", "Elastic strain (crystal frame)",
                   strain_path, subtitle, color_percentile)
    _signed_figure(maps, STRESS_FIELDS, 1, "GPa", "Stress (crystal frame)",
                   stress_path, subtitle, color_percentile)
    sample_axis_note = (f"; sample axes +{sample_axis_rotation_degrees:g}°"
                        if sample_axis_rotation_degrees else "")
    if all(key in maps for key, _ in SAMPLE_STRAIN_FIELDS):
        sample_strain = rotate_tensor_maps(maps, "strain_sample",
                                           sample_axis_rotation_degrees)
        _signed_figure(sample_strain, SAMPLE_STRAIN_FIELDS, 100, "%",
                       f"Elastic strain (sample frame{sample_axis_note})",
                       output_dir / "sample_strain_maps.png", subtitle,
                       color_percentile)
    if all(key in maps for key, _ in SAMPLE_STRESS_FIELDS):
        sample_stress = rotate_tensor_maps(maps, "stress_sample_gpa",
                                           sample_axis_rotation_degrees)
        _signed_figure(sample_stress, SAMPLE_STRESS_FIELDS, 1, "GPa",
                       f"Stress (sample frame{sample_axis_note})",
                       output_dir / "sample_stress_maps.png", subtitle,
                       color_percentile)
    if all(key in maps for key, _ in SAMPLE_ROTATION_FIELDS):
        _rotation_figure(maps, output_dir / "sample_rotation_maps.png",
                         subtitle, color_percentile)
    if all(key in maps for key, _ in SAMPLE_DEVIATORIC_FIELDS):
        maps = rotate_tensor_maps(maps, "deviatoric_strain_sample",
                                  sample_axis_rotation_degrees)
        axis_note = (f"; sample axes +{sample_axis_rotation_degrees:g}°"
                     if sample_axis_rotation_degrees else "")
        _signed_figure(maps, SAMPLE_DEVIATORIC_FIELDS, 1000, "mm/m",
                       f"Relative deviatoric strain (sample frame{axis_note})",
                       output_dir / "sample_deviatoric_strain_maps.png", subtitle,
                       color_percentile)
        homography = "znssd_rms" in maps
        quality_key = "znssd_rms" if homography else "rms_shift_error_pixels"
        quality_limit = 0.5 if homography else 2.0
        quality_mask = np.isfinite(maps[quality_key]) & (
            maps[quality_key] <= quality_limit)
        if homography:
            quality_mask &= maps["homography_converged"] == 1
        masked_maps = dict(maps)
        for key, _ in SAMPLE_DEVIATORIC_FIELDS:
            masked_maps[key] = np.where(quality_mask, maps[key], np.nan)
        _signed_figure(masked_maps, SAMPLE_DEVIATORIC_FIELDS, 1000, "mm/m",
                       f"Relative deviatoric strain (sample frame{axis_note}; {quality_key} <= {quality_limit:g})",
                       output_dir / "sample_deviatoric_strain_quality_masked.png",
                       f"{np.count_nonzero(quality_mask):,}/{total:,} shown | gray = poor fit",
                       color_percentile)

    s = [maps[f"stress_gpa_{i}{j}"] for i, j in ((1, 1), (2, 2), (3, 3),
                                                  (1, 2), (1, 3), (2, 3))]
    vm = np.sqrt(0.5 * ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 +
                        (s[2] - s[0]) ** 2) + 3 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2))
    homography = "znssd_rms" in maps
    rms = maps["znssd_rms" if homography else "rms_shift_error_pixels"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    vm_limit = max(float(np.nanpercentile(vm, 99.5)), 1e-12)
    rms_limit = max(float(np.nanpercentile(rms, 99.5)), 1e-12)
    _map_axis(axes[0], vm, "Von Mises stress", _cmap("magma"),
              PowerNorm(0.55, vmin=0, vmax=vm_limit), "GPa")
    _map_axis(axes[1], rms, "ZNSSD RMS" if homography else "RMS shift fit error",
              _cmap("viridis"), PowerNorm(0.55, vmin=0, vmax=rms_limit),
              None if homography else "pixels")
    if homography:
        _map_axis(axes[2], maps["homography_converged"], "Converged", _cmap("viridis"),
                  None, "1 = yes")
    else:
        _map_axis(axes[2], maps["used_rois"], "ROIs used", _cmap("cividis"),
                  None, "count")
    fig.suptitle(f"Fit quality and stress magnitude\n{subtitle} · stress/error clipped at 99.5th percentile",
                 fontsize=14)
    fig.savefig(quality_path, dpi=220)
    plt.close(fig)
    return strain_path, stress_path, quality_path, complete, failed, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path,
                        default=Path("results/scan_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--rows", type=int, default=112)
    parser.add_argument("--cols", type=int, default=164)
    parser.add_argument("--color-percentile", type=float, default=99.5,
                        help="shared signed limit from the widest map; 100 uses the full range")
    parser.add_argument("--sample-axis-rotation-degrees", type=float, default=0.0,
                        help="rotate displayed sample-frame tensor components about axis 3")
    args = parser.parse_args()
    paths = plot_scan(args.csv_path, args.output_dir, (args.rows, args.cols),
                      args.color_percentile, args.sample_axis_rotation_degrees)
    print(f"Plotted {paths[3]:,}/{paths[5]:,} points, {paths[4]:,} failed")
    for path in paths[:3]:
        print(path)


if __name__ == "__main__":
    main()
