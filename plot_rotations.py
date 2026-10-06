"""Extract and plot sample-frame lattice rotations from a PyHREBSD scan CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from pyhrebsd.frame_views import in_plane_rotation, rotate_vector_maps
from pyhrebsd.rotations import sample_rotation_vector_mrad
from run_pyhrebsd import CONFIG


def extract_rotations(csv_path: Path, h5_path: Path, output_dir: Path,
                      scan_group: str | None = None,
                      sample_axis_rotation_degrees: float = 0.0):
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("scan CSV is empty")
    homography = rows[0].get("analysis_method") == "homography"
    quality_key = "znssd_rms" if homography else "rms_shift_error_pixels"
    reference_index = int(rows[0]["reference_index"])
    width = max(int(row["column"]) for row in rows) + 1
    height = max(int(row["row"]) for row in rows) + 1
    maps = np.full((3, height, width), np.nan)
    rms = np.full((height, width), np.nan)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output = output_dir / "sample_rotations.csv"
    display_rotation = in_plane_rotation(sample_axis_rotation_degrees)
    from run_pyhrebsd import _open_reader
    reader_config = {"h5_scan_group": scan_group, "pattern_type": "unprocessed"}
    with _open_reader(h5_path, reader_config) as reader:
        g = reader.orientation(reference_index)
        with csv_output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("scan_index", "column", "row", "status",
                             quality_key, "omega_1_mrad",
                             "omega_2_mrad", "omega_3_mrad"))
            for row in rows:
                col, y = int(row["column"]), int(row["row"])
                if row["status"] == "ok":
                    f = np.array([[float(row[f"F_{i}{j}"]) for j in range(1, 4)]
                                  for i in range(1, 4)])
                    rotation = sample_rotation_vector_mrad(f, g)
                    error = float(row[quality_key])
                    maps[:, y, col] = rotation
                    rms[y, col] = (error if not homography or
                                   row.get("homography_converged") == "True" else np.nan)
                else:
                    rotation = (np.nan, np.nan, np.nan)
                    error = np.nan
                writer.writerow((row["scan_index"], col, y, row["status"], error,
                                 *(display_rotation @ rotation)))
    return maps, rms, csv_output


def _shared_rotation_limit(maps: np.ndarray, percentile: float = 99.5) -> float:
    """Use the component with the largest absolute percentile for all panels."""
    if not 0 < percentile <= 100:
        raise ValueError("color percentile must be in (0, 100]")
    limits = []
    for component in maps:
        finite = np.abs(component[np.isfinite(component)])
        if len(finite):
            limits.append(float(np.percentile(finite, percentile)))
    return max(max(limits, default=0.0), 1e-12)


def plot_rotations(maps: np.ndarray, rms: np.ndarray, output_dir: Path,
                   color_percentile: float = 99.5,
                   sample_axis_rotation_degrees: float = 0.0,
                   quality_name: str = "RMS", quality_limit: float = 2.0):
    paths = []
    display_basis = rotate_vector_maps(maps, sample_axis_rotation_degrees)
    for masked in (False, True):
        display = display_basis.copy()
        if masked:
            display[:, ~(np.isfinite(rms) & (rms <= quality_limit))] = np.nan
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#e6e6e6")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        limit = _shared_rotation_limit(display, color_percentile)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        for axis, ax in enumerate(axes):
            data = display[axis]
            image = ax.imshow(data, origin="upper", interpolation="nearest", cmap=cmap,
                              norm=norm)
            ax.set_title(rf"$\omega_{{{axis + 1}}}$")
            ax.set_xlabel("Map column")
            ax.set_ylabel("Map row")
        fig.colorbar(image, ax=axes, shrink=0.80, pad=0.02, label="mrad")
        count = int(np.count_nonzero(np.isfinite(display[0])))
        qualifier = f" | {quality_name} <= {quality_limit:g}" if masked else ""
        clipping = (f"{color_percentile:g}th percentile"
                    if color_percentile < 100 else "full range")
        axis_note = (f" | sample axes +{sample_axis_rotation_degrees:g}°"
                     if sample_axis_rotation_degrees else "")
        fig.suptitle(f"Relative lattice rotation in sample frame{axis_note}{qualifier}\n"
                     f"{count:,}/{rms.size:,} shown | right-hand positive | "
                     f"shared ±{limit:.3g} mrad ({clipping})")
        path = output_dir / ("sample_rotations_quality_masked.png" if masked
                             else "sample_rotations.png")
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path,
                        default=Path("results_roi256/scan_results.csv"))
    parser.add_argument("--input-file", "--h5oina-file", dest="input_file", type=Path,
                        default=Path(CONFIG.get("input_file", CONFIG.get("h5oina_file"))))
    parser.add_argument("--output-dir", type=Path, default=Path("results_roi256"))
    parser.add_argument("--scan-group", default=CONFIG.get("h5_scan_group"))
    parser.add_argument("--color-percentile", type=float, default=99.5,
                        help="shared signed limit from the widest rotation map; 100 uses full range")
    parser.add_argument("--sample-axis-rotation-degrees", type=float, default=0.0,
                        help="rotate displayed sample-frame rotation components about axis 3")
    args = parser.parse_args()
    maps, rms, csv_output = extract_rotations(
        args.csv_path, args.input_file, args.output_dir, args.scan_group,
        args.sample_axis_rotation_degrees)
    print(csv_output)
    with args.csv_path.open(newline="", encoding="utf-8") as stream:
        homography = next(csv.DictReader(stream)).get("analysis_method") == "homography"
    for path in plot_rotations(maps, rms, args.output_dir, args.color_percentile,
                               args.sample_axis_rotation_degrees,
                               "ZNSSD RMS" if homography else "RMS px",
                               0.5 if homography else 2.0):
        print(path)


if __name__ == "__main__":
    main()
