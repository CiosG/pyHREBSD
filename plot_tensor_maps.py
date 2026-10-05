"""Plot sample-frame strain tensor and rotation map panels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

from pyhrebsd.frame_views import rotate_tensor_maps, rotate_vector_maps


KINDS = {
    "elastic": ("strain_sample", r"Elastic strain $\epsilon$"),
    "deviatoric": ("deviatoric_strain_sample", r"Relative deviatoric strain $\epsilon'$"),
}

DISPLAY_NAMES = {
    "validation_roi_cpu": "Standard CPU",
    "validation_roi_gpu": "Standard GPU",
    "validation_roi_gpu_quadratic2d": "Standard GPU, 2D peak",
    "validation_roi_gpu_unprocessed_quadratic2d": "Raw 16-bit GPU, background corrected, 2D peak",
    "validation_homography_cpu_serial": "Homography CPU",
    "validation_homography_gpu_serial": "Homography GPU",
    "validation_homography_gpu_unprocessed_static_lmsd_binning":
        "Homography GPU, corrected 16-bit, 3x3 binning",
    "validation_homography_gpu_unprocessed_static_lmsd_binning_iter250":
        "Homography GPU, 16-bit, bin 3, 250 iter",
}


def load_result(path: Path, axis_degrees: float,
                mask_bad_fit: bool = True,
                roi_quality_limit: float = 2.0) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} has no results")
    shape = (max(int(row["row"]) for row in rows) + 1,
             max(int(row["column"]) for row in rows) + 1)
    homography = rows[0]["analysis_method"] == "homography"
    quality_name = "znssd_rms" if homography else "rms_shift_error_pixels"
    quality_limit = 0.5 if homography else roi_quality_limit
    fields = [f"{prefix}_{i}{j}" for prefix, _ in KINDS.values()
              for i in range(1, 4) for j in range(i, 4)]
    fields += [f"rotation_sample_{i}_mrad" for i in range(1, 4)]
    maps = {field: np.full(shape, np.nan) for field in fields}
    valid = np.zeros(shape, dtype=bool)
    for row in rows:
        y, x = int(row["row"]), int(row["column"])
        if row["status"] != "ok":
            continue
        quality = float(row[quality_name])
        good = np.isfinite(quality) and quality <= quality_limit
        if homography:
            good &= row["homography_converged"] == "True"
        if mask_bad_fit and not good:
            continue
        values = {field: float(row[field]) for field in fields}
        if not all(np.isfinite(value) for value in values.values()):
            continue
        valid[y, x] = True
        for field, value in values.items():
            maps[field][y, x] = value
    if not valid.any():
        raise ValueError(f"{path} has no valid fitted points")

    tensors = {}
    for kind, (prefix, _) in KINDS.items():
        rotated = rotate_tensor_maps(maps, prefix, axis_degrees)
        tensor = np.empty((3, 3) + shape)
        for i in range(3):
            for j in range(3):
                tensor[i, j] = rotated[f"{prefix}_{min(i,j)+1}{max(i,j)+1}"] * 1000
        tensors[kind] = tensor
    rotations = np.stack([maps[f"rotation_sample_{i}_mrad"]
                          for i in range(1, 4)])
    rotations = rotate_vector_maps(rotations, axis_degrees)
    return {"path": path, "shape": shape, "valid": valid,
            "tensors": tensors, "rotations": rotations,
            "quality": f"{quality_name} ≤ {quality_limit:g}",
            "axis_degrees": axis_degrees, "mask_bad_fit": mask_bad_fit}


def shared_limit(arrays: list[np.ndarray], percentile: float) -> float:
    if not 0 < percentile <= 100:
        raise ValueError("color percentile must be in (0, 100]")
    limits = []
    for array in arrays:
        for panel in array.reshape((-1,) + array.shape[-2:]):
            finite = np.abs(panel[np.isfinite(panel)])
            if finite.size:
                limits.append(float(np.percentile(finite, percentile)))
    return max(max(limits, default=0.0), 1e-12)


def draw(result: dict, kind: str, strain_limit: float, rotation_limit: float,
         output: Path, percentile: float, step_um: float = 1.0,
         strain_linthresh: float | None = None) -> None:
    tensor = result["tensors"][kind]
    rotations = result["rotations"]
    fig = plt.figure(figsize=(11.8, 14.5), facecolor="white")
    grid = fig.add_gridspec(4, 4, width_ratios=(0.045, 1, 1, 1),
                           left=0.055, right=0.985, top=0.88, bottom=0.095,
                           wspace=0.17, hspace=0.30)
    axes = [[fig.add_subplot(grid[row, col + 1]) for col in range(3)]
            for row in range(4)]
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    strain_norm = (TwoSlopeNorm(vmin=-strain_limit, vcenter=0, vmax=strain_limit)
                   if strain_linthresh is None else
                   SymLogNorm(linthresh=strain_linthresh, linscale=1.0,
                              vmin=-strain_limit, vmax=strain_limit, base=10))
    rotation_norm = TwoSlopeNorm(vmin=-rotation_limit, vcenter=0,
                                 vmax=rotation_limit)
    for row in range(3):
        for col in range(3):
            ax = axes[row][col]
            image_strain = ax.imshow(tensor[row, col], origin="upper",
                                     interpolation="nearest", cmap=cmap,
                                     norm=strain_norm)
            ax.set_title(rf"$\epsilon_{{{row+1}{col+1}}}$" if kind == "elastic"
                         else rf"$\epsilon'_{{{row+1}{col+1}}}$", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.75)
    for col in range(3):
        ax = axes[3][col]
        image_rotation = ax.imshow(rotations[col], origin="upper",
                                   interpolation="nearest", cmap=cmap,
                                   norm=rotation_norm)
        ax.set_title(rf"$\omega_{{{col+1}}}$", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
    cax_strain = fig.add_subplot(grid[:3, 0])
    strain_ticks = (np.linspace(-strain_limit, strain_limit, 5)
                    if strain_linthresh is None else
                    np.array(sorted({-strain_limit, strain_limit, 0.0,
                                     -strain_linthresh, strain_linthresh,
                                     *[value for value in (-1.0, -0.1, 0.1, 1.0)
                                       if abs(value) < strain_limit]})))
    cbar_strain = fig.colorbar(image_strain, cax=cax_strain,
                              ticks=strain_ticks)
    if strain_linthresh is not None:
        cbar_strain.ax.set_yticklabels([f"{value:.3g}" for value in strain_ticks])
    cbar_strain.ax.yaxis.set_ticks_position("left")
    cbar_strain.ax.set_title("mm/m", fontsize=10, pad=8)
    cbar_strain.ax.tick_params(labelsize=8)
    cax_rotation = fig.add_subplot(grid[3, 0])
    cbar_rotation = fig.colorbar(image_rotation, cax=cax_rotation,
                                ticks=np.linspace(-rotation_limit, rotation_limit, 5))
    cbar_rotation.ax.yaxis.set_ticks_position("left")
    cbar_rotation.ax.set_title("mrad", fontsize=10, pad=8)
    cbar_rotation.ax.tick_params(labelsize=8)

    _, title = KINDS[kind]
    count = int(result["valid"].sum())
    total = int(result["valid"].size)
    fig.suptitle(f"{title} and lattice rotation\n"
                 f"{result.get('display_name', DISPLAY_NAMES.get(result['path'].parent.name, result['path'].parent.name))}"
                 f"  |  sample axes +{result['axis_degrees']:g}°  |  "
                 f"{count:,}/{total:,} shown", fontsize=16, y=0.96)
    white_note = "poor fit" if result["mask_bad_fit"] else "failed/nonfinite"
    strain_scale = (f"symmetric log (linear |ε| ≤ {strain_linthresh:g} mm/m)"
                    if strain_linthresh is not None else "linear")
    fig.text(0.52, 0.905,
             f"Strain {strain_scale}, shared ±{strain_limit:.3g} mm/m; "
             f"rotation linear ±{rotation_limit:.3g} mrad; "
             f"{percentile:g}th percentile; white = {white_note}",
             ha="center", va="center", fontsize=9)
    quality_note = result["quality"] if result["mask_bad_fit"] else "unmasked"
    fig.text(0.52, 0.07,
             f"Raster: {result['shape'][1]}×{result['shape'][0]}  |  "
             f"step: {step_um:g} µm  |  {quality_note}  |  "
             "X₁ ←   X₂ ↓", ha="center", fontsize=10)
    # Draw a physical scale bar with the same width-to-pixel ratio as a map.
    panel = axes[3][0].get_position()
    bar_pixels = 20 / step_um
    x0 = panel.x0
    x1 = x0 + panel.width * bar_pixels / result["shape"][1]
    y = 0.068
    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure,
                          color="black", linewidth=2))
    fig.add_artist(Line2D([x0, x0], [y - 0.004, y + 0.004],
                          transform=fig.transFigure, color="black", linewidth=1.5))
    fig.add_artist(Line2D([x1, x1], [y - 0.004, y + 0.004],
                          transform=fig.transFigure, color="black", linewidth=1.5))
    fig.text((x0 + x1) / 2, 0.049, "20 µm", ha="center", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("tensor_maps"))
    parser.add_argument("--sample-axis-rotation-degrees", type=float, default=90.0)
    parser.add_argument("--color-percentile", type=float, default=99.5)
    parser.add_argument("--step-um", type=float, required=True,
                        help="H5OINA map step in micrometres for the scale bar")
    parser.add_argument("--include-unmasked", action="store_true")
    parser.add_argument("--strain-symlog", action="store_true",
                        help="symmetric logarithmic strain colors with one scale across all inputs and strain kinds")
    parser.add_argument("--strain-linear-threshold", type=float, default=0.5,
                        help="linear region around zero in mm/m for --strain-symlog")
    args = parser.parse_args()
    if args.step_um <= 0:
        parser.error("--step-um must be positive")
    results = [load_result(path, args.sample_axis_rotation_degrees)
               for path in args.csv_paths]
    if len({result["shape"] for result in results}) != 1:
        parser.error("all CSV files must have the same raster shape")
    rotation_limit = shared_limit([r["rotations"] for r in results],
                                  args.color_percentile)
    shared_strain_limit = (shared_limit([r["tensors"][kind] for r in results
                                          for kind in KINDS], args.color_percentile)
                           if args.strain_symlog else None)
    if args.strain_symlog and not 0 < args.strain_linear_threshold < shared_strain_limit:
        parser.error("--strain-linear-threshold must be positive and below the shared strain limit")
    suffix = "_symlog" if args.strain_symlog else ""
    for kind in KINDS:
        strain_limit = (shared_strain_limit if args.strain_symlog else
                        shared_limit([r["tensors"][kind] for r in results],
                                     args.color_percentile))
        for result in results:
            output = (args.output_dir / result["path"].parent.name /
                      f"{kind}_strain_rotations{suffix}.png")
            draw(result, kind, strain_limit, rotation_limit, output,
                 args.color_percentile, args.step_um,
                 args.strain_linear_threshold if args.strain_symlog else None)
            print(output)
            if args.include_unmasked:
                raw = load_result(result["path"], args.sample_axis_rotation_degrees,
                                  mask_bad_fit=False)
                raw_output = (args.output_dir / result["path"].parent.name /
                              f"{kind}_strain_rotations_unmasked{suffix}.png")
                draw(raw, kind, strain_limit, rotation_limit, raw_output,
                     args.color_percentile, args.step_um,
                     args.strain_linear_threshold if args.strain_symlog else None)
                print(raw_output)


if __name__ == "__main__":
    main()
