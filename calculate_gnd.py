"""Estimate system-resolved GND maps from PyHREBSD HR-EBSD scan CSV files.

Example: python calculate_gnd.py si_piezo_site1_results_tilt70
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from pyhrebsd.geometry import euler_to_matrix
from pyhrebsd.gnd import (dislocation_matrix, fcc_perfect_dislocations,
                         partial_nye_from_maps, solve_gnd_l1)


def _read_h5_geometry(path: Path, scan_group: str
                      ) -> tuple[float, float, np.ndarray, float, np.ndarray]:
    with h5py.File(path) as file:
        header = file[f"{scan_group}/EBSD/Header"]
        data = file[f"{scan_group}/EBSD/Data"]
        steps = []
        for axis in "XY":
            dataset = header[f"{axis} Step"]
            value = float(np.asarray(dataset[()]).reshape(-1)[0])
            unit = dataset.attrs.get("Unit", "um")
            if isinstance(unit, bytes):
                unit = unit.decode()
            if str(unit).lower() not in ("um", "µm", "μm", "micrometer", "micrometers"):
                raise ValueError(f"{axis} Step unit {unit!r} is not micrometres")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{axis} Step must be positive")
            steps.append(value * 1e-6)
        if "Scan Rotation" in header:
            angle = float(np.asarray(header["Scan Rotation"][()]).reshape(-1)[0])
            if abs(angle) > 1e-5:
                raise ValueError("nonzero Scan Rotation needs map-to-sample axis conversion")
        euler = np.asarray(data["Euler"][0], dtype=float).reshape(-1)
        if len(euler) != 3 or not np.all(np.isfinite(euler)):
            raise ValueError("reference point has no valid crystal orientation")
        phase = np.asarray(data["Phase"][:], dtype=np.uint8).reshape(-1)
        valid_phases = np.unique(phase[phase > 0])
        if len(valid_phases) != 1:
            raise ValueError("GND analysis currently requires one indexed phase")
        phase_header = header["Phases"][str(int(valid_phases[0]))]
        lengths = np.asarray(phase_header["Lattice Dimensions"][()], dtype=float).reshape(-1)
        if len(lengths) != 3 or not np.allclose(lengths, lengths[0], rtol=1e-4):
            raise ValueError("GND FCC basis requires a cubic lattice")
        unit = phase_header["Lattice Dimensions"].attrs.get("Unit", "angstrom")
        if isinstance(unit, bytes):
            unit = unit.decode()
        if str(unit).lower() not in ("angstrom", "ångström", "ang"):
            raise ValueError(f"unknown lattice dimension unit {unit!r}")
        return steps[0], steps[1], euler_to_matrix(*euler), lengths[0] * 1e-10, phase


def _read_maps(path: Path, shape: tuple[int, int], use_strain: bool
               ) -> tuple[np.ndarray, np.ndarray | None]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    count = int(np.prod(shape))
    if len(rows) != count or sorted(int(row["scan_index"]) for row in rows) != list(range(count)):
        raise ValueError(f"{path} does not contain a complete map")
    rotation = np.full((count, 3), np.nan)
    strain = np.full((count, 3, 3), np.nan) if use_strain else None
    for row in rows:
        index = int(row["scan_index"])
        if row["status"] != "ok" or (row["analysis_method"] == "homography"
                                      and row["homography_converged"] != "True"):
            continue
        rotation[index] = [float(row[f"rotation_sample_{i}_mrad"]) for i in range(1, 4)]
        if strain is not None:
            strain[index] = [[float(row[f"strain_sample_{i}{j}"]) for j in range(1, 4)]
                             for i in range(1, 4)]
    return rotation.reshape(shape + (3,)), (strain.reshape(shape + (3, 3))
                                               if strain is not None else None)


def _plot_totals(total: np.ndarray, edge: np.ndarray, screw: np.ndarray,
                 output: Path, title: str) -> None:
    arrays = (total, edge, screw)
    positive = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in arrays])
    if not len(positive):
        return
    lo = max(float(np.percentile(positive, 2)), 1e8)
    hi = max(float(np.percentile(positive, 99)), lo * 1.01)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, name, values in zip(axes, ("Total", "Edge", "Screw"), arrays):
        display = np.ma.masked_where(~np.isfinite(values) | (values <= 0), values)
        image = ax.imshow(display, origin="upper", cmap="magma", norm=LogNorm(lo, hi))
        ax.set_title(name)
        ax.set_xlabel("Map column")
        ax.set_ylabel("Map line")
    fig.colorbar(image, ax=axes, label=r"GND density (m$^{-2}$)")
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_systems(density: np.ndarray, names: list[str], output: Path,
                  title: str) -> None:
    values = np.abs(density)
    positive = values[np.isfinite(values) & (values > 0)]
    if not len(positive):
        return
    lo = max(float(np.percentile(positive, 2)), 1e8)
    hi = max(float(np.percentile(positive, 99)), lo * 1.01)
    fig, axes = plt.subplots(3, 6, figsize=(21, 11), constrained_layout=True)
    for ax, name, values in zip(axes.flat, names, np.moveaxis(values, -1, 0)):
        display = np.ma.masked_where(~np.isfinite(values) | (values <= 0), values)
        image = ax.imshow(display, origin="upper", cmap="magma", norm=LogNorm(lo, hi))
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axes, label=r"Absolute signed density (m$^{-2}$)")
    fig.suptitle(title)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_comparison(maps: dict[str, np.ndarray], output: Path,
                     title: str) -> None:
    positive = np.concatenate([values[np.isfinite(values) & (values > 0)]
                               for values in maps.values()])
    if not len(positive):
        return
    lo = max(float(np.percentile(positive, 2)), 1e8)
    hi = max(float(np.percentile(positive, 99)), lo * 1.01)
    fig, axes = plt.subplots(1, len(maps), figsize=(11, 4.6), constrained_layout=True)
    for ax, (method, values) in zip(np.atleast_1d(axes), maps.items()):
        display = np.ma.masked_where(~np.isfinite(values) | (values <= 0), values)
        image = ax.imshow(display, origin="upper", cmap="magma", norm=LogNorm(lo, hi))
        ax.set_title(method.upper())
        ax.set_xlabel("Map column")
        ax.set_ylabel("Map line")
    fig.colorbar(image, ax=np.atleast_1d(axes).tolist(), label=r"Total GND density (m$^{-2}$)")
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--method", choices=("roi", "homography", "both"), default="both")
    parser.add_argument("--strain-correction", action="store_true",
                        help="Include measured in-plane elastic strain gradients")
    parser.add_argument("--lattice-parameter-nm", type=float, default=None,
                        help="Override cubic lattice parameter from H5OINA phase")
    parser.add_argument("--material", choices=("silicon", "nickel"), default="silicon")
    parser.add_argument("--poisson-ratio", type=float, default=0.28)
    parser.add_argument("--gradient-span", type=int, default=1,
                        help="Number of map steps across each forward-difference stencil")
    args = parser.parse_args()
    root = args.results_dir.resolve()
    summary = root / "comparison" / "summary.json"
    if summary.is_file():
        metadata = json.loads(summary.read_text(encoding="utf-8"))["metadata"]
    else:
        metadata = json.loads((root / "roi" / "settings.json").read_text(encoding="utf-8"))["metadata"]
    shape = tuple(metadata["map_shape"])
    x_step, y_step, orientation, h5_lattice_m, phase = _read_h5_geometry(
        Path(metadata["source_file"]), str(metadata["scan_group"]))
    lattice_m = (h5_lattice_m if args.lattice_parameter_nm is None
                 else args.lattice_parameter_nm * 1e-9)
    types = fcc_perfect_dislocations(lattice_m)
    matrix = dislocation_matrix(types, orientation)
    methods = ("roi", "homography") if args.method == "both" else (args.method,)
    comparison_maps = {}
    for method in methods:
        rotation, strain = _read_maps(root / method / "scan_results.csv", shape,
                                      args.strain_correction)
        rotation[phase.reshape(shape) == 0] = np.nan
        if strain is not None:
            strain[phase.reshape(shape) == 0] = np.nan
        observed = partial_nye_from_maps(rotation, x_step, y_step, strain,
                                         args.gradient_span)
        density, residual = solve_gnd_l1(observed, matrix, types, args.poisson_ratio)
        edge = np.nansum(np.abs(density[..., :12]), axis=-1)
        screw = np.nansum(np.abs(density[..., 12:]), axis=-1)
        total = edge + screw
        valid = np.all(np.isfinite(density), axis=-1)
        for values in (edge, screw, total):
            values[~valid] = np.nan
        comparison_maps[method] = total
        variant = "gnd_strain_corrected" if args.strain_correction else "gnd"
        if args.gradient_span != 1:
            variant += f"_span{args.gradient_span}"
        destination = root / method / variant
        destination.mkdir(parents=True, exist_ok=True)
        names = [kind.name for kind in types]
        np.savez_compressed(destination / "gnd_maps.npz", signed_density_m2=density,
                            total_density_m2=total, edge_density_m2=edge,
                            screw_density_m2=screw, observed_nye_per_m=observed,
                            fit_residual_per_m=residual, system_names=np.asarray(names))
        _plot_totals(total, edge, screw, destination / "gnd_total_edge_screw.png",
                     f"{method.upper()} GND: minimum line-energy estimate")
        _plot_systems(density, names, destination / "gnd_by_dislocation_type.png",
                      f"{method.upper()} GND by dislocation type")
        info = {
            "method": method, "source_csv": str(root / method / "scan_results.csv"),
            "paper_model": "six-observable partial Nye minimum-line-energy L1",
            "elastic_strain_gradients_included": args.strain_correction,
            "unmeasured_depth_strain_gradients_assumed_zero": True,
            "strain_correction_scope": ("available in-plane derivatives only; not a full 3-D Nye tensor"
                                        if args.strain_correction else "rotation gradients only"),
            "step_m": {"x": x_step, "y": y_step},
            "gradient_span_map_steps": args.gradient_span,
            "gradient_distance_m": {"x": x_step * args.gradient_span,
                                    "y": y_step * args.gradient_span},
            "material": args.material,
            "lattice_parameter_m": lattice_m,
            "burgers_m": lattice_m / np.sqrt(2),
            "poisson_ratio_for_line_energy": args.poisson_ratio,
            "valid_points": int(np.count_nonzero(valid)),
            "total_density_m2_median": float(np.nanmedian(total)),
            "total_density_m2_p95": float(np.nanpercentile(total, 95)),
            "nyefit_residual_per_m_p95": float(np.nanpercentile(
                np.linalg.norm(residual, axis=-1), 95)),
            "system_names": names,
            "note": "Signed system densities are model-dependent. Totals sum absolute values; SSD is not measured.",
        }
        (destination / "summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(method, "valid", info["valid_points"], "median", info["total_density_m2_median"])
    if len(comparison_maps) == 2:
        suffix = ("_strain_corrected" if args.strain_correction else "")
        if args.gradient_span != 1:
            suffix += f"_span{args.gradient_span}"
        _plot_comparison(
            comparison_maps,
            root / "comparison" / f"gnd_total_roi_vs_homography{suffix}.png",
            "Total GND estimate; shared logarithmic scale")


if __name__ == "__main__":
    main()
