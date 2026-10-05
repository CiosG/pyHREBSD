"""Detect grains from H5OINA orientations without rerunning pattern correlation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pyhrebsd.grains import GrainMap, segment_grains
from pyhrebsd.h5oina import H5OINAReader


def save_grain_map(grains: GrainMap, output_dir: Path, contrast: np.ndarray | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = grains.labels.shape
    with (output_dir / "grain_ids.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scan_index", "column", "row", "grain_id", "phase_id"))
        for index, grain_id in enumerate(grains.labels.flat):
            writer.writerow((index, index % width, index // width, int(grain_id),
                             int(grains.phase_ids[grain_id])))
    with (output_dir / "grain_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("grain_id", "phase_id", "points", "centroid_column", "centroid_row"))
        for grain_id in range(1, grains.count + 1):
            y, x = np.nonzero(grains.labels == grain_id)
            writer.writerow((grain_id, int(grains.phase_ids[grain_id]),
                             int(grains.sizes[grain_id]), float(x.mean()), float(y.mean())))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    if contrast is not None:
        ax.imshow(contrast.reshape(height, width), cmap="gray", origin="upper",
                  interpolation="nearest", vmin=0, vmax=255)
    else:
        ax.imshow(np.zeros((height, width)), cmap="gray", origin="upper")
    labels = grains.labels
    boundary = np.zeros((height, width), dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    overlay = np.zeros((height, width, 4), dtype=float)
    overlay[boundary] = (1, 0.1, 0.1, 0.85)
    ax.imshow(overlay, origin="upper", interpolation="nearest")
    # Label the largest grains without filling the map with tiny labels.
    largest = np.argsort(grains.sizes[1:])[-30:][::-1] + 1
    for grain_id in largest:
        if grains.sizes[grain_id] < 5:
            continue
        y, x = np.nonzero(labels == grain_id)
        ax.text(x.mean(), y.mean(), str(grain_id), ha="center", va="center",
                color="yellow", fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1})
    ax.set(xlabel="Map column", ylabel="Map row", title=f"Detected grains: {grains.count}")
    fig.savefig(output_dir / "grain_map.png", dpi=200)
    plt.close(fig)


def detect_from_reader(reader: H5OINAReader, output_dir: Path,
                       threshold_degrees: float = 5.0, min_size: int = 5,
                       symmetry: str = "cubic") -> GrainMap:
    eulers, phases, contrast = reader.grain_inputs()
    grains = segment_grains(eulers, phases, (reader.y_cells, reader.x_cells),
                            threshold_degrees, min_size, symmetry)
    save_grain_map(grains, output_dir, contrast)
    print(f"Detected {grains.count} grains; {grains.sizes[0]} unindexed/removed points; "
          f"wrote {output_dir / 'grain_map.png'}")
    return grains


def main() -> None:
    from run_pyhrebsd import CONFIG, _path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5oina", type=Path, default=_path(CONFIG["h5oina_file"]))
    parser.add_argument("--output-dir", type=Path, default=_path(CONFIG["output_dir"]))
    parser.add_argument("--threshold-degrees", type=float,
                        default=CONFIG.get("grain_threshold_degrees", 5.0))
    parser.add_argument("--min-size", type=int, default=CONFIG.get("grain_min_size", 5))
    parser.add_argument("--symmetry", choices=("cubic", "none"),
                        default=CONFIG.get("grain_symmetry", "cubic"))
    args = parser.parse_args()
    with H5OINAReader(args.h5oina, CONFIG.get("h5_scan_group"),
                       CONFIG.get("h5_pattern_type", "processed")) as reader:
        detect_from_reader(reader, args.output_dir, args.threshold_degrees,
                           args.min_size, args.symmetry)


if __name__ == "__main__":
    main()
