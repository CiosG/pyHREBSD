"""Command-line entry point for PyHREBSD ROI shift measurement."""

import argparse
import csv
from pathlib import Path

from .correlation import grid_rois, measure_pattern_shifts
from .io import read_pattern


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure ROI shifts between two EBSD patterns")
    parser.add_argument("reference", type=Path)
    parser.add_argument("scan", type=Path)
    parser.add_argument("--roi-size", type=int, required=True, help="square ROI width in pixels")
    parser.add_argument("--roi-count", type=int, default=49, help="requested grid count; rounded to a square")
    parser.add_argument("--output", type=Path, required=True, help="CSV output path")
    args = parser.parse_args()

    reference = read_pattern(args.reference)
    scan = read_pattern(args.scan)
    centers = grid_rois(reference.shape, args.roi_size, args.roi_count)
    results = measure_pattern_shifts(reference, scan, centers, args.roi_size)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("x", "y", "dx", "dy", "coefficient", "peak_confidence"))
        for result in results:
            writer.writerow((result.x, result.y, result.dx, result.dy, result.coefficient, result.peak_confidence))
    print(f"Wrote {len(results)} ROI shifts to {args.output}")


if __name__ == "__main__":
    main()
