"""Export a human-readable inventory of the HDF5 material database."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py


FIELDS = (
    "key", "name", "crystal_system", "verification_status", "c11_gpa",
    "c12_gpa", "c13_gpa", "c33_gpa", "c44_gpa", "c66_gpa",
    "minimum_eigenvalue_gpa", "legacy_max_difference_gpa",
    "temperature_k", "temperature_description", "uncertainty_available",
    "reference_id", "reference_path", "reference_title", "reference_doi",
    "reference_url", "source_line", "note",
)


def export(database: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(database, "r") as file, output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for key in sorted(file["materials"]):
            group = file["materials"][key]
            matrix = group["stiffness_voigt_gpa"]
            writer.writerow({
                "key": key,
                "name": group.attrs["name"],
                "crystal_system": group.attrs["crystal_system"],
                "verification_status": group.attrs["verification_status"],
                "c11_gpa": matrix[0, 0], "c12_gpa": matrix[0, 1],
                "c13_gpa": matrix[0, 2], "c33_gpa": matrix[2, 2],
                "c44_gpa": matrix[3, 3], "c66_gpa": matrix[5, 5],
                "minimum_eigenvalue_gpa": group.attrs["minimum_stiffness_eigenvalue_gpa"],
                "legacy_max_difference_gpa": group.attrs.get(
                    "legacy_max_absolute_difference_gpa", ""),
                "temperature_k": group.attrs.get("temperature_k", ""),
                "temperature_description": group.attrs.get("temperature_description", ""),
                "uncertainty_available": "stiffness_uncertainty_gpa" in group,
                "reference_id": group.attrs.get("reference_id", ""),
                "reference_path": group.attrs.get("reference_path", ""),
                "reference_title": group.attrs["reference_title"],
                "reference_doi": group.attrs["reference_doi"],
                "reference_url": group.attrs["reference_url"],
                "source_line": group.attrs["source_line"],
                "note": group.attrs["note"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.database, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
