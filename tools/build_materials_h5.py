"""Build the PyHREBSD elastic-constant database from a legacy TSV file.

The input has no header: material name, crystal-system label, then a complete
6 x 6 stiffness matrix in Voigt order 11, 22, 33, 23, 13, 12.  Values are GPa.
Only entries independently matched to the cited online NIST compilation are
written. Unverified legacy rows are deliberately omitted.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path

import h5py
import numpy as np


NIST_TITLE = (
    "H. M. Ledbetter and S. D. Kim, Monocrystal Elastic Constants and "
    "Derived Properties of the Cubic and the Hexagonal Elements (2001)"
)
NIST_URL = (
    "https://www.nist.gov/publications/monocrystal-elastic-constants-and-"
    "deprived-properties-cubic-and-hexagonal-elements"
)
NIST_DOI = "10.1016/B978-012445760-7/50033-2"
VOIGT_LABELS = ("11", "22", "33", "23", "13", "12")
COMPONENT_LABELS = np.asarray(
    [[f"C{row}{column}" for column in range(1, 7)] for row in range(1, 7)],
    dtype=object,
)

REFERENCES = {
    "ledbetter_kim_2001": {
        "authors": "H. M. Ledbetter; S. D. Kim",
        "title": "Monocrystal Elastic Constants and Derived Properties of the Cubic and the Hexagonal Elements",
        "year": 2001,
        "publication": "Handbook of Elastic Properties of Solids, Liquids, and Gases, Volume II",
        "doi": NIST_DOI,
        "url": NIST_URL,
    },
    "polian_grimsditch_grzegory_1996": {
        "authors": "A. Polian; M. Grimsditch; I. Grzegory",
        "title": "Elastic constants of gallium nitride",
        "year": 1996,
        "publication": "Journal of Applied Physics",
        "volume": "79",
        "pages": "3343-3344",
        "doi": "10.1063/1.361236",
        "url": "https://srdata.nist.gov/CeramicDataPortal/Scd/Z00476",
    },
    "djemia_roussigne_dirras_jackson_2004": {
        "authors": "P. Djemia; Y. Roussigné; G. F. Dirras; K. M. Jackson",
        "title": "Elastic properties of beta-SiC films by Brillouin light scattering",
        "year": 2004,
        "publication": "Journal of Applied Physics",
        "volume": "95",
        "pages": "2324-2330",
        "doi": "10.1063/1.1642281",
        "url": "https://doi.org/10.1063/1.1642281",
    },
    "kamitani_et_al_1997": {
        "authors": "K. Kamitani; M. Grimsditch; J. C. Nipko; C.-K. Loong; M. Okada; I. Kimura",
        "title": "The elastic constants of silicon carbide: A Brillouin-scattering study of 4H and 6H SiC single crystals",
        "year": 1997,
        "publication": "Journal of Applied Physics",
        "volume": "82",
        "pages": "3152-3154",
        "doi": "10.1063/1.366100",
        "url": "https://doi.org/10.1063/1.366100",
    },
    "mcneil_grimsditch_french_1993": {
        "authors": "L. E. McNeil; M. Grimsditch; R. H. French",
        "title": "Vibrational Spectroscopy of Aluminum Nitride",
        "year": 1993,
        "publication": "Journal of the American Ceramic Society",
        "volume": "76",
        "pages": "1132-1136",
        "doi": "10.1111/j.1151-2916.1993.tb03730.x",
        "url": "https://doi.org/10.1111/j.1151-2916.1993.tb03730.x",
    },
}

REFERENCE_ID_BY_DOI = {record["doi"]: key for key, record in REFERENCES.items()}

# Values transcribed from Tables 7.1 and 7.2 of Ledbetter and Kim.  The keys
# match unambiguous entries in the supplied legacy table.  C66 for hexagonal
# crystals is fixed by symmetry as (C11 - C12) / 2.
NIST_CUBIC = {
    "Carbon, diamond": (1076.0, 125.0, 575.8),
    "Aluminium": (106.75, 60.41, 28.34),
    "Calcium": (27.8, 18.2, 16.3),
    "Chromium": (350.0, 67.8, 100.8),
    "Copper": (169.68, 122.55, 74.499),
    "Iron bcc": (231.4, 134.7, 116.4),
    "Iron fcc": (276.0, 173.5, 136.0),
    "Germanium": (128.53, 48.26, 66.80),
    "Gold": (192.9, 163.8, 41.50),
    "Iridium": (599.47, 355.82, 268.82),
    "Lead": (49.66, 42.31, 14.98),
    "Lithium": (13.50, 11.44, 8.78),
    "Molybdenum": (463.7, 157.8, 109.2),
    "Nickel": (248.1, 154.9, 124.2),
    "Niobium": (240.19, 125.58, 28.22),
    "Platinum": (227.1, 176.04, 149.8),
    "Potassium": (3.70, 3.14, 1.88),
    "Rubidium": (2.45, 1.64, 1.00),
    "Silicon": (165.78, 63.94, 79.62),
    "Silver": (122.2, 90.70, 45.40),
    "Sodium": (7.39, 6.22, 4.19),
    "Strontium": (14.7, 9.90, 5.74),
    "Thorium": (75.3, 48.9, 47.89),
    "Vanadium": (228.7, 119.0, 43.15),
    "Tungsten": (522.39, 204.37, 160.83),
}

NIST_HEXAGONAL = {
    "Cobalt (hcp)": (295.0, 159.0, 111.0, 335.0, 71.0),
    "Erbium": (84.1, 29.4, 22.6, 84.7, 27.4),
    "Gadolinium": (67.8, 25.6, 20.7, 71.2, 23.8),
    "Hafnium": (181.0, 77.0, 66.0, 197.0, 55.7),
    "Magnesium": (59.3, 25.7, 21.4, 61.5, 16.4),
    "Rhenium": (616.0, 273.0, 206.0, 683.0, 161.0),
    "Ruthenium": (563.0, 188.0, 168.0, 624.0, 181.0),
    "Titanium": (160.0, 90.0, 66.0, 181.0, 46.5),
}

# Useful elemental entries from the same NIST tables that were absent from the
# supplied file.  Tuple: display name, crystal system, independent constants.
NIST_ADDITIONS = {
    "rhodium": ("Rhodium", "cubic", (413.0, 184.0, 79.62)),
    "scandium": ("Scandium", "hexagonal", (99.3, 39.7, 29.4, 107.0, 27.7)),
    "neodymium": ("Neodymium", "hexagonal", (53.8, 24.6, 16.6, 59.9, 15.0)),
    "praseodymium": ("Praseodymium", "hexagonal", (49.4, 23.0, 14.3, 57.4, 13.6)),
    "terbium": ("Terbium", "hexagonal", (69.2, 25.0, 21.8, 74.4, 21.7)),
    "lutetium": ("Lutetium", "hexagonal", (56.2, 32.0, 25.0, 80.9, 26.8)),
}

# Experimentally measured semiconductor compounds with phase-specific sources.
# Tuple fields: display name, crystal system, constants, uncertainties (or None),
# reference title, URL, DOI, and a temperature description.
VERIFIED_COMPOUNDS = {
    "gallium_nitride_wurtzite": (
        "Gallium nitride (wurtzite)", "hexagonal",
        (390.0, 145.0, 106.0, 398.0, 105.0),
        (15.0, 20.0, 20.0, 20.0, 10.0, 10.0),
        "A. Polian, M. Grimsditch and I. Grzegory, Elastic constants of gallium nitride",
        "https://srdata.nist.gov/CeramicDataPortal/Scd/Z00476",
        "10.1063/1.361236", "300 K",
    ),
    "silicon_carbide_3c": (
        "Silicon carbide (3C)", "cubic", (395.0, 132.0, 236.0),
        (12.0, 9.0, 7.0),
        "P. Djemia et al., Elastic properties of beta-SiC films by Brillouin light scattering",
        "https://doi.org/10.1063/1.1642281", "10.1063/1.1642281", "room temperature",
    ),
    "silicon_carbide_4h": (
        "Silicon carbide (4H)", "hexagonal",
        (501.0, 111.0, 52.0, 553.0, 163.0),
        (4.0, 5.0, 9.0, 4.0, 4.0, 3.2),
        "K. Kamitani et al., The elastic constants of silicon carbide: 4H and 6H single crystals",
        "https://doi.org/10.1063/1.366100", "10.1063/1.366100", "room temperature",
    ),
    "silicon_carbide_6h": (
        "Silicon carbide (6H)", "hexagonal",
        (501.0, 111.0, 52.0, 553.0, 163.0),
        (4.0, 5.0, 9.0, 4.0, 4.0, 3.2),
        "K. Kamitani et al., The elastic constants of silicon carbide: 4H and 6H single crystals",
        "https://doi.org/10.1063/1.366100", "10.1063/1.366100", "room temperature",
    ),
    "aluminium_nitride_wurtzite": (
        "Aluminium nitride (wurtzite)", "hexagonal",
        (411.0, 149.0, 99.0, 389.0, 125.0), None,
        "L. E. McNeil, M. Grimsditch and R. H. French, Vibrational spectroscopy of aluminum nitride",
        "https://doi.org/10.1111/j.1151-2916.1993.tb03730.x",
        "10.1111/j.1151-2916.1993.tb03730.x", "room temperature",
    ),
}


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def cubic(c11: float, c12: float, c44: float) -> np.ndarray:
    matrix = np.zeros((6, 6), dtype=np.float64)
    matrix[:3, :3] = c12
    np.fill_diagonal(matrix[:3, :3], c11)
    np.fill_diagonal(matrix[3:, 3:], c44)
    return matrix


def hexagonal(c11: float, c12: float, c13: float, c33: float,
              c44: float) -> np.ndarray:
    matrix = np.zeros((6, 6), dtype=np.float64)
    matrix[0, 0] = matrix[1, 1] = c11
    matrix[0, 1] = matrix[1, 0] = c12
    matrix[0, 2] = matrix[2, 0] = c13
    matrix[1, 2] = matrix[2, 1] = c13
    matrix[2, 2] = c33
    matrix[3, 3] = matrix[4, 4] = c44
    matrix[5, 5] = (c11 - c12) / 2
    return matrix


def uncertainty_matrix(crystal_system: str, values: tuple[float, ...] | None) -> np.ndarray | None:
    if values is None:
        return None
    if crystal_system == "cubic":
        return cubic(*values)
    c11, c12, c13, c33, c44, c66 = values
    matrix = hexagonal(c11, c12, c13, c33, c44)
    matrix[5, 5] = c66
    return matrix


def read_legacy(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if not row or not any(item.strip() for item in row):
                continue
            if len(row) != 38:
                raise ValueError(f"{path}:{line_number}: expected 38 fields, got {len(row)}")
            name = " ".join(row[0].replace(",", ", ").split()).replace(",  ", ", ")
            matrix = np.asarray([float(value) for value in row[2:]], dtype=np.float64).reshape(6, 6)
            records.append({"name": name, "crystal_system": row[1], "matrix": matrix,
                            "source_line": line_number})
    return records


def validation(matrix: np.ndarray) -> tuple[str, str, float]:
    symmetric = np.allclose(matrix, matrix.T, rtol=0, atol=1e-8)
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)
    minimum = float(eigenvalues.min())
    if not symmetric:
        return "suspect", "matrix is not symmetric", minimum
    if minimum <= 0:
        return "suspect", "matrix is not positive definite", minimum
    return "imported_unverified", "passed symmetry and positive-definiteness checks", minimum


def write_record(parent: h5py.Group, slug: str, name: str, crystal_system: str,
                 matrix: np.ndarray, status: str, reference_title: str,
                 reference_url: str = "", reference_doi: str = "",
                 note: str = "", source_line: int = -1,
                 temperature_k: float = np.nan, temperature_description: str = "",
                 uncertainty: np.ndarray | None = None,
                 reference_id: str = "") -> h5py.Group:
    group = parent.create_group(slug)
    data = group.create_dataset("stiffness_voigt_gpa", data=matrix,
                                compression="gzip", compression_opts=4,
                                shuffle=True, fletcher32=True)
    data.attrs["unit"] = "GPa"
    data.attrs["voigt_order"] = ",".join(VOIGT_LABELS)
    data.attrs["row_labels"] = np.asarray(
        [f"{label} (Voigt {index})" for index, label in enumerate(VOIGT_LABELS, 1)],
        dtype=h5py.string_dtype("utf-8"))
    data.attrs["column_labels"] = data.attrs["row_labels"]
    data.attrs["index_examples"] = (
        "[0,0]=C11; [0,1]=C12; [0,2]=C13; [2,2]=C33; "
        "[3,3]=C44; [4,4]=C55; [5,5]=C66"
    )
    group.attrs["name"] = name
    group.attrs["crystal_system"] = crystal_system.lower()
    group.attrs["verification_status"] = status
    group.attrs["reference_title"] = reference_title
    group.attrs["reference_url"] = reference_url
    group.attrs["reference_doi"] = reference_doi
    group.attrs["reference_id"] = reference_id
    group.attrs["reference_path"] = f"/references/{reference_id}" if reference_id else ""
    group.attrs["note"] = note
    group.attrs["temperature_k"] = temperature_k
    group.attrs["temperature_description"] = temperature_description
    group.attrs["source_line"] = source_line
    group.attrs["minimum_stiffness_eigenvalue_gpa"] = float(
        np.linalg.eigvalsh((matrix + matrix.T) / 2).min())
    if uncertainty is not None:
        uncertainty_data = group.create_dataset(
            "stiffness_uncertainty_gpa", data=uncertainty,
            compression="gzip", compression_opts=4, shuffle=True, fletcher32=True)
        uncertainty_data.attrs["unit"] = "GPa"
        uncertainty_data.attrs["component_labels_path"] = "/component_labels"
    return group


def build(source: Path, output: Path) -> dict[str, int]:
    records = read_legacy(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}
    counts = {"source_rows": len(records), "verified_from_source": 0,
              "skipped_unverified": 0, "added_elements": 0,
              "added_compounds": 0}
    with h5py.File(output, "w") as file:
        file.attrs["schema_name"] = "PyHREBSD elastic materials"
        file.attrs["schema_version"] = "1.0"
        file.attrs["stiffness_unit"] = "GPa"
        file.attrs["voigt_order"] = ",".join(VOIGT_LABELS)
        file.attrs["matrix_indexing"] = (
            "stiffness_voigt_gpa[row,column] uses zero-based array indices; "
            "see /voigt_axis_labels and /component_labels"
        )
        file.attrs["legacy_source"] = source.name
        file.attrs["notes"] = (
            "Elastic constants depend on temperature, phase, purity and measurement method. "
            "This distribution contains only records matched to the cited online NIST source."
        )
        materials = file.create_group("materials")
        string_type = h5py.string_dtype("utf-8")
        axis = file.create_dataset("voigt_axis_labels",
                                   data=np.asarray(VOIGT_LABELS, dtype=string_type))
        axis.attrs["description"] = (
            "Tensor-pair represented by each matrix row/column; array order is "
            "11, 22, 33, 23, 13, 12"
        )
        components = file.create_dataset(
            "component_labels", data=np.asarray(COMPONENT_LABELS, dtype=string_type))
        components.attrs["description"] = (
            "Human-readable label for every cell of each 6x6 stiffness_voigt_gpa matrix"
        )
        references = file.create_group("references")
        references.attrs["description"] = (
            "Bibliographic records referenced by /materials/<key>/reference_id"
        )
        for reference_id, record in REFERENCES.items():
            reference = references.create_group(reference_id)
            for field, value in record.items():
                reference.attrs[field] = value

        for record in records:
            preferred = None
            if record["name"] in NIST_CUBIC and record["crystal_system"].lower() == "cubic":
                preferred = cubic(*NIST_CUBIC[record["name"]])
            elif record["name"] in NIST_HEXAGONAL and record["crystal_system"].lower() == "hexagonal":
                preferred = hexagonal(*NIST_HEXAGONAL[record["name"]])
            if preferred is None:
                counts["skipped_unverified"] += 1
                continue
            base = slugify(record["name"])
            seen[base] = seen.get(base, 0) + 1
            slug = base if seen[base] == 1 else f"{base}_{seen[base]}"
            selected = write_record(
                materials, slug, record["name"], record["crystal_system"],
                preferred, "verified_reference", NIST_TITLE, NIST_URL, NIST_DOI,
                "Values matched to the cited NIST compilation; the source row is not used numerically.",
                record["source_line"], reference_id="ledbetter_kim_2001")
            delta = preferred - record["matrix"]
            selected.attrs["legacy_max_absolute_difference_gpa"] = float(
                np.max(np.abs(delta)))
            denominator = np.linalg.norm(record["matrix"])
            selected.attrs["legacy_relative_frobenius_difference"] = float(
                np.linalg.norm(delta) / denominator) if denominator else np.nan
            counts["verified_from_source"] += 1

        for slug, (name, system, constants) in NIST_ADDITIONS.items():
            if slug in materials:
                continue
            matrix = cubic(*constants) if system == "cubic" else hexagonal(*constants)
            write_record(materials, slug, name, system, matrix, "verified_reference",
                         NIST_TITLE, NIST_URL, NIST_DOI,
                         "Added from the cited NIST compilation; absent from the legacy table.",
                         reference_id="ledbetter_kim_2001")
            counts["added_elements"] += 1

        for slug, (name, system, constants, errors, title, url, doi,
                   temperature_description) in VERIFIED_COMPOUNDS.items():
            matrix = cubic(*constants) if system == "cubic" else hexagonal(*constants)
            write_record(
                materials, slug, name, system, matrix, "verified_reference",
                title, url, doi, "Experimentally measured phase-specific elastic constants.",
                temperature_description=temperature_description,
                uncertainty=uncertainty_matrix(system, errors),
                reference_id=REFERENCE_ID_BY_DOI[doi],
            )
            counts["added_compounds"] += 1

        names = sorted(materials.keys())
        file.create_dataset("index", data=np.asarray(names, dtype=h5py.string_dtype("utf-8")))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="legacy Materials.txt path")
    parser.add_argument("output", type=Path, help="output materials.h5 path")
    args = parser.parse_args()
    counts = build(args.source, args.output)
    print(f"Wrote {args.output}: {counts}")


if __name__ == "__main__":
    main()
