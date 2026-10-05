# Elastic-material database

PyHREBSD stores single-crystal stiffness tensors in `pyhrebsd/materials.h5`.
Each `/materials/<key>/stiffness_voigt_gpa` dataset is a symmetric 6 by 6
matrix in GPa, using Voigt order `11,22,33,23,13,12`. The complete matrices
allow the stress calculation to support cubic, hexagonal, tetragonal,
trigonal, orthorhombic, and monoclinic materials without reducing them to
three elastic constants.

The root datasets `/voigt_axis_labels` and `/component_labels` describe every
matrix position. `/component_labels` has the same 6 by 6 shape and contains
`C11` through `C66`; for example, matrix cells `[0,0]`, `[0,1]`, `[3,3]`, and
`[5,5]` are respectively `C11`, `C12`, `C44`, and `C66`. The same row and
column labels are also stored as attributes on every stiffness dataset.

Select a material in `run_pyhrebsd.py` with:

```python
"material_database": "pyhrebsd/materials.h5",
"material_name": "silicon",
```

The key or the human-readable material name may be used. List keys with
`Material.list_hdf5("pyhrebsd/materials.h5")`.

## Provenance and verification

The distributed database deliberately excludes 101 rows from the supplied
`Materials.txt` whose individual provenance could not be verified online. It
contains 33 source materials matched unambiguously to NIST, six additional
NIST elements, and five phase-specific experimental semiconductor records:
wurtzite GaN, wurtzite AlN, 3C-SiC, 4H-SiC, and 6H-SiC. There is no copy of the
omitted matrices under a legacy group.

Preferred elemental values that could be matched unambiguously were checked
against Tables 7.1 and 7.2 of:

H. M. Ledbetter and S. D. Kim, *Monocrystal Elastic Constants and Derived
Properties of the Cubic and the Hexagonal Elements*, Handbook of Elastic
Properties of Solids, Liquids, and Gases, Vol. II (2001),
doi:10.1016/B978-012445760-7/50033-2. The publication is available from
[NIST](https://www.nist.gov/publications/monocrystal-elastic-constants-and-deprived-properties-cubic-and-hexagonal-elements).

All records have status `verified_reference`. Measurement uncertainties are
stored in `stiffness_uncertainty_gpa` beside the stiffness matrix when they are
reported by the source.

Complete bibliographic records are stored under `/references/<reference_id>`.
Each material contains `reference_id` and `reference_path` attributes pointing
to the applicable record. The title, DOI, and URL are repeated on the material
group so a single group remains self-describing when exported.

Every record stores `verification_status`, `reference_title`, `reference_url`,
`reference_doi`, and `temperature_k` attributes. `temperature_k` is NaN where
the cited compilation or legacy row does not identify a single measurement
temperature. Elastic constants vary with temperature, phase, purity, magnetic
state, and whether a measurement is adiabatic or isothermal, so these metadata
are part of the material definition.

A flat, reviewable inventory of every current record is provided in
[`materials_audit.csv`](materials_audit.csv).

Rebuild the database after changing the source table with:

```powershell
python tools/build_materials_h5.py ..\Materials.txt pyhrebsd\materials.h5
```
