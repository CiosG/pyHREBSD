# PyHREBSD

PyHREBSD is a Python research implementation of high-resolution electron
backscatter diffraction analysis. It reads Oxford Instruments `.h5oina`
files directly and provides two complementary registration methods:

- multi-ROI FFT cross-correlation with optional two-pass remapping;
- whole-pattern inverse-compositional homography fitting.

Both methods run on CPU. CUDA acceleration is available through CuPy for ROI
correlation, raw-pattern preprocessing, and homography fitting.

> **Status:** alpha research software. Validate pattern-center conventions,
> detector geometry, signs, and tensor frames against a known specimen before
> using results quantitatively.

## Features

- processed 8-bit and unprocessed 16-bit H5OINA pattern stacks;
- direct per-point PC from `EBSD/Data` or `Data Processing/Data`;
- optional pattern binning and raw-pattern background correction;
- CPU and CUDA implementations of ROI and homography registration;
- strain, stress, and lattice-rotation maps;
- grain segmentation, HR-KAM, and experimental GND analysis;
- shared-scale strain and rotation comparison figures.

## Installation

Python 3.10 or newer is required.

```powershell
git clone https://github.com/CiosG/pyHREBSD.git PyHREBSD
cd PyHREBSD
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[test]"
.venv\Scripts\python -m pytest
```

On Linux or macOS, activate the environment with
`source .venv/bin/activate` and use `python -m pip` in the same way.

GPU execution additionally requires an NVIDIA CUDA installation and the CuPy
package matching that CUDA version. For example, a CUDA 12 installation often
uses:

```powershell
python -m pip install cupy-cuda12x
```

Check the current CuPy installation guide for the package appropriate to your
driver and CUDA runtime. CPU mode never imports CuPy.

## First analysis

1. Copy `run_pyhrebsd.py` if you want to preserve a configuration for a
   particular experiment.
2. Set `h5oina_file`, `h5_pattern_type`, `h5_pc_source`, `material_name`, and
   `reference_map_point` in its `CONFIG` block.
3. Start with `analysis_method="roi"`, CPU devices, `workers=1`, and a short
   `scan_indices` list. Inspect the output before processing the entire map.
4. Select a new `output_dir` for every run; PyHREBSD refuses to silently
   overwrite completed analyses.

Run the configured analysis with:

```powershell
python run_pyhrebsd.py
```

The default configuration is intentionally portable: it expects
`scan.h5oina`, uses processed patterns, direct H5 PC values, the bundled
HDF5 silicon constants, ROI analysis, and CPU execution. Elastic
constants, their verification status, and literature references are stored in
`pyhrebsd/materials.h5`; see [the material database notes](docs/materials.md).

For a direct two-image ROI shift measurement:

```powershell
python -m pyhrebsd reference.tif scan.tif --roi-size 256 --roi-count 48 --output shifts.csv
```

## Outputs

Full-map analysis writes `scan_results.csv` and `run_settings.json` plus
optional per-pattern ROI data. Plotting utilities include:

```powershell
python plot_scan.py results_roi_cpu/scan_results.csv --output-dir results_roi_cpu/maps
python plot_tensor_maps.py results_roi_cpu/scan_results.csv --output-dir results_roi_cpu/tensor_maps
```

`calculate_hr_kam.py`, `segment_grains.py`, and `calculate_gnd.py` provide
post-processing. GND values are especially sensitive to step size, noise,
missing depth derivatives, and the selected dislocation basis; see
[the GND notes](docs/gnd.md).

## Method and validation notes

Strain and rotation from a real-pattern reference are relative to the selected
reference pattern. Results therefore contain any elastic state already present
at that point. Absolute pattern-center accuracy, detector orientation, sample
tilt, material constants, preprocessing, and reference selection all affect
the recovered tensor.

The detailed implementation notes are in
[docs/implementation.md](docs/implementation.md). They describe coordinate
conventions, PC handling, remapping, GPU paths, calibration, and numerical
options. The scientific basis and method-by-method provenance for ROI
correlation, remapping, and homography are listed in
[docs/references.md](docs/references.md).

## Origin and attribution

Some routines were developed using OpenXY as a source reference. Its license
permits modification and redistribution with attribution. The
exact origins and retained notices are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), with complete license texts
in `third_party_licenses/`.

The PyHREBSD name does not imply endorsement by OpenXY, Brigham Young
University, or AGH University of Krakow.

## License

Original PyHREBSD material is available under the
[PyHREBSD Source-Available No-Resale License 1.0](LICENSE). It permits free
internal, research, educational, and governmental use, including use by
commercial organizations. It also permits publication and sale of scientific
results and paid research performed with the software. Selling the software,
a modified version, or hosted access to substantial PyHREBSD functionality
requires prior written permission.

This restriction means PyHREBSD is source-available rather than OSI-approved
open-source software. Third-party components retain their own licenses.
