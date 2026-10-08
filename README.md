# PyHREBSD

PyHREBSD is a Python research implementation of high-resolution electron
backscatter diffraction analysis. It reads Thermo Fisher `.tfs.hdf5`, Oxford
Instruments `.h5oina`, EDAX `.oh5`, `.ang` + `.up2`, and Bruker ESPRIT
`.bcf` files directly and provides two complementary registration methods:

- multi-ROI FFT cross-correlation with optional two-pass remapping;
- whole-pattern inverse-compositional homography fitting.

Both methods run on CPU. CUDA acceleration is available through CuPy for ROI
correlation, raw-pattern preprocessing, and homography fitting.

> **Status:** alpha research software. Validate pattern-center conventions,
> detector geometry, signs, and tensor frames against a known specimen before
> using results quantitatively.

## Features

- processed 8-bit and unprocessed 16-bit H5OINA pattern stacks;
- direct, lazy access to raw 8-/16-bit BCF patterns, indexing results,
  acquisition geometry, and reconstructed per-point pattern centres;
- direct per-point PC from `EBSD/Data` or `Data Processing/Data`;
- full Oxford detector geometry from the three detector Euler angles;
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

1. Open `run_pyhrebsd.py` and edit the required settings at the top of its
   `CONFIG` block.
2. Set `input_file` to a `.tfs.hdf5`, `.h5oina`, `.oh5`, `.ang`, `.up2`, or `.bcf` file, then set a new `output_dir`,
   `analysis_method`, `pattern_type`, `pc_mode`, `material_name`,
   `reference_map_point`, and `pattern_binning`.
3. Leave geometry overrides as `None` to read sample tilt, reference
   orientation, and detector orientation from the dataset. The default
   `detector_geometry="full"` uses all three detector Euler angles for H5OINA
   and TFS when available. BCF provides elevation geometry only.
4. Select `"cpu"` or `"gpu"` separately for the chosen registration method.
   CPU analysis defaults to the number of logical processors minus one;
   set `workers=1` when validating a new configuration.

Run the configured analysis with:

```powershell
python run_pyhrebsd.py
```

The default configuration expects `scan.h5oina`, uses processed patterns,
the dataset PC values, the bundled HDF5 silicon constants, ROI analysis, and CPU
execution. It analyzes the complete map, including the reference point.
Elastic constants, their verification status, and literature references are
stored in `pyhrebsd/materials.h5`; see
[the material database notes](docs/materials.md).

For TFS input, `pattern_type="processed"` is selected automatically; TFS exports
contain processed patterns, map PC values, Euler angles, phases, and sample geometry.

For `pattern_type="unprocessed"`, H5OINA uses its static background when
available and otherwise falls back to per-pattern `dynamic_lmsd`; BCF uses the
same dynamic fallback because it has no standalone static image. TFS exports
currently provide processed patterns only. Set `unprocessed_preprocess_device`
to `"cpu"` or `"gpu"`.

## EDAX OH5, ANG, and UP2 input

Set `input_file` to an EDAX `.oh5` file to read its patterns, orientations,
pattern quality, PC calibration, and acquisition geometry directly. For an
`.ang` file, PyHREBSD automatically opens the same-name `.up2` file. A `.up2`
file can also be selected directly; its v3 header supplies the map dimensions,
step size, and square/hexagonal grid. For UP2 v1 files, set
`up2_map_width`, `up2_map_height`, `up2_step_x`, `up2_step_y`, and optionally
`up2_grid` in `CONFIG`.

The `.ang` file supplies Euler angles, phase IDs, IQ, pattern centre, sample
tilt, and detector elevation. A standalone `.up2` has pattern data only, so
set `pattern_center_fallback`, `sample_tilt_degrees`, and
`camera_elevation_degrees` when those values are unavailable. EDAX processed
mode applies dynamic LMSD correction in memory; unprocessed mode keeps the
original uint16 detector values and uses the selected raw-pattern correction.
Hexagonal EDAX maps are exposed as the square-grid subset used by the supplied
converters, without interpolating patterns.

## Beam-shift calibration

Use a separate strain-free single-crystal scan acquired with the same detector
resolution, geometry, SEM conditions, and scan convention as the analysis.
Set the common input-source settings once and change:

```python
"run_mode": "calibration",
"input_file": r"path\to\calibration.h5oina",
"output_dir": "beam_shift_calibration",
"reference_map_point": (0, 0),
"calibration_method": "roi",  # or "homography"
```

Calibration correlates every point in the complete map row containing the
reference point. With `reference_map_point=(0, 0)` this is the entire first
row. The row must contain at least seven points. The output JSON reports
`effective_pixel_size_um_per_pixel` and `detector_x_shift_sign`.

The `roi` method measures the displacement of a PC-centred subset. The
`homography` method registers the whole pattern and evaluates the fitted
projective warp at the PC. It uses `homography_device`,
`homography_gpu_device_id`, `homography_margin_fraction`, and
`homography_max_iterations`. Optional `calibration_line_extent_um` and
`calibration_line_spacing` restrict or subsample the selected row.

For the subsequent analysis, select the corresponding H5OINA, TFS, or BCF
file and set:

```python
"run_mode": "analysis",
"pc_mode": "beam_shift_eps",
"beam_shift_effective_pixel_size_um": 4.123,  # value from calibration JSON
"beam_shift_detector_x_sign": -1,             # value from calibration JSON
```

The external effective pixel size replaces the scan-X PC drift. The absolute
PC and the remaining PC gradients still come from the selected input source.

## Bruker BCF input

Set `input_file` to the `.bcf` file. With `pattern_type="processed"`, PyHREBSD
reads each raw pattern lazily and applies dynamic-background LMSD correction in
memory; no intermediate H5OINA file is written. Select CPU or CUDA for this
step with `bcf_preprocess_device`. With `pattern_type="unprocessed"`, the raw
8- or 16-bit pattern is passed to the general raw-pattern correction settings.
If `static_lmsd` is selected but no static-background image exists, PyHREBSD
automatically uses the same per-pattern dynamic LMSD correction.

BCF pattern centres are reconstructed for every map point from the stored PC,
working distance, phosphor size, scan calibration, detector tilt, specimen
tilt, and scan rotation. `h5_pc_source` is ignored for BCF. Missing map slots
are skipped automatically rather than replaced with zero-valued patterns.
Set `reference_map_point=None` to use the first stored pattern, which is useful
when a sparse BCF acquisition starts away from map coordinate `(0, 0)`.
The reader is implemented directly in PyHREBSD and does not require a format
converter or Bruker software library.

For a direct two-image ROI shift measurement:

```powershell
python -m pyhrebsd reference.tif scan.tif --roi-size 256 --roi-count 48 --output shifts.csv
```

## Outputs

Full-map analysis writes `scan_results.csv` plus optional per-pattern ROI
data. Beam-shift calibration writes `beam_shift_calibration.json`,
`beam_shift_line.csv`, and `beam_shift_calibration.png`. Plotting utilities
include:

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
permits modification and redistribution with attribution. The exact origins
and retained notices are documented in
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
