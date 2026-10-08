# PyHREBSD implementation notes

PyHREBSD measures the projective displacement between an EBSD reference
pattern and every pattern in a scan, then converts that displacement into a
deformation gradient, elastic strain, lattice rotation, and stress.

All map coordinates and pattern indices in the Python API are zero-based.
Configuration angles are in degrees. Pattern centers use fractions of the
uncropped detector width unless a function explicitly documents pixel units.

## H5OINA input

`H5OINAReader` reads one pattern at a time and does not load the entire stack
into memory. `pattern_type="processed"` selects the processed 8-bit stack;
`"unprocessed"` selects the raw stack. Rectangular patterns are cropped to a
centered square for correlation, and the pattern center is transformed to the
cropped detector coordinates. If a writer stores unsigned detector words in a
signed integer dataset, wrapped negative values are reinterpreted as unsigned
intensities without changing their bits.

Per-point pattern centers can be selected from either `/<scan>/EBSD/Data` or
`/<scan>/Data Processing/Data`. The selected source is recorded in the output.
`pc_mode="h5"` uses every stored PC directly. `pc_mode="affine"` robustly fits
a scan-position plane to the selected values. External beam-shift calibration
replaces only the calibrated PC gradient; it does not independently determine
the absolute PC.

When no explicit overrides are supplied, sample tilt and the full Oxford
detector orientation are read from the H5OINA header. Euler orientations use
the Bunge convention implemented in `pyhrebsd.geometry`.

## Thermo Fisher TFS HDF5 input

`TFSReader` reads xTalView `.tfs.hdf5` exports directly from the public
Thermo Fisher layout. Patterns are read lazily from
`/Site/EBSD/Patterns/Processed`; TFS exports currently provide processed
patterns only. Per-point pattern centers, Euler angles, phases, and indexing
quality come from `/Site/EBSD/MapData`. TFS map coordinates are stored as
`(row, column)` and are exposed as the common zero-based row-major index
`row * x_cells + column`.

TFS pattern centers use the xTalView normalized detector convention and are
transformed to the same centered-square coordinates as H5OINA. `SpecimenTilt`
is read in radians and reported in degrees. The detector-to-sample
transformation is reconstructed from `DCStoSCS` and specimen tilt, so
`detector_geometry="full"` can use the complete reconstructed orientation.
The `h5_pc_source` setting is ignored for TFS; TFS has one `MapData` PC source.

## EDAX OH5, ANG, and UP2 input

`OH5Reader` reads EDAX/OIM HDF5 pattern stacks directly. It maps the EDAX
pattern, Euler, phase, IQ, PC-calibration, and geometry fields to the common
reader API. `UP2Reader` reads the little-endian uint16 pattern records by
memory mapping the payload. An `.ang` input automatically uses the companion
`.up2`; an `.up2` may be used alone when map dimensions and geometry are
available in its header or configuration.

ANG Euler angles are interpreted as radians, while the numeric ANG metadata
for sample tilt, camera elevation, and pattern centre uses EDAX's degree and
normalized-detector conventions. The values are transformed to the centered
square pattern coordinates used by the analysis. Hexagonal maps use the same
non-interpolating even-row/even-column square subset as the conversion tools.
EDAX processed patterns use dynamic LMSD correction; unprocessed patterns
remain uint16 until the selected raw-pattern correction stage.

## Bruker BCF input

`BCFReader` performs read-only random access to the AidAim SFS container and
Bruker EBSD virtual files. It reads `FrameDescription` as the map-to-frame
index and uses its 64-bit virtual offsets to fetch individual records from
`FrameData`; the complete pattern stack is never loaded into memory. Both
8-bit and 16-bit pattern records are supported. Missing-frame markers remain
missing analysis points.

The reader converts Bruker's active Euler triplets to the passive Bunge
orientation used by the analysis. It reconstructs the affine per-point PC
from the calibration PC, working distance, phosphor size, SEM scan step,
camera and specimen tilts, and scan rotation. Detector geometry is exposed in
the same frame expected by the ROI and homography solvers.

For BCF, `pattern_type="processed"` means that dynamic-background LMSD is
computed in memory for each raw pattern. The default parameters are scaled by
the original detector width: Gaussian sigma `0.047 * width`, LMSD radius
`0.0375 * width`, truncated edge neighborhoods, and symmetric 0.75-percent
clipping to `uint8`. Cached edge-weight arrays avoid recomputing normalization
filters for every pattern. `bcf_preprocess_device` selects the NumPy/SciPy or
CuPy/CUDA implementation. CPU and GPU paths implement the same operations.

`pattern_type="unprocessed"` returns the original unsigned intensity values.
BCF does not normally contain a separate static-background image. When
`static_lmsd` is requested for raw BCF data, the runner automatically selects
the per-pattern dynamic LMSD correction.
The BCF reader is implemented directly in PyHREBSD and does not require an
intermediate conversion or Bruker software library.

### Unprocessed-pattern background correction

Background correction is applied only when
`pattern_type="unprocessed"`. Processed patterns bypass this entire stage.
The selected `unprocessed_background_mode` has the following behavior:

- `static_lmsd` uses the static detector background embedded at
  `/<scan>/EBSD/Header/Unprocessed Static Background`; if it is absent, the
  runner automatically switches to `dynamic_lmsd`;
- `dynamic_lmsd` estimates the Gaussian background and local normalization
  independently for every full detector pattern, using the same calculation
  as processed BCF input;
- `divide_gaussian` estimates a broad background independently for every
  pattern and divides the pattern by it;
- `subtract_gaussian` estimates a broad background independently for every
  pattern and subtracts it;
- `none` returns the unprocessed intensities without background correction.

For `static_lmsd`, the H5 dataset must be a single two-dimensional image with
the same full rectangular shape as every unprocessed detector pattern. It is
loaded once when an analysis context is created and reused for the reference
and all target patterns. Parallel worker processes each create their own
reader and load their own copy once. A missing dataset selects
`dynamic_lmsd`; an existing static background with an incompatible shape is
still reported as an error.

The `static_lmsd` pipeline is executed in this order:

1. Convert the full, uncropped raw pattern and static background to floating
   point. Replace zero background pixels by `1e-6` to avoid division by zero.
2. Divide the raw pattern by the static background.
3. Gaussian-smooth the divided image with
   `sigma = detector_width * unprocessed_static_sigma_factor` and subtract
   that smooth image.
4. Calculate the local mean and standard deviation in a square window with
   `radius = int(detector_width * unprocessed_lmsd_factor)` and side length
   `2 * radius + 1`. Replace each residual pixel by its local z-score;
   neighborhoods with standard deviation below `1e-8` produce zero.
5. Map the 1st and 99th percentiles of the normalized image to 0 and 65535,
   clip values outside that interval, and return `uint16`.
6. Center-crop a rectangular detector image to a square and then apply
   `pattern_binning` block averaging.

Thus the default factors `0.02` and `0.183` scale with the original,
unbinned detector width. For a 1024-pixel-wide pattern they give a Gaussian
sigma of 20.48 pixels and an LMSD radius of 187 pixels (a 375-pixel window).
For a 512-pixel-wide pattern they give 10.24 pixels and a radius of 93 pixels
(a 187-pixel window).

The parameters `unprocessed_background_sigma_pixels` and
`unprocessed_background_downsample` are ignored in `static_lmsd` mode. They
belong only to `divide_gaussian` and `subtract_gaussian`. In those modes,
PyHREBSD first obtains the centered square pattern, samples every
`unprocessed_background_downsample` pixel, applies a Gaussian whose sigma is
`unprocessed_background_sigma_pixels` in original-pattern pixel units,
interpolates the background to full size, performs division or subtraction,
and finally applies `pattern_binning`.

## ROI method

The ROI path extracts multiple square regions, applies an optional cosine and
radial-frequency window, and calculates translations by FFT
cross-correlation. Peak position is refined with either independent 1-D
parabolas or a coupled 2-D quadratic fit. This follows the local HR-EBSD
approach of Wilkinson, Meaden, and Dingley
[WMD2006](references.md#wmd2006).

The default annular layout places one ROI at the center and the remainder on
a circular ring. A regular square grid is also available. ROI size may be
given in pixels or as a percentage of the square pattern side.

With `roi_remapping=True`, a first pass estimates the finite lattice rotation.
The scan pattern is projectively back-rotated and correlated again. The final
deformation combines the finite first-pass rotation and the residual
second-pass fit. The physical basis for this two-pass correction is the
remapping method of Britton and Wilkinson [BW2012](references.md#bw2012).

CPU mode uses NumPy FFTs. GPU mode uses CuPy for remapping, ROI FFTs,
cross-correlations, and subpixel peak fitting. The small tensor solve remains
on the CPU.

## Homography method

The homography path fits one eight-parameter projective transformation over a
large central region. It uses high-pass preprocessing, zero-mean normalized
intensities, and inverse-compositional Gauss-Newton iterations. The reference
gradient and Jacobian are cached for the entire scan. The HR-EBSD formulation
follows Ernould et al. [E2020](references.md#e2020),
[E2022a](references.md#e2022a), and [E2022b](references.md#e2022b), and the
underlying IC-GN algorithm follows Baker and Matthews
[BM2004](references.md#bm2004).

The initial warp includes the geometric change implied by the reference and
target pattern centers. The optimizer stops at convergence or at
`homography_max_iterations`. Output includes the 3x3 homography, ZNSSD
residual, iteration count, and convergence flag.

Homography cannot determine isotropic dilation from one projective image pair.
PyHREBSD fixes the remaining scalar using the zero-normal-stress free-surface
condition and the selected elastic constants. Hydrostatic strain is therefore
partly constrained by this boundary condition rather than measured directly.

CPU mode uses NumPy and SciPy. GPU mode keeps interpolation, residuals, the
Jacobian, and parameter updates in CuPy float64. Final tensor conversion runs
on the CPU.

## Pattern binning

`pattern_binning` performs block averaging before either analysis method and
reduces the pattern dimensions by the same factor. Homography evaluates every
pixel remaining inside its configured margin. Compare binned and unbinned
results before selecting production settings.

## Frames and tensor output

The deformation solve first produces crystal-frame tensors. The output CSV
also stores strain, stress, deviatoric strain, and rotation in the sample
frame. Plotting utilities can apply an additional in-plane sample-axis basis
rotation for comparison with external figures.

Real-pattern strain and rotation are relative to the selected reference
pattern. A strained or rotated reference shifts the zero of every map.

## Calibration and validation

Pattern-center calibration can estimate an effective beam-shift pixel size on
a separate strain-free single-crystal scan. Use the same detector resolution,
geometry, SEM conditions, and scan convention for calibration and analysis.
The calibration uses every pattern in the complete map row containing
`reference_map_point`; a reference at `(0, 0)` therefore uses the first row.

For a new instrument or acquisition recipe:

1. verify processed and raw pattern orientation;
2. confirm the selected PC source and detector Euler convention;
3. validate tensor signs and axes on a known deformation;
4. compare CPU and GPU results on a small point set;
5. compare binned and unbinned maps using common masks and color scales;
6. keep a copy of the configured `run_pyhrebsd.py` with each result CSV.

## Scientific origins

The ROI correlation and deformation path was developed using OpenXY routines
as source references. The whole-pattern method follows published global-DIC
and homography HR-EBSD methods. See `THIRD_PARTY_NOTICES.md` for software
attribution and retained license terms. Full method-to-source mapping and
bibliographic details are in [`references.md`](references.md).

Reference keys in this page are defined in [`references.md`](references.md).
