# Scientific references and implementation provenance

This page identifies the scientific basis and the software provenance of the
three registration paths used by PyHREBSD. Citation of a paper below does not
mean that PyHREBSD reproduces every part of that paper's algorithm.

## ROI cross-correlation

The scientific basis is the local HR-EBSD method of Wilkinson, Meaden, and
Dingley [WMD2006](#wmd2006). Square subregions distributed across an EBSD pattern are
matched by FFT cross-correlation, the correlation peaks are located with
subpixel precision, and the measured shifts are fitted to the deformation
gradient. The finite-deformation equations and traction-free surface
constraint also follow the framework discussed by Britton and Wilkinson
[BW2012](#bw2012).

The Python implementation was developed using the OpenXY routines `GetROIs`,
`custfftxc`, `subpixshift`, and `CalcFShift` as source references. PyHREBSD
adds an optional center-plus-ring ROI layout, ROI size expressed as a fraction
of pattern width, its own coupled two-dimensional quadratic peak fit, CPU/GPU
batching, H5OINA input, and explicit per-point pattern-center handling. See
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for the applicable
software licenses.

## Two-pass remapping

The purpose and physical basis of remapping follow Britton and Wilkinson
[BW2012](#bw2012). A first ROI pass estimates the finite lattice rotation. PyHREBSD then
projects detector rays through the pattern center, applies the inverse of that
rotation, resamples the target pattern on the reference detector grid, and
performs a second ROI pass. The final result combines the finite first-pass
rotation with the residual deformation measured after remapping.

This is an independent Python implementation of the published method, informed
by the OpenXY remapping workflow. Cubic interpolation, invalid-pixel masking,
GPU execution, and the exact composition used by PyHREBSD are documented in
the source.

## Whole-pattern homography

The homography path follows the global DIC formulation introduced by Ernould
et al. [E2020](#e2020) and described in detail in [E2022a](#e2022a) and
[E2022b](#e2022b). One large
central image region is represented by an eight-parameter first-order
homography and registered with an inverse-compositional Gauss-Newton (IC-GN)
iteration. The general IC-GN update is based on the framework of Baker and
Matthews [BM2004](#bm2004). Pattern-center geometry is applied when the fitted
homography is converted to the relative deformation gradient; the missing
hydrostatic degree of freedom is fixed with the traction-free surface
condition.

PyHREBSD currently initializes the warp from the reference-to-target
pattern-center change. It does **not** implement the Fourier-Mellin rotation
initialization or the integrated optical-distortion correction described in
the complete Ernould workflow. Its preprocessing, convergence tests,
binning option, and CUDA implementation are PyHREBSD implementation choices.

## Bibliography

### WMD2006

A. J. Wilkinson, G. Meaden, and D. J. Dingley,
“High-resolution elastic strain measurement from electron backscatter
diffraction patterns: New levels of sensitivity,” *Ultramicroscopy* 106
(2006) 307–313. [doi:10.1016/j.ultramic.2005.10.001](https://doi.org/10.1016/j.ultramic.2005.10.001).

### BW2012

T. B. Britton and A. J. Wilkinson, “High resolution electron
backscatter diffraction measurements of elastic strain variations in the
presence of larger lattice rotations,” *Ultramicroscopy* 114 (2012) 82–95.
[doi:10.1016/j.ultramic.2012.01.004](https://doi.org/10.1016/j.ultramic.2012.01.004).

### E2020

C. Ernould, B. Beausir, J.-J. Fundenberger, V. Taupin, and
E. Bouzy, “Global DIC approach guided by a cross-correlation based initial
guess for HR-EBSD and on-axis HR-TKD,” *Acta Materialia* 191 (2020) 131–148.
[doi:10.1016/j.actamat.2020.03.026](https://doi.org/10.1016/j.actamat.2020.03.026).

### E2022a

C. Ernould, B. Beausir, J.-J. Fundenberger, V. Taupin, and
E. Bouzy, “Development of a homography-based global DIC approach for
high-angular resolution in the SEM,” *Advances in Imaging and Electron
Physics* 223 (2022) 49–73.
[doi:10.1016/bs.aiep.2022.07.002](https://doi.org/10.1016/bs.aiep.2022.07.002).

### E2022b

C. Ernould, B. Beausir, J.-J. Fundenberger, V. Taupin, and
E. Bouzy, “Implementing the homography-based global HR-EBSD/TKD approach,”
*Advances in Imaging and Electron Physics* 223 (2022) 75–114.
[doi:10.1016/bs.aiep.2022.07.003](https://doi.org/10.1016/bs.aiep.2022.07.003).

### BM2004

S. Baker and I. Matthews, “Lucas-Kanade 20 Years On: A Unifying
Framework,” *International Journal of Computer Vision* 56 (2004) 221–255.
[doi:10.1023/B:VISI.0000011205.11775.fd](https://doi.org/10.1023/B:VISI.0000011205.11775.fd).
