# GND estimates from PyHREBSD maps

`calculate_gnd.py` reads sample-frame rotation fields from `scan_results.csv`
and scan geometry from the source H5OINA file. It estimates geometrically
necessary dislocation density using an FCC `a/2<110>{111}` basis with 12 edge
and 6 unique screw types.

For a result directory containing both ROI and homography outputs, run:

```powershell
python calculate_gnd.py path/to/results --material nickel --poisson-ratio 0.31
```

Use `--strain-correction` to include the measurable in-plane elastic-strain
gradients. PyHREBSD sets unavailable depth derivatives to zero, so this remains
a 2-D approximation rather than a complete 3-D Nye tensor. Use
`--gradient-span N` to differentiate across N map steps. A wider span reduces
noise amplification while lowering spatial resolution.

The solver minimizes weighted L1 line energy while matching six partial Nye
observables. Edge dislocations use weight `1/(1-nu)` and screw dislocations
use weight 1. Multiple density vectors can satisfy the same observables;
per-system allocation depends on the assumed slip types, material constants,
and energy weights. Statistically stored dislocations are not measured.

Each method writes a GND result folder containing `.npz` arrays, summary JSON,
and total, edge, screw, and per-type maps. Boundary points that lack the
required finite-difference neighbors are stored as NaN.

Map step and density use SI units. Spatial differentiation strongly amplifies
rotation noise, particularly for nanometer-step scans. Treat raw GND values as
model-dependent estimates and validate them against the noise floor and known
microstructure.

The implementation follows Wilkinson and Randman, *Philosophical Magazine*
90 (2010), especially Eqs. 6-9 and 11, and Arsenlis and Parks,
*Acta Materialia* 47 (1999). Consult the original papers when interpreting the
partial Nye tensor and dislocation basis.
