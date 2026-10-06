# Changelog

## Unreleased

- Added direct, lazy Bruker ESPRIT BCF input for ROI and homography analysis.
- Added CPU and CUDA dynamic-LMSD processing of raw BCF patterns and
  unprocessed H5OINA patterns without a stored static background.
- Added BCF acquisition-geometry, per-point PC, indexing, and grain metadata.
- Interpret wrapped signed H5OINA raw-pattern storage as unsigned detector
  intensities.
- Added sparse-BCF handling that skips missing map slots and can select the
  first stored pattern automatically as the reference.
- Replaced the primary configuration keys with format-neutral `input_file` and
  `pattern_type`; legacy H5OINA keys remain accepted by the runner.

## 0.1.0 - 2026-10-05

- Initial source-available research release.
- Direct processed and unprocessed pattern reading from Oxford H5OINA files.
- ROI cross-correlation with optional remapping on CPU or CUDA.
- Whole-pattern homography fitting on CPU or CUDA.
- Per-point H5 pattern-center support, grain segmentation, HR-KAM, and
  experimental GND analysis.
- Shared-scale strain, stress, rotation, and quality maps.
