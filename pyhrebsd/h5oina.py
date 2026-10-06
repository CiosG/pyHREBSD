"""Lazy reader for Oxford Instruments H5OINA EBSD pattern datasets.

Oxford's format stores patterns as (point, height, width) under
``/<scan>/EBSD/Data``. Indices in this API are zero-based and follow the
dataset order; map point (column, row) is ``row * x_cells + column``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .geometry import euler_to_matrix, phosphor_to_sample_from_oxford


PATTERN_DATASETS = {
    "processed": "Processed Patterns",
    "unprocessed": "Unprocessed Patterns",
}

PC_SOURCES = {
    "ebsd": "EBSD/Data",
    "data_processing": "Data Processing/Data",
}

PC_DATASETS = ("Pattern Center X", "Pattern Center Y", "Detector Distance")


class H5OINAReader:
    """Read one pattern at a time without loading the HDF5 stack into RAM."""

    def __init__(self, path: str | Path, scan_group: str | None = None,
                 pattern_type: str = "processed", pc_source: str = "ebsd") -> None:
        self.path = Path(path)
        self._file = h5py.File(self.path, "r")
        try:
            self.scan_group = self._choose_scan_group(scan_group)
            data = self._file[f"{self.scan_group}/EBSD/Data"]
            if pattern_type == "auto":
                names = ("Processed Patterns", "Unprocessed Patterns")
            elif pattern_type in PATTERN_DATASETS:
                names = (PATTERN_DATASETS[pattern_type],)
            else:
                raise ValueError("pattern_type must be 'processed', 'unprocessed', or 'auto'")
            selected = next((name for name in names if name in data), None)
            if selected is None:
                raise KeyError(f"No {pattern_type} pattern dataset in /{self.scan_group}/EBSD/Data; "
                               f"available datasets: {list(data.keys())}")
            self.pattern_type = "processed" if selected == "Processed Patterns" else "unprocessed"
            self._patterns = data[selected]
            if self._patterns.ndim != 3:
                raise ValueError(f"/{self.scan_group}/EBSD/Data/{selected} must have shape "
                                 "(points, height, width)")
            self.count, self.height, self.width = map(int, self._patterns.shape)
            self._data = data
            if pc_source not in PC_SOURCES:
                raise ValueError("pc_source must be 'ebsd' or 'data_processing'")
            self.pc_source = pc_source
            self.pc_source_path = f"/{self.scan_group}/{PC_SOURCES[pc_source]}"
            self._pc_data = self._file.get(self.pc_source_path)
            if pc_source == "data_processing":
                if not isinstance(self._pc_data, h5py.Group):
                    raise KeyError(f"PC source {self.pc_source_path} is missing")
                missing = [name for name in PC_DATASETS if name not in self._pc_data]
                if missing:
                    raise KeyError(f"PC source {self.pc_source_path} lacks {missing}")
            if self._pc_data is not None and all(name in self._pc_data for name in PC_DATASETS):
                for name in PC_DATASETS:
                    if self._pc_data[name].shape[0] != self.count:
                        raise ValueError(f"{self.pc_source_path}/{name} must have one value per pattern")
            header = self._file.get(f"{self.scan_group}/EBSD/Header")
            self._header = header
            self.x_cells = self._scalar(header, "X Cells")
            self.y_cells = self._scalar(header, "Y Cells")
        except Exception:
            self._file.close()
            raise

    def _choose_scan_group(self, name: str | None) -> str:
        groups = sorted(key for key, obj in self._file.items()
                        if isinstance(obj, h5py.Group) and "EBSD/Data" in obj)
        if name is not None:
            name = name.strip("/")
            if name not in groups:
                raise KeyError(f"scan group {name!r} not found; EBSD groups: {groups}")
            return name
        if len(groups) != 1:
            raise ValueError(f"expected one EBSD scan group, found {groups}; set scan_group")
        return groups[0]

    @staticmethod
    def _scalar(group: h5py.Group | None, name: str) -> int | None:
        if group is None or name not in group:
            return None
        values = np.asarray(group[name][()]).reshape(-1)
        return int(values[0]) if len(values) else None

    def _check_index(self, index: int) -> None:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError("pattern index must be an integer")
        if index < 0 or index >= self.count:
            raise IndexError(f"pattern index {index} outside 0..{self.count - 1}")

    def available_indices(self) -> np.ndarray:
        """Return all pattern indices; H5OINA pattern stacks are dense."""
        return np.arange(self.count, dtype=np.int64)

    def has_pattern(self, index: int) -> bool:
        """Return whether the dense H5OINA stack contains this index."""
        return (not isinstance(index, (bool, np.bool_)) and
                isinstance(index, (int, np.integer)) and 0 <= index < self.count)

    def pattern(self, index: int) -> np.ndarray:
        """Return a centered square crop of one pattern as float64."""
        image = self.uncropped_pattern(index)
        if self.width > self.height:
            left = (self.width - self.height) // 2
            image = image[:, left : left + self.height]
        elif self.height > self.width:
            top = (self.height - self.width) // 2
            image = image[top : top + self.width, :]
        return image

    def uncropped_pattern(self, index: int) -> np.ndarray:
        """Return the full rectangular detector image before analysis cropping."""
        self._check_index(index)
        return np.asarray(self._patterns[index], dtype=np.float64)

    def unprocessed_static_background(self) -> np.ndarray:
        """Return the full detector static background embedded in H5OINA."""
        if self._header is None or "Unprocessed Static Background" not in self._header:
            raise KeyError("H5OINA lacks EBSD/Header/Unprocessed Static Background")
        background = np.asarray(self._header["Unprocessed Static Background"][()])
        if background.shape != (self.height, self.width):
            raise ValueError("unprocessed static background has wrong shape")
        return background

    def pattern_center(self, index: int) -> tuple[float, float, float] | None:
        """Return (x*, y*, z*) adjusted to the cropped square pattern.

        H5OINA scales all three coordinates to the *original width*;
        y* has a bottom-left origin. A missing component returns ``None``
        so the caller can supply a calibrated center in the config block.
        """
        self._check_index(index)
        if self._pc_data is None or any(name not in self._pc_data for name in PC_DATASETS):
            return None
        values = []
        for name in PC_DATASETS:
            value = np.asarray(self._pc_data[name][index]).reshape(-1)
            if len(value) != 1 or not np.isfinite(value[0]):
                return None
            values.append(float(value[0]))
        x, y, z = values
        size = min(self.width, self.height)
        left = (self.width - size) // 2
        bottom = self.height - size - (self.height - size) // 2
        scale = self.width / size
        return ((x * self.width - left) / size,
                (y * self.width - bottom) / size,
                z * scale)

    def pattern_centers(self) -> np.ndarray | None:
        """Read all per-point centers, adjusted to the centered square crop."""
        if self._pc_data is None or any(name not in self._pc_data for name in PC_DATASETS):
            return None
        centers = np.column_stack([np.asarray(self._pc_data[name][:], dtype=np.float64).reshape(-1)
                                   for name in PC_DATASETS])
        if centers.shape != (self.count, 3):
            raise ValueError("pattern center datasets must have one value per pattern")
        size = min(self.width, self.height)
        left = (self.width - size) // 2
        bottom = self.height - size - (self.height - size) // 2
        centers[:, 0] = (centers[:, 0] * self.width - left) / size
        centers[:, 1] = (centers[:, 1] * self.width - bottom) / size
        centers[:, 2] *= self.width / size
        return centers

    def sample_tilt_degrees(self) -> float | None:
        """Oxford Tilt Angle is stored in radians."""
        if self._header is None or "Tilt Angle" not in self._header:
            return None
        values = np.asarray(self._header["Tilt Angle"][()], dtype=float).reshape(-1)
        if len(values) != 1 or not np.isfinite(values[0]):
            raise ValueError("invalid H5OINA Tilt Angle")
        return float(np.rad2deg(values[0]))

    def camera_elevation_degrees(self) -> float | None:
        """Oxford's second detector Bunge angle is 90 degrees plus elevation."""
        if self._header is None or "Detector Orientation Euler" not in self._header:
            return None
        values = np.asarray(self._header["Detector Orientation Euler"][()], dtype=float).reshape(-1)
        if len(values) != 3 or not np.all(np.isfinite(values)):
            raise ValueError("invalid H5OINA Detector Orientation Euler")
        return float(np.rad2deg(values[1]) - 90.0)

    def orientation(self, index: int) -> np.ndarray:
        """Return the sample-to-crystal rotation for an indexed point.

        H5OINA Euler angles use the same Bunge sample-to-crystal convention
        as OpenXY's ``euler2gmat``.
        """
        self._check_index(index)
        if "Euler" not in self._data:
            raise KeyError(f"/{self.scan_group}/EBSD/Data/Euler is missing")
        angles = np.asarray(self._data["Euler"][index], dtype=np.float64).reshape(-1)
        if len(angles) != 3 or not np.all(np.isfinite(angles)):
            raise ValueError(f"invalid Euler angles at pattern index {index}")
        return euler_to_matrix(*angles)

    def grain_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return Euler radians, phase IDs, and optional band contrast for the map.

        Phase zero denotes an unindexed point in Oxford's EBSD data.
        """
        if self.x_cells is None or self.y_cells is None or self.x_cells * self.y_cells != self.count:
            raise ValueError("grain segmentation needs a complete rectangular EBSD map")
        if "Euler" not in self._data or "Phase" not in self._data:
            raise KeyError("grain segmentation needs EBSD/Data/Euler and Phase")
        eulers = np.asarray(self._data["Euler"][:], dtype=np.float64)
        phases = np.asarray(self._data["Phase"][:], dtype=np.int32).reshape(-1)
        if eulers.shape != (self.count, 3) or phases.shape != (self.count,):
            raise ValueError("Euler and Phase must have one entry per pattern")
        contrast = (np.asarray(self._data["Band Contrast"][:], dtype=np.float64).reshape(-1)
                    if "Band Contrast" in self._data else None)
        if contrast is not None and contrast.shape != (self.count,):
            raise ValueError("Band Contrast must have one entry per pattern")
        return eulers, phases, contrast

    def phosphor_to_sample(self, sample_tilt_degrees: float | None = None) -> np.ndarray | None:
        """Read full detector Euler geometry, optionally overriding sample tilt."""
        if self._header is None or "Detector Orientation Euler" not in self._header:
            return None
        tilt = (self.sample_tilt_degrees() if sample_tilt_degrees is None
                else float(sample_tilt_degrees))
        if tilt is None:
            return None
        values = np.asarray(self._header["Detector Orientation Euler"][()], dtype=float).reshape(-1)
        if len(values) != 3 or not np.all(np.isfinite(values)):
            raise ValueError("invalid H5OINA Detector Orientation Euler")
        return phosphor_to_sample_from_oxford(np.deg2rad(tilt), values)

    def map_index(self, column: int, row: int) -> int:
        """Convert zero-based map coordinates to a pattern index."""
        if self.x_cells is None or self.y_cells is None:
            raise ValueError("X Cells and Y Cells are missing from the EBSD header")
        if column < 0 or row < 0 or column >= self.x_cells or row >= self.y_cells:
            raise IndexError("map coordinate outside EBSD scan")
        index = row * self.x_cells + column
        self._check_index(index)
        return index

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> H5OINAReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
