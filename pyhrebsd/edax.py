"""Lazy readers for EDAX/OIM OH5, ANG and UP2 EBSD data."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import struct

import h5py
import numpy as np

from .bcf import correct_pattern_dynamic_lmsd
from .geometry import euler_to_matrix, phosphor_to_sample_from_oxford


@dataclass
class _AngPhase:
    values: dict[str, str] = field(default_factory=dict)


@dataclass
class _AngData:
    values: dict[str, str]
    columns: list[str]
    data: np.ndarray
    phases: list[_AngPhase]


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _parse_ang(path: Path) -> _AngData:
    values: dict[str, str] = {}
    columns: list[str] = []
    phases: list[_AngPhase] = []
    current: _AngPhase | None = None
    with path.open("r", encoding="latin-1") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            phase = re.fullmatch(r"phase\s+(\d+)", body, re.IGNORECASE)
            if phase:
                current = _AngPhase()
                phases.append(current)
                continue
            if ":" in body:
                key, value = body.split(":", 1)
            else:
                parts = body.split(None, 1)
                key, value = parts[0], parts[1] if len(parts) > 1 else ""
            key, value = key.strip(), value.strip()
            normalized = _norm(key)
            if normalized == "columnheaders":
                columns = [item.strip() for item in value.split(",")]
            elif current is not None and normalized in {
                    "materialname", "formula", "symmetry", "pointgroupid",
                    "latticeconstants", "numberfamilies"}:
                current.values[normalized] = value
            else:
                values[normalized] = value
    try:
        data = np.loadtxt(path, comments="#", dtype=np.float64, ndmin=2)
    except ValueError as exc:
        raise ValueError(f"cannot parse ANG numeric table: {path}") from exc
    if not columns:
        defaults = ["phi1", "PHI", "phi2", "x", "y", "IQ", "CI",
                    "Phase index", "SEM", "Fit"]
        columns = defaults[:data.shape[1]]
    if len(columns) != data.shape[1]:
        raise ValueError(f"ANG column count does not match numeric table: {path}")
    return _AngData(values, columns, data, phases)


def _ang_value(ang: _AngData | None, *keys: str, default=None, cast=float):
    if ang is None:
        return default
    values = ang.values
    for key in keys:
        value = values.get(_norm(key))
        if value is not None:
            try:
                return cast(value.split()[0])
            except (ValueError, IndexError):
                return default
    return default


def _column(ang: _AngData | None, *names: str) -> int | None:
    if ang is None:
        return None
    lookup = {_norm(name): i for i, name in enumerate(ang.columns)}
    for name in names:
        if _norm(name) in lookup:
            return lookup[_norm(name)]
    return None


def _scalar(group, names: tuple[str, ...], default=None):
    for name in names:
        if group is not None and name in group:
            value = np.asarray(group[name][()]).reshape(-1)
            if len(value):
                item = value[0]
                return item.item() if hasattr(item, "item") else item
    return default


def _source_oh5(file: h5py.File):
    groups = [obj for obj in file.values()
              if isinstance(obj, h5py.Group) and "EBSD" in obj and
              "Data" in obj["EBSD"] and "Header" in obj["EBSD"]]
    if len(groups) != 1:
        raise ValueError(f"expected one EDAX EBSD group in OH5, found {len(groups)}")
    ebsd = groups[0]["EBSD"]
    return ebsd["Data"], ebsd["Header"]


def _oh5_layout(data, header):
    patterns = data.get("Pattern")
    if patterns is None or patterns.ndim != 3:
        raise ValueError("OH5 EBSD/Data/Pattern must have shape (points, height, width)")
    nx = int(_scalar(header, ("nColumns",), 0))
    ny = int(_scalar(header, ("nRows",), 0))
    if nx <= 0 or ny <= 0:
        raise ValueError("OH5 header lacks positive nColumns/nRows")
    grid = str(_scalar(header, ("Grid Type",), "SqrGrid"))
    if grid == "HexGrid":
        rows = np.arange(0, ny, 2, dtype=np.int64)
        columns = np.arange(0, nx, 2, dtype=np.int64)
        pattern_indices = np.concatenate([
            row * nx - row // 2 + columns for row in rows])
        expected = nx * ny - ny // 2
        if patterns.shape[0] < expected:
            raise ValueError("OH5 pattern dataset is shorter than its HexGrid map")
        return len(columns), len(rows), 2.0 * float(_scalar(header, ("Step X",), 1.0)), \
            2.0 * float(_scalar(header, ("Step X",), 1.0)), pattern_indices
    count = nx * ny
    if patterns.shape[0] < count:
        raise ValueError("OH5 pattern dataset is shorter than its map")
    return nx, ny, float(_scalar(header, ("Step X",), 1.0)), \
        float(_scalar(header, ("Step Y",), 1.0)), np.arange(count, dtype=np.int64)


def _crop(image):
    height, width = image.shape
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    return image[top:top + side, left:left + side]


class _EDAXReaderBase:
    source_format = "edax"

    def _init_common(self, path, pattern_type, processing_device, gpu_device_id,
                     lmsd_sigma_factor, lmsd_radius_factor, lmsd_edge_mode,
                     lmsd_clip_percentile):
        if pattern_type not in ("processed", "unprocessed", "auto"):
            raise ValueError("pattern_type must be 'processed', 'unprocessed', or 'auto'")
        self.path = Path(path)
        self.pattern_type = "processed" if pattern_type in ("processed", "auto") else "unprocessed"
        self.processing_device = processing_device
        self.gpu_device_id = int(gpu_device_id)
        self.lmsd_sigma_factor = float(lmsd_sigma_factor)
        self.lmsd_radius_factor = float(lmsd_radius_factor)
        self.lmsd_edge_mode = lmsd_edge_mode
        self.lmsd_clip_percentile = float(lmsd_clip_percentile)
        self.scan_group = None
        self.pc_source = "edax"
        self.pc_source_path = "EDAX metadata / Pattern Center Calibration"

    def _finalize_geometry(self, pc, sample_tilt, camera_elevation, camera_azimuth=0.0):
        self._pc = None if pc is None else tuple(float(v) for v in pc)
        self._sample_tilt = sample_tilt
        self._camera_elevation = camera_elevation
        self._camera_azimuth = float(camera_azimuth) if camera_azimuth is not None else 0.0
        self._detector_euler = None if camera_elevation is None else np.deg2rad(
            [self._camera_azimuth, 90.0 + float(camera_elevation), 0.0])

    def available_indices(self):
        return np.arange(self.count, dtype=np.int64)

    def has_pattern(self, index):
        return (not isinstance(index, (bool, np.bool_)) and
                isinstance(index, (int, np.integer)) and 0 <= index < self.count)

    def _check_index(self, index):
        if not self.has_pattern(index):
            raise IndexError(f"pattern index {index} outside 0..{self.count - 1}")

    def _process(self, image):
        if self.pattern_type == "processed":
            image = correct_pattern_dynamic_lmsd(
                image, self.lmsd_sigma_factor, self.lmsd_radius_factor,
                self.lmsd_edge_mode, self.lmsd_clip_percentile,
                self.processing_device, self.gpu_device_id)
        return np.asarray(image, dtype=np.float64)

    def pattern(self, index):
        return _crop(self.uncropped_pattern(index))

    def pattern_center(self, index):
        self._check_index(index)
        if self._pc is None:
            return None
        x, y, z = self._pc
        return (x, y, z)

    def pattern_centers(self):
        if self._pc is None:
            return None
        return np.tile(np.asarray(self._pc, dtype=np.float64), (self.count, 1))

    def sample_tilt_degrees(self):
        return self._sample_tilt

    def camera_elevation_degrees(self):
        return self._camera_elevation

    def phosphor_to_sample(self, sample_tilt_degrees=None):
        if self._detector_euler is None:
            return None
        tilt = self._sample_tilt if sample_tilt_degrees is None else float(sample_tilt_degrees)
        if tilt is None:
            return None
        return phosphor_to_sample_from_oxford(np.deg2rad(tilt), self._detector_euler)

    def orientation(self, index):
        self._check_index(index)
        if self._eulers is None:
            raise KeyError("ANG/OH5 orientation data are unavailable")
        return euler_to_matrix(*self._eulers[index])

    def grain_inputs(self):
        if self._eulers is None or self._phases is None:
            raise KeyError("grain segmentation needs ANG/OH5 orientation data")
        return self._eulers.copy(), self._phases.copy(), self._quality.copy()

    def map_index(self, column, row):
        if column < 0 or row < 0 or column >= self.x_cells or row >= self.y_cells:
            raise IndexError("map coordinate outside EDAX scan")
        index = int(row * self.x_cells + column)
        self._check_index(index)
        return index

    def has_unprocessed_static_background(self):
        return False

    def unprocessed_static_background(self):
        raise KeyError("EDAX input contains no static background")

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class OH5Reader(_EDAXReaderBase):
    """Read EDAX/OIM OH5 patterns and map metadata lazily."""
    source_format = "oh5"

    def __init__(self, path, pattern_type="processed", *, processing_device="cpu",
                 gpu_device_id=0, lmsd_sigma_factor=0.047,
                 lmsd_radius_factor=0.0375, lmsd_edge_mode="truncate",
                 lmsd_clip_percentile=0.75):
        self._init_common(path, pattern_type, processing_device, gpu_device_id,
                          lmsd_sigma_factor, lmsd_radius_factor, lmsd_edge_mode,
                          lmsd_clip_percentile)
        self._file = h5py.File(self.path, "r")
        try:
            self._data, self._header = _source_oh5(self._file)
            self.x_cells, self.y_cells, self._x_step, self._y_step, self._indices = \
                _oh5_layout(self._data, self._header)
            source = self._data["Pattern"]
            self.count = self.x_cells * self.y_cells
            self.height, self.width = map(int, source.shape[1:])
            self._patterns = source
            self._eulers = self._read_eulers()
            self._phases = self._read_phases()
            quality = self._data.get("IQ")
            self._quality = (np.asarray(quality[self._indices], dtype=np.float64)
                             if quality is not None else np.zeros(self.count))
            calibration = self._header.get("Pattern Center Calibration")
            pcx = _scalar(calibration, ("x-star",), None)
            pcy = _scalar(calibration, ("y-star",), None)
            pcz = _scalar(calibration, ("z-star",), None)
            self._finalize_geometry(
                None if any(v is None or not np.isfinite(v) for v in (pcx, pcy, pcz))
                else (pcx, pcy * self.height / self.width,
                      pcz * min(self.width, self.height) / self.width),
                _scalar(self._header, ("Sample Tilt",), None),
                _scalar(self._header, ("Camera Elevation Angle",), None),
                _scalar(self._header, ("Camera Azimuthal Angle",), 0.0))
        except Exception:
            self._file.close()
            raise

    def _read_eulers(self):
        names = ("Phi1", "Phi", "Phi2")
        if not all(name in self._data for name in names):
            return None
        values = np.column_stack([np.asarray(self._data[name][self._indices], dtype=np.float64)
                                   for name in names])
        return values if values.shape == (self.count, 3) else None

    def _read_phases(self):
        if "Phase" not in self._data:
            return None
        values = np.asarray(self._data["Phase"][self._indices], dtype=np.int32) + 1
        valid = self._data.get("Valid")
        if valid is not None:
            values[np.asarray(valid[self._indices]) != 1] = 0
        values[values < 0] = 0
        return values

    def uncropped_pattern(self, index):
        self._check_index(index)
        return self._process(self._patterns[int(self._indices[index])])

    def pattern_center(self, index):
        self._check_index(index)
        if self._pc is None:
            return None
        return self._pc

    def close(self):
        self._file.close()


@dataclass
class _Up2Header:
    version: int
    width: int
    height: int
    offset: int
    count: int
    map_width: int | None
    map_height: int | None
    is_hex: bool | None
    step_x: float | None
    step_y: float | None


def _read_up2_header(path, map_width=None, map_height=None, step_x=None,
                     step_y=None, grid=None):
    size = path.stat().st_size
    with path.open("rb") as stream:
        first = stream.read(42)
    if len(first) < 16:
        raise ValueError("UP2 is shorter than its basic header")
    version, width, height, offset = struct.unpack_from("<4I", first, 0)
    if version == 2 or width <= 0 or height <= 0 or offset < 16 or offset > size:
        raise ValueError("unsupported or invalid UP2 header")
    itemsize = width * height * 2
    payload = size - offset
    if payload % itemsize:
        raise ValueError("UP2 payload is not an integer number of patterns")
    count = payload // itemsize
    if version >= 3:
        if len(first) < 42:
            raise ValueError("UP2 v3+ is missing its extended header")
        file_nx, file_ny = struct.unpack_from("<2I", first, 17)
        file_hex = bool(first[25])
        file_dx, file_dy = struct.unpack_from("<2d", first, 26)
        map_width = file_nx if map_width is None else map_width
        map_height = file_ny if map_height is None else map_height
        step_x = file_dx if step_x is None else step_x
        step_y = file_dy if step_y is None else step_y
        is_hex = file_hex if grid is None else grid == "hex"
    else:
        is_hex = None if grid is None else grid == "hex"
    if map_width is None or map_height is None:
        raise ValueError("UP2 v1 has no map size; its companion ANG file must provide NCOLS_ODD and NROWS")
    expected = map_width * map_height - (map_height // 2 if is_hex else 0)
    if count != expected:
        raise ValueError(f"UP2 contains {count} patterns, expected {expected} from map dimensions")
    return _Up2Header(version, width, height, offset, count, map_width, map_height,
                      bool(is_hex), step_x, step_y)


def _up2_indices(header):
    if not header.is_hex:
        return np.arange(header.count, dtype=np.int64), header.map_width, header.map_height
    rows = np.arange(0, header.map_height, 2, dtype=np.int64)
    columns = np.arange(0, header.map_width, 2, dtype=np.int64)
    source = np.concatenate([row * header.map_width - row // 2 + columns for row in rows])
    return source, len(columns), len(rows)


class UP2Reader(_EDAXReaderBase):
    """Read EDAX UP2 uint16 patterns, optionally enriched by a companion ANG."""
    source_format = "up2"

    def __init__(self, path, pattern_type="processed", *, ang_path=None,
                 map_width=None, map_height=None, step_x=None, step_y=None,
                 grid=None, processing_device="cpu", gpu_device_id=0,
                 lmsd_sigma_factor=0.047, lmsd_radius_factor=0.0375,
                 lmsd_edge_mode="truncate", lmsd_clip_percentile=0.75):
        self._init_common(path, pattern_type, processing_device, gpu_device_id,
                          lmsd_sigma_factor, lmsd_radius_factor, lmsd_edge_mode,
                          lmsd_clip_percentile)
        candidate = Path(ang_path) if ang_path else self.path.with_suffix(".ang")
        self._ang_path = candidate if candidate.exists() else None
        self._ang = _parse_ang(self._ang_path) if self._ang_path else None
        if self._ang is None:
            raise ValueError("UP2 analysis requires a companion ANG file with map and indexing metadata")
        # UP2 v1 stores only pattern dimensions; ANG carries the map metadata.
        if self._ang is not None:
            map_width = map_width if map_width is not None else _ang_value(self._ang, "NCOLS_ODD", cast=int)
            map_height = map_height if map_height is not None else _ang_value(self._ang, "NROWS", cast=int)
            step_x = step_x if step_x is not None else _ang_value(self._ang, "XSTEP")
            step_y = step_y if step_y is not None else _ang_value(self._ang, "YSTEP")
            if grid is None:
                ang_grid = _ang_value(self._ang, "GRID", default="", cast=str).lower()
                if ang_grid:
                    grid = "hex" if "hex" in ang_grid else "square"
        self._header = _read_up2_header(self.path, map_width, map_height, step_x, step_y, grid)
        source_indices, self.x_cells, self.y_cells = _up2_indices(self._header)
        self._indices = source_indices
        self.count = len(source_indices)
        self.height, self.width = self._header.height, self._header.width
        self._patterns = np.memmap(self.path, dtype="<u2", mode="r",
                                   offset=self._header.offset,
                                   shape=(self._header.count, self.height, self.width))
        if len(self._ang.data) != self._header.count:
            raise ValueError("ANG and UP2 contain different numbers of patterns")
        self._load_ang_metadata()

    def _load_ang_metadata(self):
        ang = self._ang
        euler_cols = [_column(ang, name) for name in ("phi1", "PHI", "phi2")]
        self._eulers = (ang.data[self._indices][:, euler_cols] if ang is not None and all(i is not None for i in euler_cols)
                        else None)
        phase_col = _column(ang, "Phase index", "Phase")
        self._phases = (np.asarray(ang.data[self._indices, phase_col], dtype=np.int32) + 1
                        if ang is not None and phase_col is not None else None)
        quality_col = _column(ang, "IQ", "Image Quality")
        self._quality = (np.asarray(ang.data[self._indices, quality_col], dtype=np.float64)
                         if ang is not None and quality_col is not None else None)
        pc = tuple(_ang_value(ang, name, default=np.nan)
                   for name in ("x-star", "y-star", "z-star"))
        self._finalize_geometry(
            None if not np.all(np.isfinite(pc)) else
            (pc[0], pc[1] * self.height / self.width,
             pc[2] * min(self.width, self.height) / self.width),
            _ang_value(ang, "SampleTiltAngle"),
            _ang_value(ang, "CameraElevationAngle"),
            _ang_value(ang, "CameraAzimuthalAngle", default=0.0))

    def uncropped_pattern(self, index):
        self._check_index(index)
        return self._process(self._patterns[int(self._indices[index])])

    def pattern_center(self, index):
        self._check_index(index)
        if self._pc is None:
            return None
        source = int(self._indices[index])
        if self._ang is not None:
            pc = tuple(_ang_value(self._ang, name, default=np.nan)
                       for name in ("x-star", "y-star", "z-star"))
            if np.all(np.isfinite(pc)):
                return (pc[0], pc[1] * self.height / self.width,
                        pc[2] * min(self.width, self.height) / self.width)
        return self._pc

    def pattern_centers(self):
        center = self.pattern_center(0) if self.count else None
        return None if center is None else np.tile(center, (self.count, 1))

    def map_index(self, column, row):
        if column < 0 or row < 0 or column >= self.x_cells or row >= self.y_cells:
            raise IndexError("map coordinate outside UP2 scan")
        index = int(row * self.x_cells + column)
        self._check_index(index)
        return index

    def close(self):
        if hasattr(self._patterns, "_mmap") and self._patterns._mmap is not None:
            self._patterns._mmap.close()


def open_edax(path, pattern_type="processed", **kwargs):
    """Open an OH5, ANG+UP2, or UP2 input using the common reader API."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".oh5":
        return OH5Reader(path, pattern_type, **kwargs)
    if suffix == ".ang":
        return UP2Reader(path.with_suffix(".up2"), pattern_type, ang_path=path, **kwargs)
    if suffix == ".up2":
        return UP2Reader(path, pattern_type, **kwargs)
    raise ValueError("EDAX input must have .oh5, .ang, or .up2 extension")
