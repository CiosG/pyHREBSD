"""Read Bruker ESPRIT BCF EBSD maps without an intermediate conversion.

Only the read-only EBSD functionality needed by PyHREBSD is included; EDS
data are left untouched.
"""

from __future__ import annotations

import math
import os
import struct
import sys
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from .geometry import euler_to_matrix, phosphor_to_sample_from_oxford


MAGIC = b"AAMVHFSS"
PREFIX = 0x118
BLOCK_HEADER = 0x20
U64_MAX = (1 << 64) - 1
_LMSD_WEIGHT_CACHE = {}
_DLL_DIRECTORY_HANDLES = []


def _enable_windows_conda_dlls():
    """Make the active Conda environment's native libraries discoverable."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    conda_dlls = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(conda_dlls) and not _DLL_DIRECTORY_HANDLES:
        os.environ["PATH"] = conda_dlls + os.pathsep + os.environ.get("PATH", "")
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(conda_dlls))


_enable_windows_conda_dlls()


class BCFError(RuntimeError):
    """A malformed, truncated, or unsupported BCF input."""


class _VirtualFile:
    def __init__(self, container, path, size, pointers):
        self.container = container
        self.path = path
        self.size = int(size)
        self.pointers = pointers
        self._compression_checked = False
        self._inflated = None

    def _read_raw(self, offset, length):
        if offset < 0 or length < 0 or offset + length > self.size:
            raise BCFError(f"range outside virtual file {self.path}")
        out = bytearray()
        block_no, within = divmod(offset, self.container.usable)
        remaining = length
        while remaining:
            if block_no >= len(self.pointers):
                raise BCFError(f"short block table for {self.path}")
            take = min(remaining, self.container.usable - within)
            physical = self.container.payload_offset(self.pointers[block_no]) + within
            out.extend(self.container.read_at(physical, take))
            remaining -= take
            block_no += 1
            within = 0
        return bytes(out)

    def _check_compression(self):
        if self._compression_checked:
            return
        self._compression_checked = True
        if self.size >= 4 and self._read_raw(0, 4) == b"AACS":
            raw = self._read_raw(0, self.size)
            if len(raw) < 0x80:
                raise BCFError(f"truncated AACS header in {self.path}")
            count = struct.unpack_from("<I", raw, 0x0C)[0]
            position, chunks = 0x80, []
            for _ in range(count):
                if position + 16 > len(raw):
                    raise BCFError(f"truncated AACS block header in {self.path}")
                size = struct.unpack_from("<I", raw, position)[0]
                position += 16
                if position + size > len(raw):
                    raise BCFError(f"truncated AACS block in {self.path}")
                chunks.append(zlib.decompress(raw[position:position + size]))
                position += size
            self._inflated = b"".join(chunks)

    def read(self, offset=0, length=None):
        self._check_compression()
        logical_size = len(self._inflated) if self._inflated is not None else self.size
        if length is None:
            length = logical_size - offset
        if offset < 0 or length < 0 or offset + length > logical_size:
            raise BCFError(f"range outside virtual file {self.path}")
        if self._inflated is not None:
            return self._inflated[offset:offset + length]
        return self._read_raw(offset, length)


class _SFSReader:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.handle = open(self.path, "rb")
        try:
            self.physical_size = os.path.getsize(self.path)
            head = self.read_at(0, 0x14C)
            if head[:8] != MAGIC:
                raise BCFError("not an AidAim SFS/Bruker BCF file")
            self.version = struct.unpack_from("<f", head, 0x124)[0]
            self.block_size = struct.unpack_from("<I", head, 0x128)[0]
            self.tree_block = struct.unpack_from("<I", head, 0x140)[0]
            self.tree_count = struct.unpack_from("<I", head, 0x144)[0]
            self.block_count = struct.unpack_from("<I", head, 0x148)[0]
            if self.block_size <= BLOCK_HEADER:
                raise BCFError("invalid SFS block size")
            self.usable = self.block_size - BLOCK_HEADER
            expected = PREFIX + self.block_count * self.block_size
            if expected != self.physical_size:
                raise BCFError(
                    f"truncated or overlong BCF: expected {expected} bytes, "
                    f"got {self.physical_size}"
                )
            self.files = self._read_directory()
        except Exception:
            self.handle.close()
            raise

    def close(self):
        self.handle.close()

    def read_at(self, offset, size):
        if offset < 0 or size < 0 or offset + size > self.physical_size:
            raise BCFError("physical read outside BCF")
        self.handle.seek(offset)
        data = self.handle.read(size)
        if len(data) != size:
            raise BCFError("unexpected physical EOF")
        return data

    def block_offset(self, index):
        if index < 0 or index >= self.block_count:
            raise BCFError(f"SFS block index {index} out of range")
        return PREFIX + index * self.block_size

    def payload_offset(self, index):
        return self.block_offset(index) + BLOCK_HEADER

    def _linked_payload(self, first, blocks, bytes_per_block):
        out, current, seen = bytearray(), first, set()
        if bytes_per_block < 0 or bytes_per_block > self.usable:
            raise BCFError("invalid linked-block payload length")
        for _ in range(blocks):
            if current in seen:
                raise BCFError("cycle in SFS linked-block chain")
            seen.add(current)
            header = self.read_at(self.block_offset(current), BLOCK_HEADER)
            next_block = struct.unpack_from("<I", header, 0)[0]
            out.extend(self.read_at(self.payload_offset(current), bytes_per_block))
            current = next_block
        return bytes(out)

    def _read_directory(self):
        items_per_block = self.usable // 512
        if not items_per_block:
            raise BCFError("SFS block too small for a directory entry")
        blocks = (self.tree_count + items_per_block - 1) // items_per_block
        raw = self._linked_payload(self.tree_block, blocks, items_per_block * 512)
        entries = []
        for index in range(self.tree_count):
            entry = raw[index * 512:(index + 1) * 512]
            first = struct.unpack_from("<i", entry, 0)[0]
            size = struct.unpack_from("<Q", entry, 4)[0]
            parent = struct.unpack_from("<i", entry, 0x28)[0]
            is_directory = entry[0xDC] != 0
            name = entry[0xE0:0x1E0].split(b"\0", 1)[0].decode("utf-8", "replace")
            entries.append((first, size, parent, is_directory, name))

        def full_path(index):
            names, seen = [], set()
            while index != -1:
                if index < 0 or index >= len(entries) or index in seen:
                    raise BCFError("invalid directory parent chain")
                seen.add(index)
                names.append(entries[index][4])
                index = entries[index][2]
            return "/".join(reversed([name for name in names if name]))

        files = {}
        capacity = self.usable // 4
        for index, (first, size, _parent, is_directory, _name) in enumerate(entries):
            if is_directory:
                continue
            path = full_path(index)
            data_blocks = (size + self.usable - 1) // self.usable if size else 0
            table_blocks = (data_blocks + capacity - 1) // capacity if data_blocks else 0
            if table_blocks:
                table = self._linked_payload(first, table_blocks, capacity * 4)
                pointers = list(struct.unpack_from(f"<{data_blocks}I", table, 0))
                for pointer in pointers:
                    self.block_offset(pointer)
            else:
                pointers = []
            files[path] = _VirtualFile(self, path, size, pointers)
        return files

    def require(self, path):
        try:
            return self.files[path]
        except KeyError as exc:
            raise BCFError(f"required virtual file is absent: {path}") from exc


def _xml_root(vfile):
    return ET.fromstring(vfile.read().decode("cp1252", "replace"))


def _local_name(tag):
    return tag.rsplit("}", 1)[-1]


def _named_text(root, name, default=None):
    if root is None:
        return default
    for element in root.iter():
        if _local_name(element.tag) == name:
            text = (element.text or "").strip()
            if text:
                return text
            for child in element.iter():
                text = (child.text or "").strip()
                if text:
                    return text
    return default


def _number(root, name, default=None):
    value = _named_text(root, name)
    if value is None:
        return default
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return default


def _parse_key_values(vfile):
    result = {}
    for line in vfile.read().decode("cp1252", "replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def _pattern_center_model(bcf, map_width, map_height, pattern_width, pattern_height):
    calibration = _xml_root(bcf.require("EBSDData/Calibration"))
    sem = _xml_root(bcf.require("EBSDData/SEMImage"))
    auxiliary = _parse_key_values(bcf.require("EBSDData/Auxiliarien"))
    names = ("PCX", "PCY", "PCL", "ProbeTilt", "CameraTilt",
             "PhosphorSize", "WorkingDistance")
    values = {name: _number(calibration, name) for name in names}
    if any(values[name] is None for name in names):
        raise BCFError("BCF lacks values required for pattern-centre geometry")
    if values["PhosphorSize"] <= 0:
        raise BCFError("invalid BCF phosphor size")
    acquisition_step = float(auxiliary.get("AcquisitionStep", 1.0))
    step_x, step_y = _number(sem, "XCalibration"), _number(sem, "YCalibration")
    if step_x is None or step_y is None:
        raise BCFError("BCF lacks SEM calibration required for pattern-centre geometry")
    step_x *= acquisition_step
    step_y *= acquisition_step
    aspect = float(pattern_width) / pattern_height
    camera_tilt = values["CameraTilt"]
    base_x = 0.5 + values["PCX"] / aspect
    base_y = (values["PCY"] + values["WorkingDistance"] /
              values["PhosphorSize"] * math.cos(camera_tilt))
    base_z = (values["PCL"] - values["WorkingDistance"] /
              values["PhosphorSize"] * math.sin(camera_tilt))
    return {
        "center_x": map_width / 2.0, "center_y": map_height / 2.0,
        "step_x": step_x, "step_y": step_y,
        "phosphor_um": values["PhosphorSize"] * 1000.0,
        "aspect": aspect, "base_x": base_x, "base_y": base_y, "base_z": base_z,
        "detector_sample_angle": math.pi / 2.0 - values["ProbeTilt"] + camera_tilt,
        "scan_rotation": _number(calibration, "ScanRotAngle", 0.0),
    }


def _pattern_center_from_model(model, column, row):
    dx = (column - model["center_x"]) * model["step_x"]
    dy = (row - model["center_y"]) * model["step_y"]
    rotation = model["scan_rotation"]
    scan_x = dx * math.cos(rotation) - dy * math.sin(rotation)
    scan_y = dx * math.sin(rotation) + dy * math.cos(rotation)
    angle = model["detector_sample_angle"]
    x = model["base_x"] - scan_x / (model["phosphor_um"] * model["aspect"])
    y = model["base_y"] - scan_y * math.cos(angle) / model["phosphor_um"]
    z = model["base_z"] + scan_y * math.sin(angle) / model["phosphor_um"]
    return x, (1.0 - y) / model["aspect"], z / model["aspect"]


def _dynamic_lmsd(image, sigma_factor, radius_factor, edge_mode, clip_percentile,
                  device, gpu_device_id):
    if edge_mode not in ("truncate", "reflect", "nearest"):
        raise ValueError("BCF LMSD edge mode must be truncate, reflect, or nearest")
    if sigma_factor <= 0 or radius_factor < 0 or not 0 <= clip_percentile < 50:
        raise ValueError("invalid BCF dynamic-LMSD settings")
    if device == "gpu":
        _enable_windows_conda_dlls()
        import cupy as xp
        from cupyx.scipy.ndimage import gaussian_filter as gaussian
        from cupyx.scipy.ndimage import uniform_filter as uniform
        xp.cuda.Device(gpu_device_id).use()
    elif device == "cpu":
        xp, gaussian, uniform = np, gaussian_filter, uniform_filter
    else:
        raise ValueError("BCF processing device must be 'cpu' or 'gpu'")
    values = xp.asarray(image, dtype=xp.float32)
    sigma = values.shape[1] * sigma_factor
    size = 2 * int(values.shape[1] * radius_factor) + 1
    if edge_mode == "truncate":
        gaussian_key = (device, gpu_device_id, tuple(values.shape), "gaussian", float(sigma))
        gaussian_weights = _LMSD_WEIGHT_CACHE.get(gaussian_key)
        if gaussian_weights is None:
            gaussian_weights = gaussian(xp.ones(values.shape, xp.float32), sigma=sigma,
                                        mode="constant", cval=0.0)
            _LMSD_WEIGHT_CACHE[gaussian_key] = gaussian_weights
        background = gaussian(values, sigma=sigma, mode="constant", cval=0.0) / gaussian_weights
        uniform_key = (device, gpu_device_id, tuple(values.shape), "uniform", int(size))
        uniform_weights = _LMSD_WEIGHT_CACHE.get(uniform_key)
        if uniform_weights is None:
            uniform_weights = uniform(xp.ones(values.shape, xp.float32), size=size,
                                      mode="constant", cval=0.0)
            _LMSD_WEIGHT_CACHE[uniform_key] = uniform_weights
        mean = uniform(values - background, size=size, mode="constant", cval=0.0) / uniform_weights
        second = uniform((values - background) ** 2, size=size,
                         mode="constant", cval=0.0) / uniform_weights
    else:
        background = gaussian(values, sigma=sigma, mode=edge_mode)
        residual = values - background
        mean = uniform(residual, size=size, mode=edge_mode)
        second = uniform(residual * residual, size=size, mode=edge_mode)
    residual = values - background
    deviation = xp.sqrt(xp.maximum(second - mean * mean, 0.0))
    filtered = xp.where(deviation >= 1e-8,
                        (residual - mean) / xp.maximum(deviation, 1e-8), 0.0)
    low, high = xp.percentile(filtered, (clip_percentile, 100.0 - clip_percentile))
    if float(high - low) <= 0:
        raise ValueError("background-corrected BCF pattern has no contrast")
    result = (xp.clip((filtered - low) / (high - low), 0.0, 1.0) * 255.0).astype(xp.uint8)
    return xp.asnumpy(result) if device == "gpu" else np.asarray(result)


class BCFReader:
    """Lazy random-access reader exposing the same analysis API as H5OINAReader."""

    source_format = "bcf"

    def __init__(self, path: str | Path, pattern_type="processed", *,
                 processing_device="cpu", gpu_device_id=0,
                 lmsd_sigma_factor=0.047, lmsd_radius_factor=0.0375,
                 lmsd_edge_mode="truncate", lmsd_clip_percentile=0.75):
        self.path = Path(path)
        if pattern_type == "auto":
            pattern_type = "processed"
        if pattern_type not in ("processed", "unprocessed"):
            raise ValueError("pattern_type must be 'processed', 'unprocessed', or 'auto'")
        self.pattern_type = pattern_type
        self.processing_device = processing_device
        self.gpu_device_id = int(gpu_device_id)
        self.lmsd_sigma_factor = float(lmsd_sigma_factor)
        self.lmsd_radius_factor = float(lmsd_radius_factor)
        self.lmsd_edge_mode = lmsd_edge_mode
        self.lmsd_clip_percentile = float(lmsd_clip_percentile)
        self.scan_group = None
        self.pc_source = "bcf"
        self.pc_source_path = "EBSDData/Calibration (reconstructed per point)"
        self._file = _SFSReader(self.path)
        try:
            description = self._file.require("EBSDData/FrameDescription").read()
            if len(description) < 12:
                raise BCFError("short FrameDescription")
            self.x_cells, self.y_cells, self.count = struct.unpack_from("<iii", description, 0)
            if (self.x_cells <= 0 or self.y_cells <= 0 or
                    self.count != self.x_cells * self.y_cells or
                    len(description) != 12 + self.count * 8):
                raise BCFError("invalid FrameDescription dimensions")
            self._offsets = np.frombuffer(description, dtype="<u8", offset=12,
                                          count=self.count).copy()
            valid = np.flatnonzero(self._offsets != U64_MAX)
            if not len(valid):
                raise BCFError("BCF contains no EBSD patterns")
            self._frames = self._file.require("EBSDData/FrameData")
            first = self._frames.read(int(self._offsets[valid[0]]), 25)
            _x, _y, _field, self.width, self.height, self._bpp, self._pixel_format = \
                struct.unpack("<iiiiiiB", first)
            if self.width <= 0 or self.height <= 0 or self._bpp not in (1, 2):
                raise BCFError("unsupported EBSD pattern representation")
            self._pc_model = _pattern_center_model(
                self._file, self.x_cells, self.y_cells, self.width, self.height)
            calibration = _xml_root(self._file.require("EBSDData/Calibration"))
            self._probe_tilt = _number(calibration, "ProbeTilt")
            self._camera_tilt = _number(calibration, "CameraTilt")
            self._header = {
                "X Step": np.asarray([self._pc_model["step_x"]]),
                "Y Step": np.asarray([self._pc_model["step_y"]]),
            }
            self._eulers, self._phases, self._quality = self._read_indexing_results()
        except Exception:
            self._file.close()
            raise

    def _read_indexing_results(self):
        eulers = np.zeros((self.count, 3), dtype=np.float64)
        phases = np.zeros(self.count, dtype=np.int32)
        quality = np.zeros(self.count, dtype=np.float64)
        results = self._file.files.get("EBSDData/IndexingResults")
        if results is None:
            return eulers, phases, quality
        raw = results.read()
        record_type = np.dtype([
            ("x", "<u2"), ("y", "<u2"), ("pq", "<f4"), ("det", "<u2"),
            ("e3", "<f4"), ("e2", "<f4"), ("e1", "<f4"),
            ("phase", "<i2"), ("indexed", "<u2"), ("mad", "<f4"),
        ])
        if len(raw) % record_type.itemsize:
            raise BCFError("IndexingResults size is not divisible by 30")
        for record in np.frombuffer(raw, dtype=record_type):
            column, row = int(record["x"]), int(record["y"])
            if column >= self.x_cells or row >= self.y_cells:
                raise BCFError("IndexingResults coordinate out of range")
            index = row * self.x_cells + column
            eulers[index] = ((math.pi - float(record["e1"])) % (2 * math.pi),
                              float(record["e2"]) % (2 * math.pi),
                              (math.pi - float(record["e3"])) % (2 * math.pi))
            quality[index] = record["pq"]
            if int(record["phase"]) >= 0:
                phases[index] = int(record["phase"]) + 1
        return eulers, phases, quality

    def _check_index(self, index):
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError("pattern index must be an integer")
        if index < 0 or index >= self.count:
            raise IndexError(f"pattern index {index} outside 0..{self.count - 1}")
        if self._offsets[index] == U64_MAX:
            raise ValueError(f"BCF has no pattern at map index {index}")

    def available_indices(self):
        """Return map indices whose pattern payload is present in the BCF."""
        return np.flatnonzero(self._offsets != U64_MAX)

    def has_pattern(self, index):
        """Return whether a map slot contains a stored pattern."""
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            return False
        return 0 <= index < self.count and self._offsets[index] != U64_MAX

    def uncropped_pattern(self, index):
        self._check_index(index)
        offset = int(self._offsets[index])
        header = self._frames.read(offset, 25)
        column, row, _field, width, height, bpp, _format = struct.unpack("<iiiiiiB", header)
        if (column != index % self.x_cells or row != index // self.x_cells or
                (width, height, bpp) != (self.width, self.height, self._bpp)):
            raise BCFError(f"inconsistent EBSD pattern header at slot {index}")
        raw = self._frames.read(offset + 25, width * height * bpp)
        dtype = "<u2" if bpp == 2 else np.uint8
        image = np.frombuffer(raw, dtype=dtype).reshape(height, width)
        if self.pattern_type == "processed":
            image = _dynamic_lmsd(
                image, self.lmsd_sigma_factor, self.lmsd_radius_factor,
                self.lmsd_edge_mode, self.lmsd_clip_percentile,
                self.processing_device, self.gpu_device_id)
        return np.asarray(image, dtype=np.float64)

    def pattern(self, index):
        image = self.uncropped_pattern(index)
        side = min(self.width, self.height)
        top, left = (self.height - side) // 2, (self.width - side) // 2
        return image[top:top + side, left:left + side]

    def unprocessed_static_background(self):
        raise KeyError("BCF contains no standalone static-background image; use processed "
                       "dynamic LMSD or a non-static raw-pattern correction")

    def pattern_center(self, index):
        self._check_index(index)
        column, row = index % self.x_cells, index // self.x_cells
        x, y, z = _pattern_center_from_model(self._pc_model, column, row)
        side = min(self.width, self.height)
        left = (self.width - side) // 2
        bottom = self.height - side - (self.height - side) // 2
        return ((x * self.width - left) / side,
                (y * self.width - bottom) / side,
                z * self.width / side)

    def pattern_centers(self):
        centers = np.empty((self.count, 3), dtype=np.float64)
        centers[:] = np.nan
        for index in np.flatnonzero(self._offsets != U64_MAX):
            centers[index] = self.pattern_center(int(index))
        return centers

    def sample_tilt_degrees(self):
        return None if self._probe_tilt is None else math.degrees(self._probe_tilt)

    def camera_elevation_degrees(self):
        return None if self._camera_tilt is None else math.degrees(self._camera_tilt)

    def phosphor_to_sample(self, sample_tilt_degrees=None):
        tilt = self.sample_tilt_degrees() if sample_tilt_degrees is None else sample_tilt_degrees
        if tilt is None or self._camera_tilt is None:
            return None
        detector_euler = np.asarray([0.0, math.pi / 2.0 + self._camera_tilt, 0.0])
        return phosphor_to_sample_from_oxford(math.radians(tilt), detector_euler)

    def orientation(self, index):
        self._check_index(index)
        return euler_to_matrix(*self._eulers[index])

    def grain_inputs(self):
        return self._eulers.copy(), self._phases.copy(), self._quality.copy()

    def map_index(self, column, row):
        if column < 0 or row < 0 or column >= self.x_cells or row >= self.y_cells:
            raise IndexError("map coordinate outside EBSD scan")
        index = row * self.x_cells + column
        self._check_index(index)
        return index

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
