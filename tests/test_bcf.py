import math
import struct
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pyhrebsd.bcf import BCFReader, U64_MAX, correct_pattern_dynamic_lmsd
from pyhrebsd.geometry import euler_to_matrix
from run_pyhrebsd import _reference_index


class _BytesFile:
    def __init__(self, data):
        self.data = data

    def read(self, offset=0, length=None):
        return self.data[offset:] if length is None else self.data[offset:offset + length]


class _FakeSFS:
    def __init__(self, _path):
        width, height = 2, 1
        first = np.arange(12, dtype=np.uint16).reshape(3, 4)
        second = first + 100
        records = []
        offsets = []
        for column, image in enumerate((first, second)):
            offsets.append(sum(map(len, records)))
            records.append(struct.pack("<iiiiiiB", column, 0, 17, 4, 3, 2, 0) +
                           image.astype("<u2").tobytes())
        description = (struct.pack("<iii", width, height, width * height) +
                       struct.pack("<2Q", *offsets))
        calibration = b"""<Calibration>
          <PCX>0.0</PCX><PCY>-0.5</PCY><PCL>0.9</PCL>
          <ProbeTilt>1.2217304764</ProbeTilt><CameraTilt>0.1745329252</CameraTilt>
          <PhosphorSize>20</PhosphorSize><WorkingDistance>15</WorkingDistance>
          <ScanRotAngle>0</ScanRotAngle></Calibration>"""
        sem = b"<SEMImage><XCalibration>0.5</XCalibration><YCalibration>0.6</YCalibration></SEMImage>"
        rec = np.dtype([("x", "<u2"), ("y", "<u2"), ("pq", "<f4"),
                        ("det", "<u2"), ("e3", "<f4"), ("e2", "<f4"),
                        ("e1", "<f4"), ("phase", "<i2"),
                        ("indexed", "<u2"), ("mad", "<f4")])
        indexing = np.zeros(2, dtype=rec)
        indexing[0] = (0, 0, 20, 0, 0.3, 0.2, 0.1, 0, 8, 0.2)
        indexing[1] = (1, 0, 30, 0, 0.6, 0.5, 0.4, 0, 8, 0.2)
        self.files = {
            "EBSDData/FrameDescription": _BytesFile(description),
            "EBSDData/FrameData": _BytesFile(b"".join(records)),
            "EBSDData/Calibration": _BytesFile(calibration),
            "EBSDData/SEMImage": _BytesFile(sem),
            "EBSDData/Auxiliarien": _BytesFile(b"AcquisitionStep=2\n"),
            "EBSDData/IndexingResults": _BytesFile(indexing.tobytes()),
        }
        self.closed = False

    def require(self, name):
        return self.files[name]

    def close(self):
        self.closed = True


class BCFReaderTests(unittest.TestCase):
    @patch("pyhrebsd.bcf._SFSReader", _FakeSFS)
    def test_reads_raw_patterns_geometry_pc_and_indexing(self):
        with BCFReader(Path("scan.bcf"), "unprocessed") as reader:
            self.assertEqual((reader.count, reader.height, reader.width), (2, 3, 4))
            self.assertFalse(reader.has_unprocessed_static_background())
            np.testing.assert_array_equal(reader.uncropped_pattern(1),
                                          np.arange(12).reshape(3, 4) + 100)
            np.testing.assert_array_equal(reader.pattern(0),
                                          np.arange(12).reshape(3, 4)[:, :3])
            self.assertAlmostEqual(reader.sample_tilt_degrees(), 70.0, places=6)
            self.assertAlmostEqual(reader.camera_elevation_degrees(), 10.0, places=6)
            self.assertTrue(np.all(np.isfinite(reader.pattern_center(0))))
            self.assertGreater(reader.pattern_center(0)[2], 0)
            np.testing.assert_allclose(
                reader.orientation(0),
                euler_to_matrix(math.pi - 0.1, 0.2, math.pi - 0.3), atol=1e-7)
            eulers, phases, quality = reader.grain_inputs()
            self.assertEqual(eulers.shape, (2, 3))
            np.testing.assert_array_equal(phases, [1, 1])
            np.testing.assert_allclose(quality, [20, 30])
            self.assertEqual(reader.map_index(1, 0), 1)

    def test_dynamic_lmsd_returns_eight_bit_pattern(self):
        image = np.random.default_rng(5).integers(0, 65535, (48, 64), dtype=np.uint16)
        corrected = correct_pattern_dynamic_lmsd(image, 0.047, 0.0375, "truncate", 0.75, "cpu", 0)
        self.assertEqual(corrected.shape, image.shape)
        self.assertEqual(corrected.dtype, np.uint8)
        self.assertGreater(np.ptp(corrected), 0)

    @patch("pyhrebsd.bcf._SFSReader", _FakeSFS)
    def test_rejects_missing_pattern_slot(self):
        with BCFReader("scan.bcf", "unprocessed") as reader:
            reader._offsets[1] = U64_MAX
            np.testing.assert_array_equal(reader.available_indices(), [0])
            self.assertTrue(reader.has_pattern(0))
            self.assertFalse(reader.has_pattern(1))
            with self.assertRaisesRegex(ValueError, "no pattern"):
                reader.pattern(1)

    @patch("pyhrebsd.bcf._SFSReader", _FakeSFS)
    def test_default_reference_is_first_stored_pattern(self):
        with BCFReader("scan.bcf", "unprocessed") as reader:
            reader._offsets[0] = U64_MAX
            self.assertEqual(_reference_index(reader, {"reference_map_point": None}), 1)


if __name__ == "__main__":
    unittest.main()
