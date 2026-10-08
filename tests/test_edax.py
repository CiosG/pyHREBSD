import struct
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from pyhrebsd.edax import OH5Reader, open_edax


class EDAXReaderTests(unittest.TestCase):
    def _write_oh5(self, path):
        with h5py.File(path, "w") as file:
            data = file.create_group("Slice").create_group("EBSD").create_group("Data")
            header = file["Slice/EBSD"].create_group("Header")
            data.create_dataset("Pattern", data=np.arange(6 * 8 * 8, dtype=np.uint16).reshape(6, 8, 8))
            for name in ("Phi1", "Phi", "Phi2"):
                data.create_dataset(name, data=np.zeros(6))
            data.create_dataset("Phase", data=np.zeros(6, dtype=np.int32))
            data.create_dataset("IQ", data=np.arange(6, dtype=float))
            for name, value in (("nColumns", 3), ("nRows", 2), ("Step X", 1.0),
                                ("Step Y", 1.0), ("Sample Tilt", 70.0),
                                ("Camera Elevation Angle", 10.0)):
                header.create_dataset(name, data=value)
            pc = header.create_group("Pattern Center Calibration")
            for name, value in (("x-star", .45), ("y-star", .53), ("z-star", .65)):
                pc.create_dataset(name, data=value)

    def test_oh5_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.oh5"
            self._write_oh5(path)
            with OH5Reader(path, "unprocessed") as reader:
                self.assertEqual((reader.count, reader.x_cells, reader.y_cells), (6, 3, 2))
                self.assertEqual(reader.pattern(0).shape, (8, 8))
                self.assertEqual(reader.map_index(2, 1), 5)
                self.assertAlmostEqual(reader.sample_tilt_degrees(), 70.0)
                self.assertEqual(reader.orientation(2).shape, (3, 3))

    def test_ang_up2_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            up2 = directory / "scan.up2"
            payload = np.arange(6 * 8 * 8, dtype=np.uint16).tobytes()
            header = struct.pack("<4I", 3, 8, 8, 42) + bytes([0])
            header += struct.pack("<2I", 3, 2) + bytes([0]) + struct.pack("<2d", 1.0, 1.0)
            up2.write_bytes(header + payload)
            ang = directory / "scan.ang"
            rows = "\n".join(f"0 0 0 0 0 {i} 0 0" for i in range(6))
            ang.write_text("# COLUMN_HEADERS: phi1, PHI, phi2, x, y, IQ, CI, Phase\n"
                           "# x-star: 0.45\n# y-star: 0.53\n# z-star: 0.65\n"
                           "# SampleTiltAngle: 70\n# CameraElevationAngle: 10\n" + rows)
            with open_edax(ang, "unprocessed") as reader:
                self.assertEqual((reader.count, reader.x_cells, reader.y_cells), (6, 3, 2))
                self.assertEqual(reader.pattern(5).shape, (8, 8))
                self.assertEqual(reader.map_index(2, 1), 5)
                self.assertAlmostEqual(reader.camera_elevation_degrees(), 10.0)


if __name__ == "__main__":
    unittest.main()
