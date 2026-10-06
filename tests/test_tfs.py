import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from pyhrebsd.tfs import TFSReader


class TFSReaderTests(unittest.TestCase):
    def test_reads_xetalview_layout_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.tfs.hdf5"
            patterns = np.arange(2 * 3 * 8 * 8, dtype=np.uint8).reshape(2, 3, 8, 8)
            with h5py.File(path, "w") as file:
                file.create_dataset("Site/EBSD/Patterns/Processed", data=patterns)
                data = file.create_group("Site/EBSD/MapData")
                data.create_dataset("PatternCenter", data=np.full((2, 3, 3), [0.5, 0.6, 0.7]))
                data.create_dataset("EulerAngles", data=np.zeros((2, 3, 3)))
                data.create_dataset("Phase", data=np.ones((2, 3), dtype=np.int32))
                data.create_dataset("IndexQuality", data=np.ones((2, 3)))
                file.create_dataset("Site/Acquisition/SpecimenTilt", data=[np.deg2rad(70)])
                file.create_dataset("Site/Acquisition/StepSize", data=[1e-6])
                file.create_dataset("Site/EBSD/Info/DCStoSCS", data=[0., 90., 0.])
            with TFSReader(path) as reader:
                self.assertEqual((reader.x_cells, reader.y_cells, reader.count), (3, 2, 6))
                np.testing.assert_array_equal(reader.pattern(4), patterns[1, 1])
                np.testing.assert_allclose(reader.pattern_center(4), [0.5, 0.6, 0.7])
                self.assertAlmostEqual(reader.sample_tilt_degrees(), 70.)
                self.assertEqual(reader.grain_inputs()[0].shape, (6, 3))
                self.assertEqual(reader.map_index(1, 1), 4)


if __name__ == "__main__":
    unittest.main()
