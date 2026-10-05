import json
import csv
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from pyhrebsd.h5oina import H5OINAReader
from pyhrebsd.geometry import euler_to_matrix, phosphor_to_sample_from_oxford
from pyhrebsd.pc_calibration import (fit_beam_shift_pc_plane, fit_pc_plane,
                                   measure_effective_pixel_size,
                                   pc_plane_from_effective_pixel_size)
from run_pyhrebsd import run


class H5OINATests(unittest.TestCase):
    def _make_file(self, path, patterns, *, with_pc=True, group="1"):
        with h5py.File(path, "w") as file:
            data = file.create_group(f"{group}/EBSD/Data")
            header = file.create_group(f"{group}/EBSD/Header")
            data.create_dataset("Processed Patterns", data=patterns)
            data.create_dataset("Unprocessed Patterns", data=patterns.astype(np.int16) * 2)
            data.create_dataset("Euler", data=np.tile([0.2, 0.3, 0.4], (len(patterns), 1)))
            header.create_dataset("X Cells", data=[[len(patterns)]])
            header.create_dataset("Y Cells", data=[[1]])
            if with_pc:
                for name, value in (("Pattern Center X", 0.5),
                                    ("Pattern Center Y", 0.25),
                                    ("Detector Distance", 0.6)):
                    data.create_dataset(name, data=np.full((len(patterns), 1), value))

    def test_reads_processed_and_unprocessed_patterns_by_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.h5oina"
            patterns = np.arange(3 * 12 * 16, dtype=np.uint8).reshape(3, 12, 16)
            self._make_file(path, patterns)
            with H5OINAReader(path) as reader:
                self.assertEqual(reader.count, 3)
                self.assertEqual(reader.map_index(2, 0), 2)
                np.testing.assert_array_equal(reader.pattern(1), patterns[1, :, 2:14])
                np.testing.assert_allclose(reader.pattern_center(1),
                                           ((0.5 * 16 - 2) / 12, 0.25 * 16 / 12,
                                            0.6 * 16 / 12))
                np.testing.assert_allclose(reader.orientation(1) @ reader.orientation(1).T,
                                           np.eye(3), atol=1e-14)
                np.testing.assert_allclose(reader.orientation(1),
                                           euler_to_matrix(0.2, 0.3, 0.4), atol=1e-14)
                with self.assertRaises(IndexError):
                    reader.pattern(3)
            with H5OINAReader(path, pattern_type="unprocessed") as reader:
                np.testing.assert_array_equal(reader.pattern(1), patterns[1, :, 2:14].astype(np.int16) * 2)

    def test_missing_pattern_centers_use_config_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "scan.h5oina"
            pattern = np.random.default_rng(7).integers(0, 255, size=(128, 128), dtype=np.uint8)
            self._make_file(path, np.stack((pattern, pattern)), with_pc=False)
            config = {
                "input_mode": "h5oina", "h5oina_file": str(path),
                "h5_scan_group": None, "h5_pattern_type": "processed",
                "reference_map_point": (0, 0), "scan_indices": None,
                "save_per_pattern_files": True,
                "pattern_center_fallback": (0.45, 0.53, 0.65),
                "output_dir": str(directory / "results"),
                "material_database": "pyhrebsd/materials.h5",
                "material_name": "copper",
                "reference_euler_degrees": None,
                "sample_tilt_degrees": 70, "camera_elevation_degrees": 10,
                "roi_size": 24, "roi_count": 49,
                "roi_filter": None, "outlier_standard_deviation": 2,
            }
            run(config)
            summary = json.loads((directory / "results" / "scan_000001_analysis.json").read_text())
            self.assertEqual(summary["reference_index"], 0)
            self.assertEqual(summary["scan_index"], 1)
            self.assertLess(summary["rms_shift_error_pixels"], 1e-8)
            with (directory / "results" / "scan_results.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["scan_index"] for row in rows], ["0", "1"])
            self.assertEqual([row["reference_index"] for row in rows], ["0", "0"])
            self.assertTrue(all(row["status"] == "ok" for row in rows))

    def test_pc_source_switch_reads_only_selected_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.h5oina"
            self._make_file(path, np.zeros((4, 12, 16), dtype=np.uint8))
            with h5py.File(path, "a") as file:
                data = file.create_group("1/Data Processing/Data")
                for name, value in (("Pattern Center X", 0.55),
                                    ("Pattern Center Y", 0.30),
                                    ("Detector Distance", 0.65)):
                    data.create_dataset(name, data=np.full(4, value))
            with H5OINAReader(path, pc_source="ebsd") as reader:
                ebsd_pc = reader.pattern_center(0)
                self.assertEqual(reader.pc_source_path, "/1/EBSD/Data")
            with H5OINAReader(path, pc_source="data_processing") as reader:
                processing_pc = reader.pattern_center(0)
                self.assertEqual(reader.pc_source_path, "/1/Data Processing/Data")
                np.testing.assert_allclose(reader.pattern_centers()[0], processing_pc)
                plane = fit_pc_plane(reader)
                self.assertEqual(plane.report()["source_path"], reader.pc_source_path)
                np.testing.assert_allclose(plane.at(0, 4), processing_pc)
            np.testing.assert_allclose(np.subtract(processing_pc, ebsd_pc),
                                       [0.05 * 16 / 12, 0.05 * 16 / 12,
                                        0.05 * 16 / 12])
            with self.assertRaisesRegex(ValueError, "pc_source"):
                H5OINAReader(path, pc_source="unknown")

    def test_missing_requested_processing_pc_is_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.h5oina"
            self._make_file(path, np.zeros((2, 8, 8), dtype=np.uint8))
            with self.assertRaisesRegex(KeyError, "Data Processing/Data"):
                H5OINAReader(path, pc_source="data_processing")

    def test_runner_records_selected_processing_pc(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "scan.h5oina"
            pattern = np.random.default_rng(31).integers(0, 255, size=(128, 128), dtype=np.uint8)
            self._make_file(path, np.stack((pattern, pattern)))
            with h5py.File(path, "a") as file:
                data = file.create_group("1/Data Processing/Data")
                for name, value in (("Pattern Center X", 0.55),
                                    ("Pattern Center Y", 0.30),
                                    ("Detector Distance", 0.65)):
                    data.create_dataset(name, data=np.full(2, value))
            config = {
                "input_mode": "h5oina", "h5oina_file": str(path),
                "h5_pc_source": "data_processing", "pc_mode": "h5",
                "h5_scan_group": None, "h5_pattern_type": "processed",
                "reference_map_point": (0, 0), "scan_indices": [1],
                "output_dir": str(directory / "results"),
                "material_database": "pyhrebsd/materials.h5",
                "material_name": "copper",
                "reference_euler_degrees": None,
                "sample_tilt_degrees": 70, "camera_elevation_degrees": 10,
                "roi_size": 24, "roi_count": 49,
                "roi_filter": None, "outlier_standard_deviation": 2,
            }
            run(config)
            with (directory / "results" / "scan_results.csv").open(newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["pc_source"], "data_processing")
            self.assertEqual(row["pc_mode"], "h5")
            self.assertAlmostEqual(float(row["pattern_center_x"]), 0.55)
            self.assertAlmostEqual(float(row["pattern_center_y"]), 0.30)
            self.assertAlmostEqual(float(row["pattern_center_z"]), 0.65)

    def test_multiple_scans_require_explicit_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.h5oina"
            self._make_file(path, np.zeros((2, 8, 8), dtype=np.uint8), group="1")
            with h5py.File(path, "a") as file:
                file.copy("1", "2")
            with self.assertRaisesRegex(ValueError, "set scan_group"):
                H5OINAReader(path)
            with H5OINAReader(path, scan_group="2") as reader:
                self.assertEqual(reader.scan_group, "2")

    def test_grain_inputs_read_indexed_map_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.h5oina"
            self._make_file(path, np.zeros((3, 8, 8), dtype=np.uint8))
            with h5py.File(path, "a") as file:
                data = file["1/EBSD/Data"]
                data.create_dataset("Phase", data=[1, 0, 2])
                data.create_dataset("Band Contrast", data=[80, 20, 90])
            with H5OINAReader(path) as reader:
                eulers, phases, contrast = reader.grain_inputs()
                self.assertEqual(eulers.shape, (3, 3))
                np.testing.assert_array_equal(phases, [1, 0, 2])
                np.testing.assert_array_equal(contrast, [80, 20, 90])

    def test_pc_plane_recovers_scan_drift_and_ignores_bad_point(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.h5oina"
            self._make_file(path, np.zeros((12, 8, 8), dtype=np.uint8))
            with h5py.File(path, "a") as file:
                header = file["1/EBSD/Header"]
                header["X Cells"][...] = 4
                header["Y Cells"][...] = 3
                data = file["1/EBSD/Data"]
                for axis, name in enumerate(("Pattern Center X", "Pattern Center Y",
                                             "Detector Distance")):
                    values = np.array([0.4 + axis * 0.1 + 0.001 * (i % 4) +
                                       0.002 * (i // 4) for i in range(12)])
                    values[5] += 0.2
                    data[name][...] = values[:, None]
            with H5OINAReader(path) as reader:
                plane = fit_pc_plane(reader)
                self.assertEqual(plane.points_used, 11)
                np.testing.assert_allclose(plane.at(10, 4),
                                           [0.406, 0.506, 0.606], atol=1e-12)
                self.assertLess(max(plane.rms_residual_pixels), 1e-10)

    def test_beam_shift_pc_plane_recovers_measured_pattern_translation(self):
        from scipy.ndimage import gaussian_filter, shift

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.h5oina"
            base = gaussian_filter(np.random.default_rng(47).normal(size=(128, 128)), 1.2)
            patterns = np.stack([
                shift(base, (0.20 * (i // 10), -0.25 * (i % 10)),
                      order=3, mode="nearest").astype(np.float32)
                for i in range(100)])
            self._make_file(path, patterns)
            with h5py.File(path, "a") as file:
                header = file["1/EBSD/Header"]
                header["X Cells"][...] = 10
                header["Y Cells"][...] = 10
                header.create_dataset("X Step", data=[1.0])
                header.create_dataset("Y Step", data=[1.0])
            with H5OINAReader(path) as reader:
                plane = fit_beam_shift_pc_plane(reader, reader, 0, roi_size=64,
                                                spacing=1, extent_um=9)
                reference_pc = reader.pattern_center(0)
                np.testing.assert_allclose(plane.at(0, 10), reference_pc, atol=1e-12)
                np.testing.assert_allclose(plane.coefficients[0, 1] * 128, -0.25,
                                           atol=0.025)
                np.testing.assert_allclose(plane.coefficients[1, 2], 0.0,
                                           atol=1e-12)
                np.testing.assert_allclose(
                    plane.report()["beam_shift_calibration"]["y_line"]
                    ["slope_pixels_per_map_step"][1], 0.20, atol=0.025)
                self.assertEqual(plane.report()["method"],
                                 "center_roi_beam_shift_calibration")
                calibration = measure_effective_pixel_size(
                    reader, 0, roi_size_percent=50, spacing=1, extent_um=9)
                self.assertAlmostEqual(calibration["effective_pixel_size_um_per_pixel"],
                                       4.0, delta=0.35)
                self.assertEqual(calibration["detector_x_shift_sign"], -1)
                external = pc_plane_from_effective_pixel_size(
                    reader, 0, calibration["effective_pixel_size_um_per_pixel"], -1)
                np.testing.assert_allclose(external.at(0, 10), reference_pc, atol=1e-12)
                np.testing.assert_allclose(external.coefficients[0, 1] * 128,
                                           -0.25, atol=0.025)
                np.testing.assert_allclose(external.coefficients[1, 2], 0.0,
                                           atol=1e-12)
            output = Path(directory) / "calibration_output"
            report = run({"run_mode": "calibration",
                          "h5oina_file": str(path),
                          "output_dir": str(output),
                          "reference_map_point": (4, 0),
                          "roi_size_percent": 50.0})
            self.assertAlmostEqual(report["effective_pixel_size_um_per_pixel"],
                                   4.0, delta=0.35)
            self.assertEqual(report["line_fit"]["line_start"], [0, 0])
            self.assertEqual(report["line_fit"]["line_end"], [9, 0])
            self.assertEqual(report["line_fit"]["points_total"], 10)
            self.assertTrue((output / "beam_shift_calibration.json").is_file())
            self.assertTrue((output / "beam_shift_calibration.png").is_file())

    def test_oxford_geometry_angles_are_read_in_degrees(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.h5oina"
            self._make_file(path, np.zeros((2, 8, 8), dtype=np.uint8))
            with h5py.File(path, "a") as file:
                header = file["1/EBSD/Header"]
                header.create_dataset("Tilt Angle", data=[np.deg2rad(70)])
                header.create_dataset("Detector Orientation Euler",
                                      data=[np.deg2rad([7, 98, -4])])
            with H5OINAReader(path) as reader:
                self.assertAlmostEqual(reader.sample_tilt_degrees(), 70)
                self.assertAlmostEqual(reader.camera_elevation_degrees(), 8)
                np.testing.assert_allclose(
                    reader.phosphor_to_sample(),
                    phosphor_to_sample_from_oxford(
                        np.deg2rad(70), np.deg2rad([7, 98, -4])), atol=1e-14)
                np.testing.assert_allclose(
                    reader.phosphor_to_sample(60),
                    phosphor_to_sample_from_oxford(
                        np.deg2rad(60), np.deg2rad([7, 98, -4])), atol=1e-14)


if __name__ == "__main__":
    unittest.main()
