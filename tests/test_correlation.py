import unittest
import importlib.util
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import shift

from pyhrebsd.correlation import (annular_rois, grid_rois, measure_pattern_shifts,
                                 roi_size_from_percent, subpixel_shift,
                                 prepare_reference)


class CorrelationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("cupy"), "CuPy is not installed")
    def test_gpu_batch_matches_cpu_filtered_roi_shifts(self):
        from pyhrebsd.analysis import roi_frequency_filter
        from pyhrebsd.correlation_gpu import (measure_pattern_shifts_gpu,
                                             prepare_reference_gpu)
        reference = np.random.default_rng(41).normal(size=(128, 128))
        scan = shift(reference, (1.4, -2.3), order=3, mode="wrap")
        centers = annular_rois(reference.shape, 32, 12)
        scan_centers = centers + np.array([1.0, -1.0])
        frequency_filter, window = roi_frequency_filter(32, 2, 12)
        cpu_cache = prepare_reference(reference, centers, 32, frequency_filter, window)
        gpu_cache = prepare_reference_gpu(reference, centers, 32, frequency_filter, window)
        options = dict(scan_centers=scan_centers, frequency_filter=frequency_filter,
                       window=window)
        cpu = measure_pattern_shifts(reference, scan, centers, 32,
                                     prepared_reference=cpu_cache, **options)
        gpu = measure_pattern_shifts_gpu(reference, scan, centers, 32,
                                         prepared_reference=gpu_cache, **options)
        np.testing.assert_allclose(
            [[r.dx, r.dy, r.coefficient, r.peak_confidence] for r in gpu],
            [[r.dx, r.dy, r.coefficient, r.peak_confidence] for r in cpu],
            atol=1e-8)
        cpu_2d = measure_pattern_shifts(reference, scan, centers, 32,
                                        prepared_reference=cpu_cache,
                                        subpixel_method="quadratic_2d", **options)
        gpu_2d = measure_pattern_shifts_gpu(reference, scan, centers, 32,
                                            prepared_reference=gpu_cache,
                                            subpixel_method="quadratic_2d", **options)
        np.testing.assert_allclose(
            [[r.dx, r.dy, r.coefficient, r.peak_confidence] for r in gpu_2d],
            [[r.dx, r.dy, r.coefficient, r.peak_confidence] for r in cpu_2d],
            atol=1e-8)

    def test_annular_centers_and_relative_size(self):
        size = roi_size_from_percent((1024, 1024), 25)
        self.assertEqual(size, 256)
        points = annular_rois((1024, 1024), size, 48)
        self.assertEqual(points.shape, (48, 2))
        np.testing.assert_allclose(points[0], [511, 511])
        np.testing.assert_allclose(points[1], [767, 511])
        np.testing.assert_allclose(np.linalg.norm(points[1:] - points[0], axis=1), 256)
        self.assertEqual(roi_size_from_percent((512, 512), 25), 128)

    def test_annular_rois_measure_known_translation(self):
        rng = np.random.default_rng(23)
        reference = rng.normal(size=(128, 128))
        scan = np.roll(reference, (2, -3), axis=(0, 1))
        points = annular_rois(reference.shape, 32, 8)
        results = measure_pattern_shifts(reference, scan, points, 32)
        np.testing.assert_allclose([(r.dx, r.dy) for r in results], [(-3, 2)] * 8,
                                   atol=0.2)

    def test_integer_translation(self):
        rng = np.random.default_rng(4)
        reference = rng.normal(size=(128, 128))
        scan = np.roll(reference, (3, -2), axis=(0, 1))
        centers = grid_rois(reference.shape, 32, 16)
        results = measure_pattern_shifts(reference, scan, centers, 32)
        self.assertEqual(len(results), 16)
        np.testing.assert_allclose([(r.dx, r.dy) for r in results], [(-2, 3)] * 16, atol=0.2)

    def test_fractional_translation(self):
        rng = np.random.default_rng(5)
        reference = rng.normal(size=(128, 128))
        scan = shift(reference, (1.4, -2.3), order=3, mode="wrap")
        results = measure_pattern_shifts(reference, scan, [[63, 63]], 64)
        self.assertAlmostEqual(results[0].dx, -2.3, delta=0.35)
        self.assertAlmostEqual(results[0].dy, 1.4, delta=0.35)

    def test_flat_peak_is_finite(self):
        image = np.zeros((9, 9))
        image[4, 4] = 1
        self.assertEqual(subpixel_shift(image), (0.0, 0.0))

    def test_quadratic_2d_peak_recovers_coupled_shift(self):
        y, x = np.mgrid[-4:5, -4:5]
        true_x, true_y = 0.35, -0.25
        xx, yy = x-true_x, y-true_y
        peak = 10 - 1.4*xx**2 - 1.1*yy**2 + 0.55*xx*yy
        dx, dy = subpixel_shift(peak, "quadratic_2d")
        self.assertAlmostEqual(dx, -true_x, places=12)
        self.assertAlmostEqual(dy, -true_y, places=12)
        with self.assertRaises(ValueError):
            subpixel_shift(peak, "unknown")

    def test_command_writes_measured_shifts(self):
        rng = np.random.default_rng(6)
        reference = rng.integers(0, 256, size=(96, 96), dtype=np.uint8)
        scan = np.roll(reference, (2, 1), axis=(0, 1))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            Image.fromarray(reference).save(str(directory / "reference.tif"))
            Image.fromarray(scan).save(str(directory / "scan.tif"))
            output = directory / "shifts.csv"
            subprocess.run(
                [sys.executable, "-m", "pyhrebsd", str(directory / "reference.tif"),
                 str(directory / "scan.tif"), "--roi-size", "32", "--roi-count", "9",
                 "--output", str(output)],
                check=True, capture_output=True, text=True,
            )
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 9)
            self.assertAlmostEqual(float(rows[4]["dx"]), 1, delta=0.2)
            self.assertAlmostEqual(float(rows[4]["dy"]), 2, delta=0.2)


if __name__ == "__main__":
    unittest.main()
