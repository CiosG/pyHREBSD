import unittest

import numpy as np

from run_pyhrebsd import _bin_pattern, _binned_pc, _pattern_config, _pc_for_analysis


class PatternBinningTests(unittest.TestCase):
    def test_block_means_and_roi_size(self):
        image = np.arange(64, dtype=float).reshape(8, 8)
        binned = _bin_pattern(image, 2)
        np.testing.assert_allclose(binned, image.reshape(4, 2, 4, 2).mean(axis=(1, 3)))
        self.assertEqual(_pattern_config({"pattern_binning": 2,
                                          "roi_size_percent": 25}, (1024, 1024))["roi_size"],
                         128)

    def test_pc_preserves_detector_ray_at_block_centers(self):
        side = 1024
        pc = (0.45, 0.53, 0.65)
        for factor in (2, 4, 8):
            binned_pc = _binned_pc(pc, factor, side)
            for pixel in (0, 5, side // factor - 1):
                original_center = factor * pixel + (factor - 1) / 2
                self.assertAlmostEqual(
                    binned_pc[0] - (pixel + 1) / (side / factor),
                    pc[0] - (original_center + 1) / side)
                self.assertAlmostEqual(
                    1 - binned_pc[1] - (pixel + 1) / (side / factor),
                    1 - pc[1] - (original_center + 1) / side)

    def test_pc_already_calibrated_at_binned_resolution(self):
        pc = (0.463710218667984, 0.5433980822563171, 0.8486204743385315)
        self.assertEqual(_pc_for_analysis(pc, 2048, 2, 1024), pc)
        np.testing.assert_allclose(_pc_for_analysis(pc, 2048, 2),
                                   _binned_pc(pc, 2, 2048))


if __name__ == "__main__":
    unittest.main()
