import unittest

import numpy as np

from pyhrebsd.preprocess import (correct_pattern_background,
                               correct_pattern_static_lmsd,
                               local_mean_std_deviation)


class PreprocessTests(unittest.TestCase):
    def test_divide_background_removes_broad_shading(self):
        y, x = np.mgrid[:256, :256]
        shading = 8000 + 18*x + 8*y
        detail = 1 + 0.12*np.sin(2*np.pi*x/8)
        raw = shading*detail
        corrected = correct_pattern_background(raw, "divide_gaussian", 48, 4)
        self.assertLess(abs(np.median(corrected[32:-32, 32:-32])), 0.01)
        left = np.std(corrected[32:-32, 32:96])
        right = np.std(corrected[32:-32, 160:224])
        self.assertLess(abs(left/right-1), 0.05)

    def test_none_preserves_raw_values(self):
        raw = np.arange(64, dtype=np.int16).reshape(8, 8)
        np.testing.assert_array_equal(correct_pattern_background(raw, "none"), raw)
        with self.assertRaises(ValueError):
            correct_pattern_background(raw, "unknown")

    def test_fast_lmsd_matches_edge_replicating_pixel_loops(self):
        image = np.random.default_rng(31).normal(size=(6, 7))
        expected = np.empty_like(image)
        radius = 2
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                window = np.array([
                    image[np.clip(row+dy, 0, image.shape[0]-1),
                          np.clip(col+dx, 0, image.shape[1]-1)]
                    for dy in range(-radius, radius+1)
                    for dx in range(-radius, radius+1)])
                expected[row, col] = (image[row, col]-window.mean())/window.std()
        np.testing.assert_allclose(local_mean_std_deviation(image, radius),
                                   expected, atol=1e-12)

    def test_static_lmsd_outputs_full_resolution_uint16(self):
        y, x = np.mgrid[:64, :72]
        pattern = (3000 + 10*x + 8*y)*(1+0.07*np.sin(x/2))
        static = 2500 + 5*x + 3*y
        corrected = correct_pattern_static_lmsd(pattern, static)
        self.assertEqual(corrected.dtype, np.uint16)
        self.assertEqual(corrected.shape, pattern.shape)
        self.assertGreater(int(corrected.max()), int(corrected.min()))


if __name__ == "__main__":
    unittest.main()
