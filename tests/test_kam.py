import unittest

import numpy as np

from pyhrebsd.kam import hr_kam


class HRKAMTests(unittest.TestCase):
    def test_first_neighbors_average_exact_rotation_angles(self):
        vectors = np.zeros((2, 2, 3))
        vectors[:, :, 2] = np.deg2rad([[0, 1], [2, 3]]) * 1000
        kam, counts = hr_kam(vectors, np.ones((2, 2), bool),
                             np.ones((2, 2), int), None)
        np.testing.assert_allclose(kam, 1.5)
        np.testing.assert_array_equal(counts, 2)

    def test_quality_grain_and_pair_threshold_exclude_neighbors(self):
        vectors = np.zeros((1, 4, 3))
        vectors[0, :, 2] = np.deg2rad([0, 1, 4, 5]) * 1000
        grains = np.array([[1, 1, 1, 2]])
        kam, counts = hr_kam(vectors, np.ones((1, 4), bool), grains, 2)
        np.testing.assert_allclose(kam[0, :2], [1, 1])
        self.assertTrue(np.isnan(kam[0, 2:]).all())
        np.testing.assert_array_equal(counts, [[1, 1, 0, 0]])

        valid = np.array([[True, False, True, True]])
        kam, counts = hr_kam(vectors, valid, grains, None)
        self.assertTrue(np.isnan(kam).all())
        np.testing.assert_array_equal(counts, 0)

    def test_empty_valid_mask_and_threshold_are_safe(self):
        vectors = np.zeros((1, 2, 3))
        grains = np.ones((1, 2), int)
        kam, counts = hr_kam(vectors, np.zeros((1, 2), bool), grains, None)
        self.assertTrue(np.isnan(kam).all())
        np.testing.assert_array_equal(counts, 0)

        vectors[0, 1, 2] = np.deg2rad(20) * 1000
        kam, counts = hr_kam(vectors, np.ones((1, 2), bool), grains, 5)
        self.assertTrue(np.isnan(kam).all())
        np.testing.assert_array_equal(counts, 0)

    def test_composes_noncommuting_rotations_not_vector_distance(self):
        vectors = np.zeros((1, 2, 3))
        vectors[0, 0, 0] = np.pi * 500
        vectors[0, 1, 1] = np.pi * 500
        kam, _ = hr_kam(vectors, np.ones((1, 2), bool),
                        np.ones((1, 2), int), None)
        np.testing.assert_allclose(kam, 120, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
