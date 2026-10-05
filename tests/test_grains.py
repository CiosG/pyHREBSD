import unittest
import csv
import tempfile
from pathlib import Path

import numpy as np

from pyhrebsd.geometry import euler_to_matrix
from pyhrebsd.grains import _orientation_matrices, cubic_symmetry, segment_grains
from segment_grains import save_grain_map


class GrainTests(unittest.TestCase):
    def test_vectorized_euler_and_cubic_symmetry(self):
        angles = np.array([[0.1, 0.2, 0.3], [1.7, 0.8, 2.1]])
        np.testing.assert_allclose(_orientation_matrices(angles),
                                   np.stack([euler_to_matrix(*row) for row in angles]))
        rotations = cubic_symmetry()
        self.assertEqual(rotations.shape, (24, 3, 3))
        np.testing.assert_allclose(np.linalg.det(rotations), 1)

    def test_cubic_equivalents_are_one_grain_but_true_boundary_splits(self):
        # A 90-degree z rotation is a cubic symmetry operation; 12 degrees is not.
        angles = np.array([[0, 0, 0], [np.pi / 2, 0, 0],
                           [np.pi / 2 + np.deg2rad(12), 0, 0]], dtype=float)
        phases = np.ones(3, dtype=int)
        cubic = segment_grains(angles, phases, (1, 3), 5, 1, "cubic")
        np.testing.assert_array_equal(cubic.labels, [[1, 1, 2]])
        none = segment_grains(angles, phases, (1, 3), 5, 1, "none")
        self.assertEqual(none.count, 3)

    def test_phase_and_unindexed_points_do_not_join(self):
        angles = np.zeros((5, 3))
        grains = segment_grains(angles, np.array([1, 2, 0, 2, 2]),
                                (1, 5), 5, 1)
        np.testing.assert_array_equal(grains.labels, [[1, 2, 0, 3, 3]])
        self.assertEqual(grains.sizes[0], 1)
        np.testing.assert_array_equal(grains.phase_ids, [0, 1, 2, 2])

    def test_small_inclusion_merges_with_neighbor(self):
        angles = np.zeros((9, 3))
        angles[4, 0] = np.deg2rad(15)
        grains = segment_grains(angles, np.ones(9), (3, 3), 5, 2)
        self.assertEqual(grains.count, 1)
        self.assertEqual(grains.sizes[1], 9)

    def test_exports_per_point_grain_ids(self):
        grains = segment_grains(np.zeros((4, 3)), [1, 1, 0, 2], (2, 2), 5, 1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_grain_map(grains, output, np.array([50, 60, 0, 80]))
            with (output / "grain_ids.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["grain_id"] for row in rows], ["1", "1", "0", "2"])
            self.assertTrue((output / "grain_map.png").is_file())


if __name__ == "__main__":
    unittest.main()
