import unittest

import numpy as np

from pyhrebsd.frame_views import rotate_tensor_maps, rotate_vector_maps


class FrameViewTests(unittest.TestCase):
    def test_quarter_turn_changes_components_not_tensor_magnitude(self):
        prefix = "deviatoric_strain_sample"
        values = {(1, 1): 1.0, (1, 2): 2.0, (1, 3): 3.0,
                  (2, 2): 4.0, (2, 3): 5.0, (3, 3): 6.0}
        maps = {f"{prefix}_{i}{j}": np.array([[value]])
                for (i, j), value in values.items()}
        rotated = rotate_tensor_maps(maps, prefix, 90)
        expected = {(1, 1): 4.0, (1, 2): -2.0, (1, 3): -5.0,
                    (2, 2): 1.0, (2, 3): 3.0, (3, 3): 6.0}
        for (i, j), value in expected.items():
            np.testing.assert_allclose(rotated[f"{prefix}_{i}{j}"], [[value]], atol=1e-14)
        self.assertEqual(maps[f"{prefix}_11"][0, 0], 1.0)

    def test_quarter_turn_rotates_lattice_rotation_vector(self):
        maps = np.array([[[4.0]], [[-6.0]], [[8.0]]])
        np.testing.assert_allclose(rotate_vector_maps(maps, 90).reshape(3),
                                   [6.0, 4.0, 8.0], atol=1e-14)


if __name__ == "__main__":
    unittest.main()
