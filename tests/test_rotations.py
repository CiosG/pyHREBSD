import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from pyhrebsd.geometry import euler_to_matrix
from pyhrebsd.rotations import sample_rotation_vector_mrad
from plot_rotations import _shared_rotation_limit


class RotationTests(unittest.TestCase):
    def test_rotation_panels_share_widest_component_limit(self):
        maps = np.array([[[0.0, 1.0]], [[-3.0, np.nan]], [[0.5, -0.5]]])
        self.assertEqual(_shared_rotation_limit(maps, 100), 3.0)

    def test_recovers_sample_rotation_with_crystal_orientation_and_stretch(self):
        g = euler_to_matrix(0.7, 0.4, 0.2)
        expected = np.array([4.0, -6.0, 8.0])
        r_sample = Rotation.from_rotvec(expected / 1000).as_matrix()
        stretch = np.diag([1.002, 0.999, 1.001])
        f_crystal = g @ (r_sample @ stretch) @ g.T
        np.testing.assert_allclose(sample_rotation_vector_mrad(f_crystal, g),
                                   expected, atol=1e-10)

    def test_improper_deformation_has_no_proper_rotation(self):
        f = np.diag([-1.0, 1.0, 1.0])
        self.assertTrue(np.isnan(sample_rotation_vector_mrad(f, np.eye(3))).all())


if __name__ == "__main__":
    unittest.main()
