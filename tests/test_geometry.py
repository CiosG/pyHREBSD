import unittest

import numpy as np

from pyhrebsd.geometry import (euler_to_matrix, phosphor_to_sample_from_oxford,
                              theoretical_pixel_shift)


class GeometryTests(unittest.TestCase):
    def test_euler_matrix_is_rotation(self):
        matrix = euler_to_matrix(0.2, 0.3, -0.4)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(matrix), 1)

    def test_identity_deformation_has_zero_shift(self):
        centers = np.array([[25, 35], [60, 50], [80, 85]])
        shifts = theoretical_pixel_shift(
            [0.2, 0.3, -0.4], (0.45, 0.53, 0.65), centers, np.eye(3), 128,
            np.pi / 2 - np.deg2rad(70) + np.deg2rad(5),
        )
        np.testing.assert_allclose(shifts, np.zeros((3, 2)), atol=1e-12)

    def test_euler_and_matrix_inputs_agree(self):
        angles = [0.2, 0.3, -0.4]
        centers = np.array([[25, 35], [60, 50], [80, 85]])
        deformation = np.eye(3)
        deformation[0, 1] = 0.01
        args = ((0.45, 0.53, 0.65), centers, deformation, 128, 0.5)
        np.testing.assert_allclose(
            theoretical_pixel_shift(angles, *args),
            theoretical_pixel_shift(euler_to_matrix(*angles), *args),
        )

    def test_full_detector_transform_matches_scalar_at_zero_azimuth_and_roll(self):
        tilt = np.deg2rad(70)
        elev = np.deg2rad(8)
        alpha = np.pi / 2 - tilt + elev
        full = phosphor_to_sample_from_oxford(tilt, [0, np.pi / 2 + elev, 0])
        expected = np.array([[0, -np.cos(alpha), -np.sin(alpha)],
                             [-1, 0, 0], [0, np.sin(alpha), -np.cos(alpha)]])
        np.testing.assert_allclose(full, expected, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
