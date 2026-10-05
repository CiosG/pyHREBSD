import importlib.util
import unittest

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from pyhrebsd.analysis import Material
from pyhrebsd.geometry import theoretical_pixel_shift
from pyhrebsd.homography import (analyze_homography, deformation_from_homography,
                               prepare_homography, register_homography, _warp_points)


class HomographyTests(unittest.TestCase):
    def setUp(self):
        self.material = Material("Si", "cubic", 165.7, 63.9, 79.6)
        self.pc = (0.45, 0.53, 0.65)
        self.tilt = np.deg2rad(70)
        self.elevation = np.deg2rad(10)

    def test_pc_shift_alone_is_identity_deformation(self):
        other = (0.46, 0.52, 0.67)
        from pyhrebsd.homography import _ray_matrix, _rotation_phosphor_to_sample
        q = _rotation_phosphor_to_sample(np.pi/2-self.tilt+self.elevation)
        h = np.linalg.inv(_ray_matrix(other, 128, q)) @ _ray_matrix(self.pc, 128, q)
        f, strain, _, _ = deformation_from_homography(
            h, self.pc, other, np.eye(3), self.tilt, self.elevation,
            self.material, 128)
        np.testing.assert_allclose(f, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(strain, 0, atol=1e-12)

    def test_homography_matches_openxy_ray_projection(self):
        from pyhrebsd.homography import _ray_matrix, _rotation_phosphor_to_sample
        n = 128
        f = np.eye(3)
        f[0, 1] = 0.002
        f[1, 2] = -0.001
        points = np.array([[20., 21.], [56., 70.], [100., 94.]])
        q = _rotation_phosphor_to_sample(np.pi/2-self.tilt+self.elevation)
        k = _ray_matrix(self.pc, n, q)
        h = np.linalg.inv(k) @ f @ k
        xy = np.column_stack((points, np.ones(len(points))))
        projected = (h @ xy.T).T
        projected = projected[:, :2]/projected[:, 2, None]
        shifts = theoretical_pixel_shift(np.eye(3), self.pc, points, f, n,
                                         np.pi/2-self.tilt+self.elevation)
        np.testing.assert_allclose(projected-points, shifts, atol=1e-11)

    def test_full_detector_geometry_recovers_nonzero_azimuth_and_roll(self):
        from pyhrebsd.geometry import phosphor_to_sample_from_oxford
        from pyhrebsd.homography import _ray_matrix

        n = 128
        qps = phosphor_to_sample_from_oxford(
            self.tilt, np.deg2rad([7.0, 98.0, -4.0]))
        expected = np.eye(3)
        expected[0, 0] += 0.001
        expected[2, 2] -= self.material.c12/self.material.c11*0.001
        k = _ray_matrix(self.pc, n, qps)
        h = np.linalg.inv(k) @ expected @ k
        deformation, _, stress, _ = deformation_from_homography(
            h, self.pc, self.pc, np.eye(3), self.tilt, self.elevation,
            self.material, n, phosphor_to_sample=qps)
        np.testing.assert_allclose(deformation, expected, atol=1e-12)
        self.assertAlmostEqual(stress[2, 2], 0.0, delta=1e-10)

    def test_recovers_projective_warp_and_stress_free_deformation(self):
        from pyhrebsd.homography import _ray_matrix, _rotation_phosphor_to_sample
        n = 160
        rng = np.random.default_rng(15)
        ref = gaussian_filter(rng.normal(size=(n, n)), 1.3)
        ref += 0.3*gaussian_filter(rng.normal(size=(n, n)), 4)
        expected = np.eye(3)
        expected[0, 0] += 0.002
        expected[2, 2] -= self.material.c12/self.material.c11*0.002
        q = _rotation_phosphor_to_sample(np.pi/2-self.tilt+self.elevation)
        k = _ray_matrix(self.pc, n, q)
        h = np.linalg.inv(k) @ expected @ k
        yy, xx = np.mgrid[:n, :n]
        points = np.column_stack((xx.ravel(), yy.ravel(), np.ones(n*n)))
        source = np.linalg.inv(h) @ points.T
        source /= source[2]
        scan = map_coordinates(ref, [source[1].reshape(n, n), source[0].reshape(n, n)],
                               order=3, mode="reflect")
        scan = scan*1.35+0.2
        result = analyze_homography(
            ref, scan, self.pc, self.pc, np.eye(3),
            self.tilt, self.elevation, self.material)
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.deformation, expected, atol=2e-4)
        self.assertLess(result.znssd_rms, 0.13)

    def test_identity_images(self):
        image = gaussian_filter(np.random.default_rng(4).normal(size=(96, 96)), 1)
        plan = prepare_homography(image)
        h, rms, _, converged = register_homography(plan, image)
        self.assertTrue(converged)
        np.testing.assert_allclose(h, np.eye(3), atol=1e-9)
        self.assertLess(rms, 1e-9)

    def test_plan_uses_every_pixel_inside_margin(self):
        image = gaussian_filter(np.random.default_rng(44).normal(size=(96, 96)), 1)
        plan = prepare_homography(image, margin_fraction=0.125)
        edge = int(96 * 0.125)
        self.assertEqual(plan.xy.shape, ((96 - 2*edge)**2, 2))

    @unittest.skipUnless(importlib.util.find_spec("cupy"), "CuPy is not installed")
    def test_gpu_matches_cpu_on_projective_pattern(self):
        n = 128
        image = gaussian_filter(np.random.default_rng(17).normal(size=(n, n)), 1.3)
        expected = np.array([[1.001, 0.002, 0.35],
                             [-0.001, 0.999, -0.25],
                             [2e-6, -1e-6, 1.]])
        yy, xx = np.mgrid[:n, :n]
        points = np.column_stack((xx.ravel(), yy.ravel(), np.ones(n*n)))
        source = np.linalg.inv(expected) @ points.T
        source /= source[2]
        scan = map_coordinates(image,
                               [source[1].reshape(n, n), source[0].reshape(n, n)],
                               order=3, mode="reflect")
        corners = np.array([[15., 15.], [112., 15.], [112., 112.], [15., 112.]])
        cpu = register_homography(prepare_homography(image), scan)
        gpu = register_homography(
            prepare_homography(image, device="gpu"), scan)
        np.testing.assert_allclose(_warp_points(gpu[0], corners),
                                   _warp_points(cpu[0], corners), atol=0.03)
        self.assertAlmostEqual(gpu[1], cpu[1], delta=0.01)
        self.assertEqual(gpu[3], cpu[3])


if __name__ == "__main__":
    unittest.main()
