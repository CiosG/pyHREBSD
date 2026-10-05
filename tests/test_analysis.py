import unittest
from pathlib import Path
import tempfile
import json

import numpy as np
import h5py
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation

from pyhrebsd.analysis import Material, analyze_pair, fit_deformation, prepare_analysis
from pyhrebsd.correlation import CorrelationResult, grid_rois
from pyhrebsd.geometry import theoretical_pixel_shift
from pyhrebsd.remapping import remap_pattern
from run_pyhrebsd import run


MATERIAL_DATABASE = (Path(__file__).resolve().parent.parent / "pyhrebsd" /
                     "materials.h5")


class AnalysisTests(unittest.TestCase):
    def test_hdf5_material_database(self):
        material = Material.from_hdf5(MATERIAL_DATABASE, "copper")
        stiffness = material.stiffness()
        self.assertEqual(material.lattice, "cubic")
        self.assertEqual(material.verification_status, "verified_reference")
        self.assertAlmostEqual(stiffness[0, 0, 0, 0], 169.68)
        self.assertAlmostEqual(stiffness[0, 0, 1, 1], 122.55)
        self.assertAlmostEqual(stiffness[0, 1, 0, 1], 74.499)
        self.assertIn("silicon", Material.list_hdf5(MATERIAL_DATABASE))
        with h5py.File(MATERIAL_DATABASE, "r") as database:
            labels = database["component_labels"].asstr()[()]
            self.assertEqual(labels[0, 0], "C11")
            self.assertEqual(labels[0, 1], "C12")
            self.assertEqual(labels[3, 3], "C44")
            self.assertEqual(labels[5, 5], "C66")
            silicon = database["materials/silicon"]
            self.assertEqual(silicon.attrs["reference_path"],
                             "/references/ledbetter_kim_2001")
            reference = database[silicon.attrs["reference_path"]]
            self.assertEqual(reference.attrs["doi"],
                             "10.1016/B978-012445760-7/50033-2")

    def test_hdf5_general_anisotropic_matrix_mapping(self):
        material = Material.from_hdf5(MATERIAL_DATABASE, "titanium")
        voigt = material.stiffness_voigt_gpa
        tensor = material.stiffness()
        self.assertAlmostEqual(tensor[0, 0, 2, 2], voigt[0, 2])
        self.assertAlmostEqual(tensor[2, 2, 0, 0], voigt[2, 0])
        self.assertAlmostEqual(tensor[1, 2, 1, 2], voigt[3, 3])

    def test_phase_specific_semiconductor_materials(self):
        gan = Material.from_hdf5(MATERIAL_DATABASE, "gallium_nitride_wurtzite")
        sic_3c = Material.from_hdf5(MATERIAL_DATABASE, "silicon_carbide_3c")
        sic_4h = Material.from_hdf5(MATERIAL_DATABASE, "silicon_carbide_4h")
        self.assertAlmostEqual(gan.c11, 390.0)
        self.assertAlmostEqual(gan.c33, 398.0)
        self.assertAlmostEqual(sic_3c.c44, 236.0)
        self.assertAlmostEqual(sic_4h.c13, 52.0)
        with h5py.File(MATERIAL_DATABASE, "r") as database:
            group = database["materials/gallium_nitride_wurtzite"]
            self.assertIn("stiffness_uncertainty_gpa", group)
            self.assertEqual(group.attrs["reference_doi"], "10.1063/1.361236")

    def test_copper_material_constants(self):
        material = Material.from_hdf5(MATERIAL_DATABASE, "copper")
        stiffness = material.stiffness()
        self.assertEqual(material.lattice, "cubic")
        self.assertAlmostEqual(stiffness[0, 0, 0, 0], 169.68)
        self.assertAlmostEqual(stiffness[0, 0, 1, 1], 122.55)
        self.assertAlmostEqual(stiffness[0, 1, 0, 1], 74.499)

    def test_recovers_known_stress_free_deformation(self):
        material = Material.from_hdf5(MATERIAL_DATABASE, "copper")
        pc = (0.45, 0.53, 0.65)
        points = grid_rois((128, 128), 24, 49)
        deformation = np.eye(3)
        deformation[0, 0] += 0.001
        deformation[2, 2] -= material.c12 / material.c11 * 0.001
        alpha = 0.52
        shifts = theoretical_pixel_shift(np.eye(3), pc, points, deformation, 128, alpha)
        rows = [CorrelationResult(x, y, dx, dy, 1, 10)
                for (x, y), (dx, dy) in zip(points, shifts)]
        result = fit_deformation(rows, 128, pc, pc, np.eye(3), np.pi / 2 - alpha,
                                 0.0, material)
        np.testing.assert_allclose(result.deformation, deformation, atol=1e-11)
        self.assertAlmostEqual(result.stress_gpa[2, 2], 0, delta=1e-10)
        self.assertLess(result.rms_shift_error, 1e-10)

    def test_full_detector_geometry_recovers_nonzero_azimuth_and_roll(self):
        from pyhrebsd.geometry import phosphor_to_sample_from_oxford

        material = Material.from_hdf5(MATERIAL_DATABASE, "copper")
        pc = (0.45, 0.53, 0.65)
        points = grid_rois((128, 128), 24, 49)
        deformation = np.eye(3)
        deformation[0, 0] += 0.001
        deformation[2, 2] -= material.c12 / material.c11 * 0.001
        tilt = np.deg2rad(70)
        detector_euler = np.deg2rad([7.0, 98.0, -4.0])
        qps = phosphor_to_sample_from_oxford(tilt, detector_euler)
        shifts = theoretical_pixel_shift(
            np.eye(3), pc, points, deformation, 128, 0.0,
            phosphor_to_sample=qps)
        rows = [CorrelationResult(x, y, dx, dy, 1, 10)
                for (x, y), (dx, dy) in zip(points, shifts)]
        result = fit_deformation(
            rows, 128, pc, pc, np.eye(3), tilt, np.deg2rad(8), material,
            phosphor_to_sample=qps)
        np.testing.assert_allclose(result.deformation, deformation, atol=1e-11)
        self.assertLess(result.rms_shift_error, 1e-10)

    def test_identical_patterns_return_identity(self):
        material = Material.from_hdf5(MATERIAL_DATABASE, "copper")
        rng = np.random.default_rng(12)
        pattern = rng.normal(size=(128, 128))
        result = analyze_pair(
            pattern, pattern, roi_size=24, roi_count=49,
            reference_pc=(0.45, 0.53, 0.65), scan_pc=(0.45, 0.53, 0.65),
            orientation=[0, 0, 0], sample_tilt=np.deg2rad(70),
            camera_elevation=np.deg2rad(10), material=material,
            roi_filter=(2, 10, True, True),
        )
        np.testing.assert_allclose(result.deformation, np.eye(3), atol=1e-8)
        self.assertLess(result.rms_shift_error, 1e-8)

    def test_annular_prepared_analysis_uses_requested_count(self):
        material = Material.from_hdf5(MATERIAL_DATABASE, "copper")
        pattern = np.random.default_rng(29).normal(size=(128, 128))
        plan = prepare_analysis(pattern, 32, 12, None, "annular")
        result = analyze_pair(
            pattern, pattern, roi_size=32, roi_count=12, roi_layout="annular",
            reference_pc=(0.45, 0.53, 0.65), scan_pc=(0.45, 0.53, 0.65),
            orientation=[0, 0, 0], sample_tilt=np.deg2rad(70),
            camera_elevation=np.deg2rad(10), material=material, prepared=plan,
        )
        self.assertEqual(result.total_rois, 12)
        np.testing.assert_allclose(result.deformation, np.eye(3), atol=1e-8)

    def test_pattern_center_correction_removes_rigid_shift(self):
        material = Material.from_hdf5(MATERIAL_DATABASE, "copper")
        reference_pc = (0.45, 0.53, 0.65)
        scan_pc = (0.46, 0.52, 0.66)
        points = grid_rois((128, 128), 24, 49)
        r0 = np.column_stack((reference_pc[0] - (points[:, 0] + 1) / 128,
                              1 - reference_pc[1] - (points[:, 1] + 1) / 128,
                              np.full(len(points), -reference_pc[2])))
        ratio = reference_pc[2] / scan_pc[2]
        translation = np.array([reference_pc[0] - scan_pc[0],
                                scan_pc[1] - reference_pc[1],
                                scan_pc[2] - reference_pc[2]])
        apparent = -(translation + r0 * (1 - ratio) / ratio)[:, :2] * 128
        rows = [CorrelationResult(x, y, dx, dy, 1, 10)
                for (x, y), (dx, dy) in zip(points, apparent)]
        result = fit_deformation(rows, 128, reference_pc, scan_pc, np.eye(3),
                                 np.deg2rad(70), np.deg2rad(10), material)
        np.testing.assert_allclose(result.deformation, np.eye(3), atol=1e-10)

    def test_configured_runner_writes_analysis(self):
        rng = np.random.default_rng(13)
        pattern = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            Image.fromarray(pattern).save(str(directory / "reference.tif"))
            Image.fromarray(pattern).save(str(directory / "scan.tif"))
            config = {
                "reference_image": str(directory / "reference.tif"),
                "scan_images": [{"path": str(directory / "scan.tif")}],
                "output_dir": str(directory / "results"),
                "material_database": str(MATERIAL_DATABASE),
                "material_name": "copper",
                "reference_pattern_center": (0.45, 0.53, 0.65),
                "reference_euler_degrees": (0, 0, 0),
                "sample_tilt_degrees": 70,
                "camera_elevation_degrees": 10,
                "roi_size": 24, "roi_count": 49,
                "roi_filter": None, "outlier_standard_deviation": 2,
            }
            run(config)
            summary = json.loads((directory / "results" / "0001_scan_analysis.json").read_text())
            self.assertLess(summary["rms_shift_error_pixels"], 1e-8)
            self.assertTrue((directory / "results" / "0001_scan_rois.csv").exists())

    def test_recovers_strain_from_warped_pattern_images(self):
        material = Material.from_hdf5(MATERIAL_DATABASE, "copper")
        size = 192
        pc = (0.45, 0.53, 0.65)
        reference = gaussian_filter(np.random.default_rng(1).normal(size=(size, size)), 1.2)
        expected = np.eye(3)
        expected[0, 0] += 0.001
        expected[2, 2] -= material.c12 / material.c11 * 0.001
        xx, yy = np.meshgrid(np.arange(size), np.arange(size))
        points = np.column_stack((xx.ravel(), yy.ravel()))
        sample_tilt = np.deg2rad(70)
        camera_elevation = np.deg2rad(10)
        alpha = np.pi / 2 - sample_tilt + camera_elevation
        inverse_shift = theoretical_pixel_shift(
            np.eye(3), pc, points, np.linalg.inv(expected), size, alpha,
        )
        source = points + inverse_shift
        scan = map_coordinates(reference, [source[:, 1].reshape(size, size),
                                           source[:, 0].reshape(size, size)],
                               order=3, mode="reflect")
        result = analyze_pair(
            reference, scan, roi_size=48, roi_count=49,
            reference_pc=pc, scan_pc=pc, orientation=np.eye(3),
            sample_tilt=sample_tilt, camera_elevation=camera_elevation,
            material=material, roi_filter=(2, 30, True, True),
        )
        np.testing.assert_allclose(result.deformation, expected, atol=1.2e-4)
        self.assertLess(result.rms_shift_error, 0.01)
        cached = analyze_pair(
            reference, scan, roi_size=48, roi_count=49,
            reference_pc=pc, scan_pc=pc, orientation=np.eye(3),
            sample_tilt=sample_tilt, camera_elevation=camera_elevation,
            material=material, roi_filter=(2, 30, True, True),
            prepared=prepare_analysis(reference, 48, 49, (2, 30, True, True)),
        )
        np.testing.assert_allclose(cached.deformation, result.deformation, atol=1e-12)
        self.assertAlmostEqual(cached.rms_shift_error, result.rms_shift_error, places=12)

    def test_remapping_recovers_rotation_and_reduces_residual_rotation(self):
        material = Material.from_hdf5(MATERIAL_DATABASE, "copper")
        side = 192
        pc = (0.45, 0.53, 0.65)
        reference = gaussian_filter(np.random.default_rng(38).normal(size=(side, side)), 1.1)
        rotation = Rotation.from_rotvec([0.012, -0.009, 0.018]).as_matrix()
        yy, xx = np.indices(reference.shape)
        points = np.column_stack((xx.ravel(), yy.ravel()))
        alpha = np.deg2rad(30)
        source = points + theoretical_pixel_shift(
            np.eye(3), pc, points, rotation.T, side, alpha)
        scan = map_coordinates(reference,
                               [source[:, 1].reshape(side, side),
                                source[:, 0].reshape(side, side)],
                               order=3, mode="reflect")
        kwargs = dict(roi_size=48, roi_count=49, reference_pc=pc, scan_pc=pc,
                      orientation=np.eye(3), sample_tilt=np.pi/2-alpha,
                      camera_elevation=0, material=material,
                      roi_filter=(2, 30, True, True))
        result = analyze_pair(reference, scan, remapping=True, **kwargs)
        self.assertIsNotNone(result.first_pass_shifts)
        self.assertGreater(result.remapping_rotation_mrad, 10)
        self.assertLess(result.residual_rotation_mrad, 3)
        np.testing.assert_allclose(result.deformation, rotation, atol=0.004)
        self.assertLess(np.linalg.norm(result.strain), 0.002)

    def test_remapping_identity_with_different_pattern_centres(self):
        side = 128
        reference_pc = (0.45, 0.53, 0.65)
        scan_pc = (0.47, 0.51, 0.67)
        yy, xx = np.indices((side, side))
        reference = xx + 2*yy
        ray_x = scan_pc[0] - (xx + 1)/side
        ray_y = 1 - scan_pc[1] - (yy + 1)/side
        ratio = reference_pc[2]/scan_pc[2]
        source_x = (reference_pc[0] - ray_x*ratio)*side - 1
        source_y = (1 - reference_pc[1] - ray_y*ratio)*side - 1
        scan = source_x + 2*source_y
        remapped, valid = remap_pattern(
            scan, reference_pc, scan_pc, np.eye(3))
        np.testing.assert_allclose(remapped[valid], reference[valid], atol=2e-4)

    def test_gpu_remapping_pair_matches_cpu(self):
        try:
            import cupy
            cupy.cuda.runtime.getDeviceCount()
        except (ImportError, RuntimeError):
            self.skipTest("CuPy/CUDA is unavailable")
        material = Material.from_hdf5(MATERIAL_DATABASE, "copper")
        image = gaussian_filter(np.random.default_rng(42).normal(size=(128, 128)), 1.0)
        pc = (0.45, 0.53, 0.65)
        yy, xx = np.indices(image.shape)
        points = np.column_stack((xx.ravel(), yy.ravel()))
        rotation = Rotation.from_rotvec([0.007, -0.005, 0.009]).as_matrix()
        source = points + theoretical_pixel_shift(
            np.eye(3), pc, points, rotation.T, 128, np.deg2rad(30))
        scan = map_coordinates(image,
                               [source[:, 1].reshape(image.shape),
                                source[:, 0].reshape(image.shape)],
                               order=3, mode="reflect")
        kwargs = dict(roi_size=24, roi_count=49, reference_pc=pc, scan_pc=pc,
                      orientation=np.eye(3), sample_tilt=np.deg2rad(70),
                      camera_elevation=np.deg2rad(10), material=material,
                      remapping=True)
        cpu = analyze_pair(image, scan, **kwargs)
        gpu = analyze_pair(image, scan, device="gpu", **kwargs)
        np.testing.assert_allclose(gpu.deformation, cpu.deformation, atol=1e-8)
        self.assertEqual(gpu.used_rois, cpu.used_rois)


if __name__ == "__main__":
    unittest.main()
