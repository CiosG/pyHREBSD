"""ROI-remapped initialization for whole-pattern homography."""

from __future__ import annotations

import numpy as np

from .analysis import analyze_pair
from .homography import _ray_matrix


def roi_remapped_initial_homography(reference, scan, reference_pc, scan_pc,
                                    orientation, material, config, roi_plan):
    """Fit remapped ROI deformation and convert it to a homography start."""
    if roi_plan is None:
        return None
    result = analyze_pair(
        reference, scan,
        roi_size=roi_plan.reference.roi_size,
        roi_count=int(config.get("homography_prealignment_roi_count", 8)),
        roi_layout=config.get("homography_prealignment_roi_layout", "annular"),
        reference_pc=reference_pc, scan_pc=scan_pc, orientation=orientation,
        sample_tilt=np.deg2rad(config["sample_tilt_degrees"]),
        camera_elevation=np.deg2rad(config["camera_elevation_degrees"]),
        material=material,
        standard_deviation=config.get("outlier_standard_deviation", 2.0),
        roi_filter=config.get("roi_filter"), prepared=roi_plan,
        phosphor_to_sample=config.get("phosphor_to_sample"),
        device=roi_plan.device, gpu_device_id=roi_plan.gpu_device_id,
        subpixel_method="quadratic_2d", remapping=True,
    )
    size = reference.shape[0]
    alpha = (np.pi / 2 - np.deg2rad(config["sample_tilt_degrees"])
             + np.deg2rad(config["camera_elevation_degrees"]))
    qps = (config.get("phosphor_to_sample") if config.get("phosphor_to_sample") is not None
           else np.array([[0, -np.cos(alpha), -np.sin(alpha)],
                          [-1, 0, 0], [0, np.sin(alpha), -np.cos(alpha)]]))
    qpc = np.asarray(orientation) @ qps
    kr = _ray_matrix(reference_pc, size, qpc)
    ks = _ray_matrix(scan_pc, size, qpc)
    initial = np.linalg.inv(ks) @ result.deformation @ kr
    if roi_plan.device == "gpu":
        import cupy as cp
        cp.cuda.Device(roi_plan.gpu_device_id).synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    return initial
