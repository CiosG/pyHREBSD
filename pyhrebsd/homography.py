"""Whole-pattern projective DIC following Ernould et al. (2020, 2022).

The eight observable homography parameters are refined with inverse-compositional
Gauss-Newton and zero-mean, unit-variance image intensities.  The missing ninth
deformation degree of freedom is fixed by the free-surface normal stress.
Full citations and differences from the published workflow are documented in
``docs/references.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .analysis import Material, _rotation_phosphor_to_sample
from .geometry import euler_to_matrix


@dataclass(frozen=True)
class HomographyPlan:
    reference: np.ndarray
    xy: np.ndarray
    template: np.ndarray
    steepest: np.ndarray
    scale: float
    center: float
    device: str = "cpu"
    gpu_device_id: int = 0


def _preprocess(image, background_sigma):
    image = np.asarray(image, dtype=np.float64)
    image = gaussian_filter(image, 1.0)
    image -= gaussian_filter(image, background_sigma)
    return image


def prepare_homography(reference, *, margin_fraction=0.08,
                       background_sigma=None, device="cpu", gpu_device_id=0):
    """Cache reference gradients/Jacobian for one large, full-pixel region."""
    if device == "gpu":
        from .homography_gpu import prepare_gpu_homography
        return prepare_gpu_homography(
            reference, margin_fraction=margin_fraction,
            background_sigma=background_sigma, gpu_device_id=gpu_device_id)
    if device != "cpu":
        raise ValueError("homography device must be 'cpu' or 'gpu'")
    image = np.asarray(reference)
    if image.ndim != 2 or image.shape[0] != image.shape[1] or image.shape[0] < 32:
        raise ValueError("homography requires a square pattern of at least 32 pixels")
    if not 0 <= margin_fraction < 0.4:
        raise ValueError("invalid homography margin")
    size = image.shape[0]
    sigma = background_sigma or max(5.0, size / 25.0)
    ref = _preprocess(image, sigma)
    center, scale = (size - 1) / 2, size / 2
    edge = max(3, int(size * margin_fraction))
    y, x = np.mgrid[edge:size-edge, edge:size-edge]
    xp, yp = x.ravel().astype(float), y.ravel().astype(float)
    xn, yn = (xp-center)/scale, (yp-center)/scale
    values = ref[y.ravel(), x.ravel()]
    std = values.std()
    if std <= 1e-10:
        raise ValueError("reference pattern has no usable contrast")
    template = (values-values.mean())/std
    gy, gx = np.gradient(ref)
    gx = gx[y.ravel(), x.ravel()] * scale / std
    gy = gy[y.ravel(), x.ravel()] * scale / std
    jx = np.column_stack((xn, yn, np.ones_like(xn),
                          np.zeros_like(xn), np.zeros_like(xn), np.zeros_like(xn),
                          -xn*xn, -xn*yn))
    jy = np.column_stack((np.zeros_like(xn), np.zeros_like(xn), np.zeros_like(xn),
                          xn, yn, np.ones_like(xn), -xn*yn, -yn*yn))
    steepest = gx[:, None]*jx + gy[:, None]*jy
    return HomographyPlan(ref, np.column_stack((xn, yn)), template, steepest,
                          scale, center)


def _normalize_h(h):
    h = np.asarray(h, dtype=float)
    return h/h[2, 2]


def _warp_points(h, xy):
    d = h[2, 0]*xy[:, 0] + h[2, 1]*xy[:, 1] + h[2, 2]
    return np.column_stack(((h[0, 0]*xy[:, 0]+h[0, 1]*xy[:, 1]+h[0, 2])/d,
                            (h[1, 0]*xy[:, 0]+h[1, 1]*xy[:, 1]+h[1, 2])/d))


def register_homography(plan, scan, initial=None, *, max_iterations=40,
                        tolerance=1e-5, background_sigma=None,
                        trace_callback=None):
    """Map reference pixel coordinates into the scan with global IC-GN DIC.

    Returns the pixel-coordinate 3x3 homography, ZNSSD RMS, iteration count,
    and convergence flag.  ``initial`` is an optional pixel-coordinate warp.
    ``trace_callback`` receives (completed_iterations, pixel_homography,
    znssd_rms) at the initial state and after each update on CPU.
    """
    if plan.device == "gpu":
        if trace_callback is not None:
            raise ValueError("homography tracing is currently available on CPU only")
        from .homography_gpu import register_gpu_homography
        return register_gpu_homography(
            plan, scan, initial, max_iterations=max_iterations,
            tolerance=tolerance, background_sigma=background_sigma)
    image = np.asarray(scan)
    if image.shape != plan.reference.shape:
        raise ValueError("scan and reference patterns must have equal shape")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("invalid homography iteration settings")
    sigma = background_sigma or max(5.0, image.shape[0]/25.0)
    target = _preprocess(image, sigma)
    s = np.array([[1/plan.scale, 0, -plan.center/plan.scale],
                  [0, 1/plan.scale, -plan.center/plan.scale], [0, 0, 1]])
    s_inverse = np.linalg.inv(s)
    h = np.eye(3) if initial is None else _normalize_h(s @ initial @ np.linalg.inv(s))
    converged = False
    for iteration in range(1, max_iterations+1):
        uv = _warp_points(h, plan.xy)
        px, py = uv[:, 0]*plan.scale+plan.center, uv[:, 1]*plan.scale+plan.center
        valid = ((px >= 2) & (py >= 2) & (px < image.shape[1]-3) &
                 (py < image.shape[0]-3))
        if valid.sum() < 100 or valid.mean() < 0.5:
            raise ValueError("homography moves too much of the pattern outside the detector")
        warped = map_coordinates(target, [py[valid], px[valid]], order=1,
                                 mode="nearest", prefilter=False)
        sd = warped.std()
        if sd <= 1e-10:
            raise ValueError("scan pattern has no usable contrast")
        residual = (warped-warped.mean())/sd-plan.template[valid]
        if trace_callback is not None:
            trace_callback(iteration-1, _normalize_h(s_inverse @ h @ s).copy(),
                           float(np.sqrt(np.mean(residual**2))))
        jac = plan.steepest[valid]
        delta, _, rank, _ = np.linalg.lstsq(jac.T @ jac, jac.T @ residual, rcond=None)
        if rank < 8:
            raise ValueError("pattern cannot determine eight homography parameters")
        # Limit the first-order update to avoid stepping over diffraction bands.
        corners = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        for _ in range(12):
            dh = np.array([[1+delta[0], delta[1], delta[2]],
                           [delta[3], 1+delta[4], delta[5]],
                           [delta[6], delta[7], 1.]])
            movement = np.max(np.linalg.norm((_warp_points(dh, corners)-corners)*plan.scale,
                                             axis=1))
            if movement <= 3:
                break
            delta *= 0.5
        h = _normalize_h(h @ np.linalg.inv(dh))
        if movement < tolerance:
            converged = True
            break
    uv = _warp_points(h, plan.xy)
    px, py = uv[:, 0]*plan.scale+plan.center, uv[:, 1]*plan.scale+plan.center
    valid = ((px >= 2) & (py >= 2) & (px < image.shape[1]-3) &
             (py < image.shape[0]-3))
    warped = map_coordinates(target, [py[valid], px[valid]], order=1,
                             mode="nearest", prefilter=False)
    rms = float(np.sqrt(np.mean(((warped-warped.mean())/warped.std()
                                  -plan.template[valid])**2)))
    pixel_h = _normalize_h(s_inverse @ h @ s)
    if trace_callback is not None:
        trace_callback(iteration, pixel_h.copy(), rms)
    return pixel_h, rms, iteration, converged


def _ray_matrix(pc, size, qpc):
    x, y, z = pc
    if z <= 0:
        raise ValueError("pattern-center depth must be positive")
    detector = np.array([[-1., 0., x*size-1], [0., -1., (1-y)*size-1],
                         [0., 0., -z*size]])
    return qpc @ detector


def deformation_from_homography(h, reference_pc, scan_pc, orientation,
                                sample_tilt, camera_elevation, material,
                                size, phosphor_to_sample=None):
    """Recover crystal F using PC geometry and zero sample-normal stress.

    Homography fixes F up to a scalar.  The scalar is chosen so the linear
    elastic stress normal to the free sample surface vanishes.  This resolves
    the hydrostatic ambiguity only through the stated boundary assumption.
    """
    g = np.asarray(orientation, dtype=float)
    if g.shape == (3,):
        g = euler_to_matrix(*g)
    if g.shape != (3, 3):
        raise ValueError("orientation must be a matrix or three Euler angles")
    alpha = np.pi/2-sample_tilt+camera_elevation
    qps = (_rotation_phosphor_to_sample(alpha) if phosphor_to_sample is None
           else np.asarray(phosphor_to_sample, dtype=float))
    qpc = g @ qps
    kr = _ray_matrix(reference_pc, size, qpc)
    ks = _ray_matrix(scan_pc, size, qpc)
    f0 = ks @ _normalize_h(h) @ np.linalg.inv(kr)
    # Choose the positive projective gauge nearest identity first.
    f0 *= 3/np.trace(f0)
    c = material.stiffness()
    normal_crystal = g @ np.array([0., 0., 1.])
    # Polar stretch scales linearly with the projective gauge, so the normal
    # stress equation has an exact scalar solution even at finite rotation.
    _, singular0, vh0 = np.linalg.svd(f0)
    stretch0 = (vh0.T*singular0) @ vh0
    a = np.einsum('i,ijkl,kl,j->', normal_crystal, c, stretch0,
                  normal_crystal)
    b = np.einsum('i,ijkk,j->', normal_crystal, c, normal_crystal)
    f = f0*(b/a)
    _, singular, vh = np.linalg.svd(f)
    strain = (vh.T*singular) @ vh-np.eye(3)
    stress = np.einsum('ijkl,kl->ij', c, strain)
    return f, strain, stress, g


@dataclass(frozen=True)
class HomographyResult:
    deformation: np.ndarray
    strain: np.ndarray
    stress_gpa: np.ndarray
    orientation: np.ndarray
    homography: np.ndarray
    znssd_rms: float
    iterations: int
    converged: bool


def analyze_homography(reference, scan, reference_pc, scan_pc, orientation,
                       sample_tilt, camera_elevation, material, *, prepared=None,
                       margin_fraction=0.08, max_iterations=40,
                       tolerance=1e-5, phosphor_to_sample=None, device="cpu",
                       gpu_device_id=0, registration=None, initial_homography=None):
    plan = prepared or prepare_homography(reference,
                                          margin_fraction=margin_fraction,
                                          device=device, gpu_device_id=gpu_device_id)
    if plan.device != device:
        raise ValueError("prepared homography plan uses a different device")
    g = np.asarray(orientation, dtype=float)
    if g.shape == (3,):
        g = euler_to_matrix(*g)
    alpha = np.pi/2-sample_tilt+camera_elevation
    qps = (_rotation_phosphor_to_sample(alpha) if phosphor_to_sample is None
           else np.asarray(phosphor_to_sample, dtype=float))
    qpc = g @ qps
    size = np.asarray(reference).shape[0]
    initial = np.linalg.inv(_ray_matrix(scan_pc, size, qpc)) @ _ray_matrix(
        reference_pc, size, qpc)
    registration = (register_homography(
        plan, scan, initial if initial_homography is None else initial_homography,
        max_iterations=max_iterations, tolerance=tolerance)
        if registration is None else registration)
    h, rms, iterations, converged = registration
    f, strain, stress, g = deformation_from_homography(
        h, reference_pc, scan_pc, g, sample_tilt, camera_elevation,
        material, size, phosphor_to_sample)
    return HomographyResult(f, strain, stress, g, h, rms, iterations, converged)
