"""CuPy implementation of the complete global homography DIC registration.

The pattern, interpolation, residuals, normal equations and warp updates stay
on one GPU. Only the final 3x3 homography and quality scalars return to CPU.
This module is imported only when the GPU backend is selected.
"""

from __future__ import annotations

import numpy as np

from .homography import HomographyPlan, _normalize_h


def _cupy():
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter, map_coordinates
    except ImportError as exc:
        raise RuntimeError(
            "GPU homography requires CuPy with CUDA support; use device='cpu' "
            "or run with a CUDA-enabled Python environment") from exc
    return cp, gaussian_filter, map_coordinates


def _preprocess_gpu(image, sigma, cp, gaussian_filter):
    values = cp.asarray(image, dtype=cp.float64)
    return gaussian_filter(values, 1.0) - gaussian_filter(values, sigma)


def prepare_gpu_homography(reference, *, margin_fraction=0.08,
                           background_sigma=None, gpu_device_id=0):
    """Build a cached float64 reference/Jacobian on the selected CUDA device."""
    image = np.asarray(reference)
    if image.ndim != 2 or image.shape[0] != image.shape[1] or image.shape[0] < 32:
        raise ValueError("homography requires a square pattern of at least 32 pixels")
    if not 0 <= margin_fraction < 0.4:
        raise ValueError("invalid homography margin")
    cp, gaussian_filter, _ = _cupy()
    cp.cuda.Device(gpu_device_id).use()
    size = image.shape[0]
    sigma = background_sigma or max(5.0, size/25.0)
    ref = _preprocess_gpu(image, sigma, cp, gaussian_filter)
    center, scale = (size-1)/2, size/2
    edge = max(3, int(size*margin_fraction))
    y, x = np.mgrid[edge:size-edge, edge:size-edge]
    xp, yp = cp.asarray(x.ravel(), dtype=cp.float64), cp.asarray(y.ravel(), dtype=cp.float64)
    xn, yn = (xp-center)/scale, (yp-center)/scale
    values = ref[cp.asarray(y.ravel()), cp.asarray(x.ravel())]
    std = values.std()
    if float(std.get()) <= 1e-10:
        raise ValueError("reference pattern has no usable contrast")
    template = (values-values.mean())/std
    gy, gx = cp.gradient(ref)
    gx = gx[cp.asarray(y.ravel()), cp.asarray(x.ravel())]*scale/std
    gy = gy[cp.asarray(y.ravel()), cp.asarray(x.ravel())]*scale/std
    zeros, ones = cp.zeros_like(xn), cp.ones_like(xn)
    jx = cp.column_stack((xn, yn, ones, zeros, zeros, zeros, -xn*xn, -xn*yn))
    jy = cp.column_stack((zeros, zeros, zeros, xn, yn, ones, -xn*yn, -yn*yn))
    steepest = gx[:, None]*jx + gy[:, None]*jy
    return HomographyPlan(ref, cp.column_stack((xn, yn)), template, steepest,
                          scale, center, "gpu", gpu_device_id)


def _warp_gpu(h, xy):
    x, y = xy[:, 0], xy[:, 1]
    d = h[2, 0]*x+h[2, 1]*y+h[2, 2]
    return ((h[0, 0]*x+h[0, 1]*y+h[0, 2])/d,
            (h[1, 0]*x+h[1, 1]*y+h[1, 2])/d)


def _delta_matrix(delta, cp):
    one = cp.asarray(1.0)
    zero = cp.asarray(0.0)
    return cp.stack((
        cp.stack((one+delta[0], delta[1], delta[2])),
        cp.stack((delta[3], one+delta[4], delta[5])),
        cp.stack((delta[6], delta[7], one)),
    ))


def _sample_normalized(target, h, plan, cp, map_coordinates):
    u, v = _warp_gpu(h, plan.xy)
    px, py = u*plan.scale+plan.center, v*plan.scale+plan.center
    valid = ((px >= 2) & (py >= 2) & (px < target.shape[1]-3) &
             (py < target.shape[0]-3))
    count = valid.sum()
    warped = map_coordinates(target, cp.stack((py, px)), order=1,
                             mode="nearest", prefilter=False)
    mean = cp.where(valid, warped, 0).sum()/count
    sd = cp.sqrt(cp.where(valid, (warped-mean)**2, 0).sum()/count)
    residual = cp.where(valid, (warped-mean)/sd-plan.template, 0)
    return valid, count, sd, residual


def register_gpu_homography(plan, scan, initial=None, *, max_iterations=40,
                            tolerance=1e-5, background_sigma=None, stream=None):
    """Refine the eight projective parameters on GPU using float64 IC-GN."""
    cp, gaussian_filter, map_coordinates = _cupy()
    image = scan if isinstance(scan, cp.ndarray) else np.asarray(scan)
    if image.shape != plan.reference.shape:
        raise ValueError("scan and reference patterns must have equal shape")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("invalid homography iteration settings")
    cp.cuda.Device(plan.gpu_device_id).use()
    if stream is not None:
        stream.use()
    sigma = background_sigma or max(5.0, image.shape[0]/25.0)
    target = _preprocess_gpu(image, sigma, cp, gaussian_filter)
    s_cpu = np.array([[1/plan.scale, 0, -plan.center/plan.scale],
                      [0, 1/plan.scale, -plan.center/plan.scale], [0, 0, 1]])
    h_cpu = np.eye(3) if initial is None else _normalize_h(
        s_cpu @ initial @ np.linalg.inv(s_cpu))
    h = cp.asarray(h_cpu)
    corners = cp.asarray([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    converged = False
    for iteration in range(1, max_iterations+1):
        valid, count, sd, residual = _sample_normalized(
            target, h, plan, cp, map_coordinates)
        if int(count.get()) < max(100, plan.xy.shape[0]//2):
            raise ValueError("homography moves too much of the pattern outside the detector")
        if float(sd.get()) <= 1e-10:
            raise ValueError("scan pattern has no usable contrast")
        jac = plan.steepest
        gram = jac.T @ (jac*valid[:, None])
        rhs = jac.T @ residual
        delta = cp.linalg.solve(gram, rhs)
        if not bool(cp.all(cp.isfinite(delta)).get()):
            raise ValueError("pattern cannot determine eight homography parameters")
        for _ in range(12):
            dh = _delta_matrix(delta, cp)
            u, v = _warp_gpu(dh, corners)
            movement = float(cp.max(cp.sqrt(((u-corners[:, 0])*plan.scale)**2 +
                                            ((v-corners[:, 1])*plan.scale)**2)).get())
            if movement <= 3:
                break
            delta *= 0.5
        h = (h @ cp.linalg.inv(dh))
        h /= h[2, 2]
        if movement < tolerance:
            converged = True
            break
    valid, count, sd, residual = _sample_normalized(target, h, plan, cp,
                                                     map_coordinates)
    rms = float(cp.sqrt(cp.sum(residual**2)/count).get())
    h_pixel = cp.asarray(np.linalg.inv(s_cpu)) @ h @ cp.asarray(s_cpu)
    return _normalize_h(cp.asnumpy(h_pixel)), rms, iteration, converged


def register_gpu_homography_batch(plan, scans, initials=None, *, max_iterations=40,
                                   tolerance=1e-5, background_sigma=None,
                                   gpu_device_id=None, max_workers=4):
    """Register a chunk of patterns concurrently on independent CUDA streams."""
    from concurrent.futures import ThreadPoolExecutor
    cp, _, _ = _cupy()
    device_id = plan.gpu_device_id if gpu_device_id is None else int(gpu_device_id)
    values = np.asarray(scans)
    if values.ndim != 3 or values.shape[0] == 0:
        if values.ndim != 3:
            raise ValueError("scans must have shape (batch, height, width)")
        return []
    if values.shape[1:] != plan.reference.shape:
        raise ValueError("scan patterns do not match the homography plan")
    if initials is None:
        initial_list = [None] * values.shape[0]
    else:
        initial_list = list(initials)
        if len(initial_list) != values.shape[0]:
            raise ValueError("initials must contain one homography per scan")
    cp.cuda.Device(device_id).use()
    scans_gpu = cp.asarray(values, dtype=cp.float64)
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(min(max_workers, values.shape[0]))]
    def one(index):
        stream = streams[index % len(streams)]
        return register_gpu_homography(
            plan, scans_gpu[index], initial_list[index], max_iterations=max_iterations,
            tolerance=tolerance, background_sigma=background_sigma, stream=stream)
    with ThreadPoolExecutor(max_workers=len(streams)) as executor:
        futures = [executor.submit(one, index) for index in range(values.shape[0])]
        return [future.result() for future in futures]
