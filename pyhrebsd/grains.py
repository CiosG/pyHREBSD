"""Segment an indexed EBSD map by neighboring crystal misorientation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np


@dataclass(frozen=True)
class GrainMap:
    labels: np.ndarray  # map-shaped; zero is unindexed
    sizes: np.ndarray  # index by grain ID; element zero is unindexed count
    phase_ids: np.ndarray  # index by grain ID

    @property
    def count(self) -> int:
        return len(self.sizes) - 1


def cubic_symmetry() -> np.ndarray:
    """The 24 proper rotations of a cubic crystal."""
    rotations = []
    for axes in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            matrix[np.arange(3), axes] = signs
            if round(np.linalg.det(matrix)) == 1:
                rotations.append(matrix)
    return np.stack(rotations)


def _orientation_matrices(eulers: np.ndarray) -> np.ndarray:
    """Vectorized OpenXY Bunge sample-to-crystal matrices."""
    p1, p, p2 = eulers.T
    c1, s1 = np.cos(p1), np.sin(p1)
    c, s = np.cos(p), np.sin(p)
    c2, s2 = np.cos(p2), np.sin(p2)
    g = np.empty((len(eulers), 3, 3), dtype=np.float64)
    g[:, 0, 0] = c1 * c2 - s1 * s2 * c
    g[:, 0, 1] = s1 * c2 + c1 * s2 * c
    g[:, 0, 2] = s2 * s
    g[:, 1, 0] = -c1 * s2 - s1 * c2 * c
    g[:, 1, 1] = -s1 * s2 + c1 * c2 * c
    g[:, 1, 2] = c2 * s
    g[:, 2, 0] = s1 * s
    g[:, 2, 1] = -c1 * s
    g[:, 2, 2] = c
    return g


def _neighbor_pairs(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(height * width).reshape(height, width)
    a = np.concatenate((grid[:, :-1].ravel(), grid[:-1, :].ravel()))
    b = np.concatenate((grid[:, 1:].ravel(), grid[1:, :].ravel()))
    return a, b


def _merge_small(labels: np.ndarray, phase: np.ndarray, minimum: int,
                 a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Assign small regions to the adjacent same-phase region sharing most edges."""
    if minimum <= 1:
        return labels
    sizes = np.bincount(labels)
    small = np.flatnonzero((sizes > 0) & (sizes < minimum))
    for source in small:
        if source == 0 or not np.any(labels == source):
            continue
        touching = np.concatenate((labels[b[(labels[a] == source) & (labels[b] != source)]],
                                   labels[a[(labels[b] == source) & (labels[a] != source)]]))
        touching = touching[(touching != 0) & (phase[touching] == phase[source])]
        if len(touching):
            candidates, counts = np.unique(touching, return_counts=True)
            target = int(candidates[np.argmax(counts)])
            labels[labels == source] = target
        else:
            labels[labels == source] = 0
    return labels


def segment_grains(eulers: np.ndarray, phases: np.ndarray, shape: tuple[int, int],
                   threshold_degrees: float = 5.0, min_size: int = 5,
                   symmetry: str = "cubic") -> GrainMap:
    """Group 4-connected, same-phase points below a misorientation threshold.

    Euler angles are radians. Cubic symmetry is suitable for silicon; use
    ``symmetry='none'`` for a crystal with unknown symmetry. Small regions
    merge into a touching region of the same phase, or become unindexed.
    """
    height, width = map(int, shape)
    angles = np.asarray(eulers, dtype=np.float64)
    phase = np.asarray(phases, dtype=np.int32).reshape(-1)
    n = height * width
    if height < 1 or width < 1 or angles.shape != (n, 3) or phase.shape != (n,):
        raise ValueError("shape, Euler angles, and phases must describe the same map")
    if not (0 < threshold_degrees < 180) or min_size < 1:
        raise ValueError("threshold_degrees must be in (0, 180) and min_size >= 1")
    if symmetry == "cubic":
        sym = cubic_symmetry()
    elif symmetry == "none":
        sym = np.eye(3)[None]
    else:
        raise ValueError("symmetry must be 'cubic' or 'none'")

    valid = (phase > 0) & np.all(np.isfinite(angles), axis=1)
    matrices = _orientation_matrices(np.where(np.isfinite(angles), angles, 0))
    a, b = _neighbor_pairs(height, width)
    eligible = valid[a] & valid[b] & (phase[a] == phase[b])
    edge_a, edge_b = a[eligible], b[eligible]
    linked = np.zeros(len(edge_a), dtype=bool)
    cos_limit = np.cos(np.deg2rad(threshold_degrees))
    for start in range(0, len(edge_a), 4096):
        sl = slice(start, start + 4096)
        relative = matrices[edge_a[sl]] @ np.swapaxes(matrices[edge_b[sl]], 1, 2)
        traces = np.einsum("sij,nji->ns", sym, relative, optimize=True)
        linked[sl] = np.max((traces - 1) / 2, axis=1) >= cos_limit - 1e-12

    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for i, j in zip(edge_a[linked], edge_b[linked]):
        ri, rj = root(int(i)), root(int(j))
        if ri != rj:
            if size[ri] < size[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            size[ri] += size[rj]

    labels = np.zeros(n, dtype=np.int32)
    members = np.flatnonzero(valid)
    roots = np.array([root(int(i)) for i in members], dtype=np.int32)
    _, inverse = np.unique(roots, return_inverse=True)
    labels[members] = inverse + 1
    grain_phase = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    grain_phase[labels[members]] = phase[members]
    labels = _merge_small(labels, grain_phase, min_size, a, b)

    surviving = np.unique(labels)
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    positive = surviving[surviving > 0]
    remap[positive] = np.arange(1, len(positive) + 1, dtype=np.int32)
    labels = remap[labels]
    sizes = np.bincount(labels, minlength=int(labels.max()) + 1)
    phase_ids = np.zeros(len(sizes), dtype=np.int32)
    phase_ids[labels[valid & (labels > 0)]] = phase[valid & (labels > 0)]
    return GrainMap(labels.reshape(shape), sizes, phase_ids)
