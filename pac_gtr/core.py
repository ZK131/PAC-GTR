"""Probability-aware clustering and geometry-gated topology recovery."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CrackEnsemblesProtocol:
    """Frozen CrackEnsembles protocol in robust-diagonal ratios."""

    threshold: float = 0.60
    eps_ratio: float = 0.01
    topology_gap_ratio: float = 0.02
    min_samples: int = 4
    min_component_points: int = 4
    adaptive_power: float = 0.05
    merge_alignment_cos: float = 0.866
    line_points: int = 32


@dataclass(frozen=True)
class CrackStructuresProtocol:
    """Frozen CrackStructures v2.1 protocol in metric (meter) units."""

    threshold: float = 0.25
    eps_m: float = 0.035
    max_gap_m: float = 0.04
    min_samples_cluster: int = 1
    min_points: int = 5
    adaptive_power: float = 0.05
    min_alignment_cos: float = 0.866
    init_contraction: float = 10.0
    line_points: int = 32


@dataclass
class Component:
    indices: np.ndarray
    line: np.ndarray
    direction: np.ndarray
    endpoints: np.ndarray
    confidence: float
    canonical_id: int


def robust_diagonal(points: np.ndarray) -> float:
    """Robust 3D scale based on the 0.5% and 99.5% coordinate quantiles."""
    points = _points(points)
    lo = np.quantile(points, 0.005, axis=0)
    hi = np.quantile(points, 0.995, axis=0)
    diagonal = float(np.linalg.norm(hi - lo))
    if diagonal <= 0:
        raise ValueError("points must span a non-zero 3D extent")
    return diagonal


def _points(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) == 0:
        raise ValueError("points must have shape (N, 3) with N > 0")
    if not np.isfinite(value).all():
        raise ValueError("points must be finite")
    return value


def _probabilities(value: np.ndarray, n: int) -> np.ndarray:
    value = np.asarray(value, dtype=float).reshape(-1)
    if len(value) != n or not np.isfinite(value).all():
        raise ValueError("probabilities must be finite and match the point count")
    if ((value < 0) | (value > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return value


def _direction(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(3, dtype=float)
    axis = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)[2][0]
    pivot = int(np.argmax(np.abs(axis)))
    return axis if axis[pivot] >= 0 else -axis


def graph_dbscan(
    points: np.ndarray,
    probabilities: np.ndarray,
    eps: float,
    min_samples: int,
    *,
    adaptive: bool,
    weighted_core: bool,
    power: float,
) -> np.ndarray:
    """Deterministic DBSCAN analogue used by PAC.

    In adaptive mode an edge survives when d_ij/(p_i p_j)^power <= eps.
    Weighted-core mode uses normalized probability mass instead of a raw
    neighbor count for the core-point test.
    """
    points = _points(points)
    probabilities = _probabilities(probabilities, len(points))
    if eps <= 0 or min_samples < 1 or power < 0:
        raise ValueError("invalid clustering parameters")

    pairs = cKDTree(points).query_pairs(eps, output_type="ndarray")
    if len(pairs) and adaptive:
        distance = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
        factor = np.clip(
            (probabilities[pairs[:, 0]] * probabilities[pairs[:, 1]]) ** power,
            0.01,
            1.0,
        )
        pairs = pairs[(distance / factor) <= eps]

    neighbours: list[list[int]] = [[] for _ in points]
    for a, b in pairs:
        neighbours[int(a)].append(int(b))
        neighbours[int(b)].append(int(a))
    for values in neighbours:
        values.sort()

    if weighted_core:
        weights = np.clip(probabilities, 0.01, 1.0)
        weights /= max(float(weights.mean()), 1e-12)
        core = np.asarray(
            [weights[i] + weights[adjacent].sum() >= min_samples for i, adjacent in enumerate(neighbours)]
        )
    else:
        core = np.asarray([1 + len(adjacent) >= min_samples for adjacent in neighbours])

    core_ids = np.flatnonzero(core)
    labels = np.full(len(points), -1, dtype=int)
    if not len(core_ids):
        return labels

    core_map = np.full(len(points), -1, dtype=int)
    core_map[core_ids] = np.arange(len(core_ids))
    rows: list[int] = []
    cols: list[int] = []
    for i in core_ids:
        for j in neighbours[i]:
            if core[j]:
                rows.append(int(core_map[i]))
                cols.append(int(core_map[j]))
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(len(core_ids), len(core_ids)),
    ).tocsr()
    _, core_labels = connected_components(graph.maximum(graph.T), directed=False)
    labels[core_ids] = core_labels
    for i in np.flatnonzero(~core):
        adjacent_labels = sorted({labels[j] for j in neighbours[i] if labels[j] >= 0})
        if adjacent_labels:
            labels[i] = adjacent_labels[0]
    return labels


def _component(points: np.ndarray, probabilities: np.ndarray, indices: np.ndarray, line_points: int) -> Component:
    xyz = points[indices]
    centre = xyz.mean(axis=0)
    axis = _direction(xyz)
    coordinate = (xyz - centre) @ axis
    low, high = np.quantile(coordinate, [0.02, 0.98])
    line = centre + np.linspace(low, high, line_points)[:, None] * axis[None, :]
    sorted_probabilities = np.sort(probabilities[indices])
    top = max(1, int(math.ceil(0.2 * len(sorted_probabilities))))
    return Component(
        indices=indices,
        line=line.astype(np.float32),
        direction=axis,
        endpoints=np.asarray([line[0], line[-1]], dtype=np.float32),
        confidence=float(sorted_probabilities[-top:].mean()),
        canonical_id=int(indices.min()),
    )


def _merge_groups(components: list[Component], gap: float, alignment_cos: float) -> list[list[np.ndarray]]:
    parent = np.arange(len(components))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    candidates: list[tuple[float, int, int]] = []
    for i, left in enumerate(components):
        for j in range(i + 1, len(components)):
            right = components[j]
            if abs(float(np.dot(left.direction, right.direction))) < alignment_cos:
                continue
            distance = float(cKDTree(right.endpoints).query(left.endpoints, k=1)[0].min())
            if distance <= gap:
                candidates.append((distance, i, j))
    for _, i, j in sorted(candidates):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    groups: dict[int, list[np.ndarray]] = {}
    for index, component in enumerate(components):
        groups.setdefault(find(index), []).append(component.indices)
    return [groups[key] for key in sorted(groups)]


def predict_components(
    points: np.ndarray,
    probabilities: np.ndarray,
    *,
    protocol: CrackEnsemblesProtocol | CrackStructuresProtocol,
    mode: str = "full",
) -> tuple[list[Component], np.ndarray]:
    """Extract centerline components with a declared ablation mode.

    Modes: baseline, pac, gate, gtr, and full. ``full`` enables both PAC's
    adaptive edge test and weighted core test, followed by GTR component merge.
    """
    if mode not in {"baseline", "pac", "gate", "gtr", "full"}:
        raise ValueError(f"unknown mode: {mode}")
    points = _points(points)
    probabilities = _probabilities(probabilities, len(points))
    diagonal = robust_diagonal(points)
    if isinstance(protocol, CrackStructuresProtocol):
        eps = protocol.eps_m
        topology_gap = protocol.max_gap_m
        min_samples = protocol.min_samples_cluster
        min_component_points = protocol.min_points
        merge_alignment_cos = protocol.min_alignment_cos
    else:
        eps = protocol.eps_ratio * diagonal
        topology_gap = protocol.topology_gap_ratio * diagonal
        min_samples = protocol.min_samples
        min_component_points = protocol.min_component_points
        merge_alignment_cos = protocol.merge_alignment_cos
    candidate = np.flatnonzero(probabilities >= protocol.threshold)
    if not len(candidate):
        return [], np.zeros(len(points), dtype=bool)

    adaptive = mode in {"pac", "full"}
    weighted_core = mode in {"gate", "gtr", "full"}
    labels = graph_dbscan(
        points[candidate],
        probabilities[candidate],
        eps,
        min_samples,
        adaptive=adaptive,
        weighted_core=weighted_core,
        power=protocol.adaptive_power,
    )
    components = []
    for label in np.unique(labels[labels >= 0]):
        local = np.flatnonzero(labels == label)
        if len(local) >= min_component_points:
            components.append(_component(points, probabilities, candidate[local], protocol.line_points))

    if mode in {"gtr", "full"} and len(components) > 1:
        groups = _merge_groups(
            components,
            topology_gap,
            merge_alignment_cos,
        )
        components = [
            _component(points, probabilities, np.unique(np.concatenate(parts)), protocol.line_points)
            for parts in groups
        ]

    components.sort(key=lambda item: (-item.confidence, -len(item.indices), item.canonical_id))
    mask = np.zeros(len(points), dtype=bool)
    for component in components:
        mask[component.indices] = True
    return components, mask
