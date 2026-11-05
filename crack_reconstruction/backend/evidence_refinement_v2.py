"""Second-generation refinement: small-cluster recovery and supported ridges.

All geometry is inferred from saved probabilities and predictions. No scene
identifier, annotation, or metric is accepted by these functions. v1 remains
unchanged for exact reproduction of its frozen experiment.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from .evidence_refinement import (
    EvidenceConfig, ProbabilitySupport, add_novel_supported_segments,
    link_supported_endpoints, seed_components,
)


def ordered_paths(graph):
    """Return open centerline components, failing loudly on unsupported topology."""
    paths = []
    for ids in nx.connected_components(graph):
        sub = graph.subgraph(ids)
        if sub.number_of_edges() == 0:
            continue
        ends = [n for n, d in sub.degree() if d == 1]
        if len(ends) != 2 or max(dict(sub.degree()).values()) > 2:
            raise ValueError("Expected open, nonbranching OBJ centerlines.")
        paths.append(nx.shortest_path(sub, ends[0], ends[1]))
    return paths


def append_path(result, points):
    if len(points) < 2:
        return
    # Remove consecutive duplicates while preserving the polyline geometry.
    points = np.asarray(points, float)
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10]]
    if len(points) < 2:
        return
    offset = len(result)
    for idx, point in enumerate(points):
        result.add_node(offset + idx, pos=point)
        if idx:
            result.add_edge(offset + idx - 1, offset + idx)


def small_cluster_centerline(xyz, probabilities, *, linearity=.80, bin_width=.0025):
    """Weighted PCA/bin centroids for elongated clusters rejected by LBC.

    This fallback uses observed cross-section centroids and stays within the
    observed longitudinal extent. Nearly isotropic noise is rejected.
    """
    xyz, probabilities = np.asarray(xyz, float), np.asarray(probabilities, float)
    graph = nx.Graph()
    if len(xyz) < 5:
        return graph, "too_few_points"
    weights = np.maximum(probabilities, 1e-6)
    centre = np.average(xyz, axis=0, weights=weights)
    offset = xyz - centre
    cov = (offset * weights[:, None]).T @ offset / weights.sum()
    values, vectors = np.linalg.eigh(cov)
    if values[-1] <= 1e-14 or values[-1] / max(values.sum(), 1e-14) < linearity:
        return graph, "nonlinear_or_isotropic"
    axis = vectors[:, -1]
    # Fix the SVD/eigenvector sign to keep identical inputs byte reproducible.
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    longitudinal = offset @ axis
    if np.ptp(longitudinal) < .005:
        return graph, "too_short"
    bins = np.floor((longitudinal - longitudinal.min()) / bin_width).astype(int)
    points = [np.average(xyz[bins == b], axis=0, weights=weights[bins == b]) for b in sorted(np.unique(bins))]
    append_path(graph, points)
    return graph, "pca_fallback"


def recovery_candidates(support, *, cached_large=None, cached_failures=None):
    """Reuse successful v1 contractions or extract new ones; recover failures."""
    import open3d as o3d
    from .contraction import contract_subcloud

    groups = seed_components(support)
    audit = {"seeded_clusters": len(groups), "lbc_success": 0, "small_fallback_success": 0,
             "rejected_clusters": [], "lbc_errors": []}
    result = nx.Graph() if cached_large is None else cached_large.copy()
    failing_indices = None if cached_failures is None else {int(v["cluster"]) for v in cached_failures}
    for index, ids in enumerate(groups):
        if failing_indices is not None and index not in failing_indices:
            audit["lbc_success"] += 1
            continue
        graph = nx.Graph()
        if cached_large is None:
            try:
                cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(support.xyz[ids]))
                graph = contract_subcloud(cloud, init_contraction=10.)
                audit["lbc_success"] += 1
            except Exception as exc:
                audit["lbc_errors"].append({"cluster": index, "points": len(ids), "error": repr(exc)})
        if graph.number_of_edges() == 0:
            graph, reason = small_cluster_centerline(support.xyz[ids], support.probabilities[ids])
            if graph.number_of_edges():
                audit["small_fallback_success"] += 1
            else:
                audit["rejected_clusters"].append({"cluster": index, "points": len(ids), "reason": reason})
        if graph.number_of_edges():
            result = nx.disjoint_union(result, graph)
    return result, audit


def trim_unsupported_runs(graph, support, max_gap=.010):
    """Cut cumulative unsupported runs, including runs across many short edges.

    Query midpoints of subdivided original edges. Only subdivisions belonging
    to a long unsupported run are removed; supported parts are preserved.
    """
    if max_gap <= 0:
        return graph.copy()
    output = nx.Graph()
    for nodes in ordered_paths(graph):
        original = np.array([graph.nodes[n]["pos"] for n in nodes])
        pieces = [original[0]]
        for start, end in zip(original[:-1], original[1:]):
            count = max(1, int(np.ceil(np.linalg.norm(end-start) / support.config.sample_spacing)))
            pieces.extend(np.linspace(start, end, count+1)[1:])
        points = np.asarray(pieces)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        evidence = support.mask((points[:-1] + points[1:]) / 2)
        keep = np.ones(len(lengths), bool)
        start = 0
        while start < len(evidence):
            if evidence[start]:
                start += 1
                continue
            end = start + 1
            while end < len(evidence) and not evidence[end]:
                end += 1
            if lengths[start:end].sum() > max_gap:
                keep[start:end] = False
            start = end
        if keep.all():
            # No cut: preserve original vertices and confidence ranking exactly.
            append_path(output, original)
            continue
        start = 0
        while start < len(keep):
            if not keep[start]:
                start += 1
                continue
            end = start + 1
            while end < len(keep) and keep[end]:
                end += 1
            if lengths[start:end].sum() >= .005:
                append_path(output, points[start:end+1])
            start = end
    return output


def ridge_project(graph, support, radius=.006, iterations=2, max_shift=.003):
    """Move each vertex toward its probability-weighted local cross section.

    Tangential displacement is removed so endpoints are not shortened by
    averaging. The total displacement from each original vertex is bounded.
    Geometry without enough probability support is kept as-is.
    """
    result = graph.copy()
    if radius <= 0 or support.tree is None:
        return result
    for nodes in ordered_paths(graph):
        original = np.array([graph.nodes[n]["pos"] for n in nodes], float)
        positions = original.copy()
        for _ in range(iterations):
            neighbours = support.tree.query_ball_point(positions, radius)
            updated = positions.copy()
            for idx, ids in enumerate(neighbours):
                if len(ids) < 4:
                    continue
                left, right = max(0, idx-2), min(len(nodes)-1, idx+2)
                tangent = positions[right] - positions[left]
                norm = np.linalg.norm(tangent)
                if norm <= 1e-10:
                    continue
                tangent /= norm
                q = support.xyz[ids]
                d = np.linalg.norm(q - positions[idx], axis=1)
                weights = support.probabilities[ids] ** 2 * np.exp(-.5 * (d / (radius*.5))**2)
                centre = np.average(q, axis=0, weights=np.maximum(weights,1e-12))
                shift = centre - positions[idx]
                shift -= tangent * np.dot(shift,tangent)
                candidate = positions[idx] + shift
                displacement = candidate - original[idx]
                scale = min(1., max_shift / max(np.linalg.norm(displacement),1e-12))
                updated[idx] = original[idx] + displacement * scale
            positions = updated
        for node, point in zip(nodes,positions):
            result.nodes[node]["pos"] = point
    return result


def refine_v2(original, support, candidates, *, mode="augment", trim_gap=.01, ridge_radius=.006):
    if mode not in ("augment", "rebuild", "refine"):
        raise ValueError(f"Unknown mode: {mode}")
    graph = candidates.copy() if mode == "rebuild" else original.copy()
    graph = trim_unsupported_runs(graph, support, trim_gap)
    if mode == "augment":
        graph, _ = add_novel_supported_segments(graph, candidates, support)
    graph = ridge_project(graph, support, radius=ridge_radius)
    graph, audit = link_supported_endpoints(graph, support)
    ordered_paths(graph)  # Never silently drop a branched/cyclic prediction.
    return graph, audit
