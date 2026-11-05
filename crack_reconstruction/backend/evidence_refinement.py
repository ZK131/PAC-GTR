"""Probability-supported refinement of frozen PAC–GTR crack centerlines.

Inference reads only prediction geometry and the corresponding probability
cloud, never annotations, scene names, or an evaluation metric. All distances
are in metres. The original extraction pipeline is left unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class EvidenceConfig:
    high_threshold: float = 0.25
    low_threshold: float = 0.15
    cluster_radius: float = 0.035
    probability_power: float = 0.05
    min_cluster_points: int = 5
    support_radius: float = 0.005
    sample_spacing: float = 0.0025
    max_unsupported_run: float = 0.010
    novel_radius: float = 0.005
    min_novel_length: float = 0.010
    link_gap: float = 0.040
    link_alignment: float = 0.866
    link_support_fraction: float = 0.80
    tangent_length: float = 0.010

    def __post_init__(self):
        if not 0 <= self.low_threshold <= self.high_threshold <= 1:
            raise ValueError("Thresholds must satisfy 0 <= low <= high <= 1.")
        for name in ("cluster_radius", "support_radius", "sample_spacing", "novel_radius", "tangent_length"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")

    def to_dict(self):
        return asdict(self)


def sample_edge(start, end, spacing):
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(end - start))
    count = max(2, int(np.ceil(length / spacing)) + 1)
    return np.linspace(start, end, count), length


def dense_points(graph, spacing=0.0025):
    pieces = [sample_edge(graph.nodes[a]["pos"], graph.nodes[b]["pos"], spacing)[0]
              for a, b in graph.edges]
    return np.vstack(pieces) if pieces else np.empty((0, 3))


def graph_length(graph):
    return sum(float(np.linalg.norm(np.asarray(graph.nodes[a]["pos"]) - graph.nodes[b]["pos"]))
               for a, b in graph.edges)


def edge_subgraph(graph, edges):
    result = nx.Graph()
    for a, b in edges:
        result.add_node(a, **graph.nodes[a])
        result.add_node(b, **graph.nodes[b])
        result.add_edge(a, b, **graph.edges[a, b])
    return result


def longest_missing_run(mask, sample_step):
    longest = run = 0
    for supported in mask:
        run = 0 if supported else run + 1
        longest = max(longest, run)
    return longest * sample_step


class ProbabilitySupport:
    def __init__(self, xyz, probabilities, config):
        self.config = config
        xyz = np.asarray(xyz, float)
        probabilities = np.asarray(probabilities, float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or probabilities.shape != (len(xyz),):
            raise ValueError("Expected xyz [N,3] and probabilities [N].")
        mask = np.isfinite(xyz).all(axis=1) & np.isfinite(probabilities)
        mask &= probabilities > config.low_threshold
        self.xyz = xyz[mask]
        self.probabilities = probabilities[mask]
        self.tree = cKDTree(self.xyz) if len(self.xyz) else None

    def mask(self, points):
        if self.tree is None:
            return np.zeros(len(points), dtype=bool)
        return self.tree.query(points, workers=1)[0] <= self.config.support_radius

    def edge_evidence(self, start, end):
        samples, length = sample_edge(start, end, self.config.sample_spacing)
        supported = self.mask(samples)
        missing_run = longest_missing_run(supported, length / (len(samples) - 1))
        return float(np.mean(supported)), missing_run


def seed_components(support):
    """Sparse adaptive-distance components seeded by high-confidence points.

    For min_samples=1 this is the connectivity part of adaptive DBSCAN. It
    avoids the original dense distance matrix and centroid-based merging.
    """
    cfg = support.config
    xyz, probability = support.xyz, support.probabilities
    if not len(xyz):
        return []
    pairs = support.tree.query_pairs(cfg.cluster_radius, output_type="ndarray")
    if len(pairs):
        distance = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
        scaling = np.maximum((probability[pairs[:, 0]] * probability[pairs[:, 1]]) ** cfg.probability_power, 0.01)
        pairs = pairs[distance <= cfg.cluster_radius * scaling]
    row = pairs[:, 0] if len(pairs) else np.empty(0, int)
    col = pairs[:, 1] if len(pairs) else np.empty(0, int)
    adjacency = coo_matrix((np.ones(len(row)), (row, col)), shape=(len(xyz), len(xyz))).tocsr()
    count, labels = connected_components(adjacency, directed=False)
    groups = []
    for label in range(count):
        ids = np.flatnonzero(labels == label)
        if len(ids) >= cfg.min_cluster_points and np.any(probability[ids] > cfg.high_threshold):
            groups.append(ids)
    return groups


def extract_recovery_candidates(support, mst_backend="networkx"):
    """Contract only low-threshold components with a high-confidence seed."""
    import open3d as o3d
    from .contraction import contract_subcloud

    result = nx.Graph()
    groups = seed_components(support)
    failures = []
    for index, ids in enumerate(groups):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(support.xyz[ids]))
        try:
            graph = contract_subcloud(cloud, init_contraction=10.0,
                                      mst_backend=mst_backend)
        except Exception as exc:
            failures.append({"cluster": index, "points": len(ids), "error": repr(exc)})
            warnings.warn(f"Recovery contraction failed: {exc}")
            continue
        graph = nx.convert_node_labels_to_integers(graph, first_label=len(result))
        result = nx.compose(result, graph)
    return result, {"candidate_points": len(support.xyz), "seeded_components": len(groups),
                    "contraction_failures": failures}


def prune_unsupported_edges(graph, support):
    """Remove complete skeleton edges containing long unsupported stretches."""
    kept, removed = [], []
    for a, b in graph.edges:
        fraction, missing_run = support.edge_evidence(graph.nodes[a]["pos"], graph.nodes[b]["pos"])
        if missing_run > support.config.max_unsupported_run:
            removed.append((a, b))
        else:
            kept.append((a, b))
    result = edge_subgraph(graph, kept)
    return result, {"removed_edges": len(removed), "removed_length_m": graph_length(edge_subgraph(graph, removed))}


def add_novel_supported_segments(graph, candidates, support):
    """Add candidate runs that have local evidence and are outside old coverage.

    Novelty is measured against prediction geometry, never GT. Whole original
    candidate edges are retained, without extrapolating or moving vertices.
    """
    points = dense_points(graph, support.config.sample_spacing)
    tree = cKDTree(points) if len(points) else None
    kept = []
    for a, b in candidates.edges:
        samples, _ = sample_edge(candidates.nodes[a]["pos"], candidates.nodes[b]["pos"], support.config.sample_spacing)
        novel = tree.query(samples, workers=1)[0] > support.config.novel_radius if tree else np.ones(len(samples), bool)
        if np.mean(novel & support.mask(samples)) >= 2 / 3:
            kept.append((a, b))
    residual = edge_subgraph(candidates, kept)
    selected = nx.Graph()
    for ids in nx.connected_components(residual):
        component = residual.subgraph(ids).copy()
        if graph_length(component) >= support.config.min_novel_length:
            selected = nx.compose(selected, component)
    result = nx.disjoint_union(graph, selected)
    return result, {"added_edges": selected.number_of_edges(), "added_length_m": graph_length(selected)}


def endpoint_tangent(graph, node, length):
    start = np.asarray(graph.nodes[node]["pos"], float)
    current, previous, walked = node, None, 0.0
    point = start
    while walked < length:
        next_nodes = [n for n in graph.neighbors(current) if n != previous]
        if len(next_nodes) != 1:
            break
        next_node = next_nodes[0]
        next_point = np.asarray(graph.nodes[next_node]["pos"], float)
        edge_length = float(np.linalg.norm(next_point - point))
        if walked + edge_length >= length and edge_length > 0:
            point = point + (next_point - point) * ((length - walked) / edge_length)
            break
        walked += edge_length
        point = next_point
        previous, current = current, next_node
    direction = start - point
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-12 else None


def link_supported_endpoints(graph, support):
    result = graph.copy()
    cfg = support.config
    endpoints = []
    for node, degree in graph.degree():
        if degree == 1:
            tangent = endpoint_tangent(graph, node, cfg.tangent_length)
            if tangent is not None:
                endpoints.append((node, np.asarray(graph.nodes[node]["pos"], float), tangent))
    if len(endpoints) < 2:
        return result, {"accepted_links": 0, "unsupported_links_rejected": 0}
    points = np.array([item[1] for item in endpoints])
    pairs = sorted(cKDTree(points).query_pairs(cfg.link_gap),
                   key=lambda pair: (float(np.linalg.norm(points[pair[0]] - points[pair[1]])), pair))
    components = nx.utils.UnionFind(result.nodes)
    for a, b in result.edges:
        components.union(a, b)
    used, accepted, rejected = set(), 0, 0
    for left, right in pairs:
        a, pa, ta = endpoints[left]
        b, pb, tb = endpoints[right]
        if a in used or b in used or components[a] == components[b]:
            continue
        distance = float(np.linalg.norm(pb - pa))
        if distance <= 1e-9:
            aligned = abs(float(np.dot(ta, tb))) >= cfg.link_alignment
        else:
            direction = (pb - pa) / distance
            aligned = np.dot(ta, direction) >= cfg.link_alignment and np.dot(tb, -direction) >= cfg.link_alignment
        if not aligned:
            continue
        fraction, run = support.edge_evidence(pa, pb)
        if fraction < cfg.link_support_fraction or run > cfg.max_unsupported_run:
            rejected += 1
            continue
        result.add_edge(a, b, weight=distance)
        components.union(a, b)
        used.update((a, b))
        accepted += 1
    return result, {"accepted_links": accepted, "unsupported_links_rejected": rejected}


def refine_centerlines(graph, support, *, candidates=None, prune=True, recover=True, link=True):
    result = graph.copy()
    audit = {}
    if prune:
        result, values = prune_unsupported_edges(result, support)
        audit.update(values)
    if recover:
        if candidates is None:
            candidates, extraction_audit = extract_recovery_candidates(support)
            audit.update(extraction_audit)
        result, values = add_novel_supported_segments(result, candidates, support)
        audit.update(values)
    if link:
        result, values = link_supported_endpoints(result, support)
        audit.update(values)
    audit.update({"output_components": nx.number_connected_components(result), "output_length_m": graph_length(result)})
    return result, audit
