import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def graph_points(graph: nx.Graph, gap: float = 0.01) -> np.ndarray:
    samples = []
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        endpoints = [node for node, degree in subgraph.degree() if degree == 1]
        if len(endpoints) != 2:
            continue
        path = nx.shortest_path(subgraph, endpoints[0], endpoints[1])
        points = np.asarray([subgraph.nodes[node]["pos"] for node in path], dtype=np.float64)
        if len(points) < 2:
            continue
        lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if lengths[-1] == 0:
            continue
        positions = np.r_[0.0, np.arange(gap, lengths[-1], gap), lengths[-1]]
        samples.append(np.column_stack([np.interp(positions, lengths, points[:, axis]) for axis in range(3)]))
    return np.vstack(samples) if samples else np.empty((0, 3), dtype=np.float64)


def metrics(gt: np.ndarray, prediction: np.ndarray, tolerance: float = 0.005) -> dict[str, float]:
    if not len(gt) or not len(prediction):
        return {key: 0.0 for key in ("iou", "precision", "recall", "f1", "chamfer_mm", "hausdorff_mm")}
    gt_to_pred = cKDTree(prediction).query(gt)[0]
    pred_to_gt = cKDTree(gt).query(prediction)[0]
    true_positive = float(np.count_nonzero(gt_to_pred <= tolerance))
    false_negative = float(len(gt) - true_positive)
    false_positive = float(np.count_nonzero(pred_to_gt > tolerance))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "iou": true_positive / (true_positive + false_negative + false_positive),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "chamfer_mm": 500.0 * (gt_to_pred.mean() + pred_to_gt.mean()),
        "hausdorff_mm": 1000.0 * max(gt_to_pred.max(), pred_to_gt.max()),
    }
