from __future__ import annotations
import math
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from .metric_sampling import graph_points, metrics
from .metric_components import components, ranked_components


def component_iou(target: nx.Graph, prediction: nx.Graph, tolerance: float) -> float:
    """Same KD-tree point-count IoU as nested_loso_full_pipeline.py."""
    target_points = graph_points(target)
    prediction_points = graph_points(prediction)
    if not len(target_points) or not len(prediction_points):
        return 0.0
    a = cKDTree(prediction_points).query(target_points, workers=1)[0]
    b = cKDTree(target_points).query(prediction_points, workers=1)[0]
    tp = int(np.count_nonzero(a <= tolerance))
    fn = len(target_points) - tp
    fp = int(np.count_nonzero(b > tolerance))
    return tp / (tp + fn + fp) if tp + fn + fp else 0.0


def ap50(targets: list[nx.Graph], predictions: list[nx.Graph], tolerance: float) -> float:
    """Same deterministic greedy AP50 as nested_loso_full_pipeline.py."""
    matrix = np.asarray([[component_iou(t, p, tolerance) for p in predictions]
                         for t in targets], dtype=float)
    matched: set[int] = set()
    true_positives = []
    for j in range(len(predictions)):
        best, best_index = 0.0, -1
        for i in range(len(targets)):
            if i not in matched and matrix[i, j] > best:
                best, best_index = float(matrix[i, j]), i
        hit = best >= 0.5 and best_index >= 0
        true_positives.append(float(hit))
        if hit:
            matched.add(best_index)
    if not true_positives:
        return 0.0
    cumulative = np.cumsum(true_positives)
    precision = cumulative / np.arange(1, len(true_positives) + 1)
    recall = np.r_[0.0, cumulative / max(1, len(targets)), 1.0]
    precision = np.r_[1.0, precision, 0.0]
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    return float(sum((recall[index] - recall[index - 1]) * precision[index]
                     for index in range(1, len(recall))))


def endpoints(graph: nx.Graph) -> np.ndarray:
    points = []
    for group in nx.connected_components(graph):
        sub = graph.subgraph(group)
        points.extend(np.asarray(sub.nodes[node]["pos"], dtype=float)
                      for node, degree in sub.degree() if degree == 1)
    return np.asarray(points, dtype=float) if points else np.empty((0, 3))


def evaluate(gt: nx.Graph, pred: nx.Graph, tree: cKDTree, prob: np.ndarray) -> dict:
    gp, pp = graph_points(gt), graph_points(pred)
    result = metrics(gp, pp)
    if not len(gp) or not len(pp):
        result["chamfer_mm"] = math.nan
        result["hausdorff_mm"] = math.nan
    targets = components(gt)
    predictions = ranked_components(pred, tree, prob)
    result["ap40"] = ap50(targets, predictions, .04)
    result["ap80"] = ap50(targets, predictions, .08)
    result["failure"] = int(not len(pp))
    result["gt_components"] = len(targets)
    result["pred_components"] = len(predictions)
    result["component_count_abs_error"] = abs(len(predictions) - len(targets))
    gt_end, pred_end = endpoints(gt), endpoints(pred)
    if len(gt_end) and len(pred_end):
        er = float(np.mean(cKDTree(pred_end).query(gt_end, workers=1)[0] <= .04))
        ep = float(np.mean(cKDTree(gt_end).query(pred_end, workers=1)[0] <= .04))
    else:
        er = ep = 0.0
    result["endpoint_f1_40mm"] = 2 * ep * er / (ep + er) if ep + er else 0.0
    return result
