"""Deterministic evaluation for 3D crack centerline components."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .core import Component, robust_diagonal


def _direction(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(3, dtype=float)
    axis = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)[2][0]
    pivot = int(np.argmax(np.abs(axis)))
    return axis if axis[pivot] >= 0 else -axis


def symmetric_distance(left: np.ndarray, right: np.ndarray, scale: float) -> float:
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    a = cKDTree(right).query(left, k=1)[0].mean()
    b = cKDTree(left).query(right, k=1)[0].mean()
    return float((a + b) / (2 * scale))


def _distance_matrix(predictions: list[Component], truths: list[np.ndarray], scale: float) -> np.ndarray:
    matrix = np.full((len(predictions), len(truths)), np.inf, dtype=float)
    for i, prediction in enumerate(predictions):
        for j, truth in enumerate(truths):
            matrix[i, j] = symmetric_distance(prediction.line, truth, scale)
    return matrix


def deterministic_ap(
    predictions: list[Component],
    truths: list[np.ndarray],
    distances: np.ndarray,
    tolerance: float,
) -> float:
    if not truths:
        return 0.0
    order = sorted(
        range(len(predictions)),
        key=lambda i: (
            -predictions[i].confidence,
            -len(predictions[i].indices),
            predictions[i].canonical_id,
        ),
    )
    used: set[int] = set()
    hits = 0
    precisions: list[float] = []
    for rank, i in enumerate(order, start=1):
        best_distance, best_truth = min(
            ((distances[i, j], j) for j in range(len(truths)) if j not in used),
            default=(np.inf, -1),
        )
        if best_distance <= tolerance:
            used.add(best_truth)
            hits += 1
            precisions.append(hits / rank)
    return float(sum(precisions) / len(truths))


def endpoint_f1(
    predictions: list[Component],
    truths: list[np.ndarray],
    scale: float,
    tolerance: float,
) -> tuple[float, float, float]:
    predicted = (
        np.concatenate([component.endpoints for component in predictions], axis=0)
        if predictions
        else np.empty((0, 3), dtype=float)
    )
    target = []
    for truth in truths:
        axis = _direction(truth)
        coordinate = (truth - truth.mean(axis=0)) @ axis
        target.extend([truth[np.argmin(coordinate)], truth[np.argmax(coordinate)]])
    target = np.asarray(target, dtype=float)
    if not len(predicted) or not len(target):
        return 0.0, 0.0, 0.0
    precision = float((cKDTree(target).query(predicted, k=1)[0] <= tolerance * scale).mean())
    recall = float((cKDTree(predicted).query(target, k=1)[0] <= tolerance * scale).mean())
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def evaluate_components(
    predictions: list[Component],
    predicted_mask: np.ndarray,
    truths: list[np.ndarray],
    truth_mask: np.ndarray,
    reference_points: np.ndarray,
    *,
    nap_strict: float = 0.01,
    nap_relaxed: float = 0.03,
    endpoint_tolerance: float = 0.03,
    component_match_tolerance: float = 0.03,
    failure_penalty: float = 1.0,
) -> dict[str, float | int]:
    """Return point, distance, instance, endpoint, and topology metrics.

    Empty predictions are retained as failures. Their normalized Chamfer and
    Hausdorff values receive ``failure_penalty`` rather than being dropped.
    """
    predicted_mask = np.asarray(predicted_mask, dtype=bool)
    truth_mask = np.asarray(truth_mask, dtype=bool)
    if predicted_mask.shape != truth_mask.shape or len(predicted_mask) != len(reference_points):
        raise ValueError("point masks and reference points must have matching lengths")
    scale = robust_diagonal(reference_points)
    tp = int((predicted_mask & truth_mask).sum())
    fp = int((predicted_mask & ~truth_mask).sum())
    fn = int((~predicted_mask & truth_mask).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    point_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)

    truth_points = np.concatenate(truths, axis=0) if truths else np.empty((0, 3), dtype=float)
    if predictions and len(truth_points):
        predicted_points = np.concatenate([component.line for component in predictions], axis=0)
        a = cKDTree(truth_points).query(predicted_points, k=1)[0]
        b = cKDTree(predicted_points).query(truth_points, k=1)[0]
        normalized_chamfer = float((a.mean() + b.mean()) / (2 * scale))
        normalized_hausdorff = float(max(a.max(), b.max()) / scale)
    else:
        normalized_chamfer = float(failure_penalty)
        normalized_hausdorff = float(failure_penalty)

    distances = _distance_matrix(predictions, truths, scale)
    strict = deterministic_ap(predictions, truths, distances, nap_strict)
    relaxed = deterministic_ap(predictions, truths, distances, nap_relaxed)
    ep, er, ef = endpoint_f1(predictions, truths, scale, endpoint_tolerance)

    if predictions and truths:
        rows, cols = linear_sum_assignment(distances)
        matched = int(sum(distances[i, j] <= component_match_tolerance for i, j in zip(rows, cols)))
        near_counts = (distances <= component_match_tolerance).sum(axis=0)
        fragmentation = float(np.maximum(near_counts - 1, 0).sum() / len(truths))
    else:
        matched = 0
        fragmentation = 0.0

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": point_f1,
        "normalized_chamfer": normalized_chamfer,
        "normalized_hausdorff": normalized_hausdorff,
        "nAP_strict": strict,
        "nAP_relaxed": relaxed,
        "endpoint_precision": ep,
        "endpoint_recall": er,
        "endpoint_f1": ef,
        "pred_components": len(predictions),
        "gt_components": len(truths),
        "component_count_error": abs(len(predictions) - len(truths)),
        "fragmentation": fragmentation,
        "component_recall": matched / max(len(truths), 1),
        "failure": int(len(predictions) == 0),
    }

