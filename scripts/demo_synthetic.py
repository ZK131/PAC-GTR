"""Run PAC-GTR on synthetic in-memory fragments (no dataset files required)."""
from __future__ import annotations

import json

import numpy as np

from pac_gtr import Protocol, evaluate_components, predict_components


def sample_line(start, stop, count, rng):
    t = np.linspace(0.0, 1.0, count)[:, None]
    return start + t * (stop - start) + rng.normal(0.0, 0.0015, size=(count, 3))


def main() -> None:
    rng = np.random.default_rng(42)
    left = sample_line(np.array([0.00, 0.0, 0.0]), np.array([0.46, 0.0, 0.0]), 70, rng)
    right = sample_line(np.array([0.54, 0.0, 0.0]), np.array([1.00, 0.0, 0.0]), 70, rng)
    background = rng.uniform([-0.1, -0.2, -0.1], [1.1, 0.2, 0.1], size=(100, 3))
    points = np.vstack([left, right, background])
    probabilities = np.concatenate([
        rng.uniform(0.82, 0.98, len(left) + len(right)),
        rng.uniform(0.01, 0.20, len(background)),
    ])
    protocol = Protocol(
        threshold=0.60,
        eps_ratio=0.018,
        topology_gap_ratio=0.12,
        min_samples=3,
        min_component_points=5,
        merge_alignment_cos=0.90,
    )
    predictions, mask = predict_components(points, probabilities, protocol=protocol, mode="full")
    truth = [np.linspace([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100)]
    truth_mask = np.zeros(len(points), dtype=bool)
    truth_mask[: len(left) + len(right)] = True
    metrics = evaluate_components(predictions, mask, truth, truth_mask, points)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
