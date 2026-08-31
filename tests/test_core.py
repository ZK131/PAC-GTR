import numpy as np

from pac_gtr import Protocol, evaluate_components, predict_components


def fixture():
    rng = np.random.default_rng(7)
    x1 = np.linspace(0.00, 0.45, 60)
    x2 = np.linspace(0.55, 1.00, 60)
    left = np.column_stack([x1, np.zeros_like(x1), np.zeros_like(x1)])
    right = np.column_stack([x2, np.zeros_like(x2), np.zeros_like(x2)])
    crack = np.vstack([left, right]) + rng.normal(0, 0.001, size=(120, 3))
    background = rng.uniform([-0.1, -0.2, -0.1], [1.1, 0.2, 0.1], size=(80, 3))
    points = np.vstack([crack, background])
    probabilities = np.r_[np.full(len(crack), 0.9), np.full(len(background), 0.05)]
    protocol = Protocol(
        eps_ratio=0.018,
        topology_gap_ratio=0.12,
        min_samples=3,
        min_component_points=5,
        merge_alignment_cos=0.9,
    )
    return points, probabilities, protocol


def test_full_is_deterministic():
    points, probabilities, protocol = fixture()
    first, first_mask = predict_components(points, probabilities, protocol=protocol, mode="full")
    second, second_mask = predict_components(points, probabilities, protocol=protocol, mode="full")
    assert np.array_equal(first_mask, second_mask)
    assert [component.canonical_id for component in first] == [component.canonical_id for component in second]
    assert len(first) == 1


def test_empty_predictions_are_failures():
    points, _, protocol = fixture()
    predictions, mask = predict_components(points, np.zeros(len(points)), protocol=protocol, mode="full")
    truth = [np.linspace([0, 0, 0], [1, 0, 0], 50)]
    truth_mask = np.zeros(len(points), dtype=bool)
    metrics = evaluate_components(predictions, mask, truth, truth_mask, points, failure_penalty=1.0)
    assert metrics["failure"] == 1
    assert metrics["normalized_chamfer"] == 1.0


def test_all_ablation_modes_run():
    points, probabilities, protocol = fixture()
    for mode in ("baseline", "pac", "gate", "gtr", "full"):
        predictions, mask = predict_components(points, probabilities, protocol=protocol, mode=mode)
        assert mask.shape == (len(points),)
        assert all(component.line.shape == (protocol.line_points, 3) for component in predictions)
