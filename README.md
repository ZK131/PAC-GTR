# PAC-GTR: code-only reproducibility release

This repository contains a compact reference implementation of the paper-specific 3D post-processing components:

- **PAC**: probability-aware adaptive graph clustering;
- **GTR**: geometry-gated recovery of compatible centerline components;
- deterministic instance, endpoint, distance, and topology evaluation;
- a frozen protocol file and a synthetic smoke test.

## Scope

This is intentionally a **code-only** repository. It does not contain or redistribute:

- CrackStructures, CrackEnsembles, or any other third-party dataset;
- mesh, image, point-cloud, or annotation files;
- model checkpoints or frontend probability archives;
- raw experiment outputs, author information, or manuscript files.

The algorithms accept arrays of 3D points and per-point crack probabilities. Users must obtain datasets under their original licenses and prepare the inputs independently.

## Installation

```bash
python -m venv .venv
python -m pip install -e ".[test]"
```

## Synthetic smoke test

The demonstration creates two noisy collinear crack fragments in memory; it is not research data.

```bash
python scripts/demo_synthetic.py
pytest -q
```

Expected behavior:

- PAC identifies candidate components from 3D coordinates and probabilities;
- GTR joins only components that satisfy the frozen gap and alignment gates;
- the evaluator returns deterministic point, instance, endpoint, and topology metrics.

## Minimal API

```python
from pac_gtr import Protocol, predict_components

records, point_mask = predict_components(
    points_xyz,
    crack_probabilities,
    protocol=Protocol(),
    mode="full",  # baseline, pac, gate, gtr, or full
)
```

`full` combines PAC and GTR. The implementation is deterministic for fixed inputs.

## Evidence boundary

This repository exposes the core post-processing and evaluation logic. It does not by itself reproduce image-to-probability inference or the numerical tables in the manuscript because datasets, checkpoints, and frozen prediction/result archives are deliberately excluded.

CrackStructures is real multi-view data with manual centerline annotations. CrackEnsembles is semi-synthetic and uses procedurally derived centerline ground truth. Neither dataset is redistributed here.

## Relationship to ENSTRECT

The work extends the ENSTRECT workflow at the 3D post-processing and evaluation stages. The original ENSTRECT project is available at <https://github.com/ben-z-original/enstrect>. No upstream dataset, asset, checkpoint, or full source tree is copied into this release.

## License

The code in this repository is released under GPL-3.0-only to remain compatible with the upstream ENSTRECT project. See `LICENSE` and `THIRD_PARTY.md`.

