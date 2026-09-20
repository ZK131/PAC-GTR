# TopoCrackNet and PAC-GTR: code-only reproducibility release

This repository contains the original code for two complementary stages of a
multi-view 2.5D crack-reconstruction workflow:

- **TopoCrackNet**: RGB crack segmentation with region, centreline and boundary heads;
- **PAC**: probability-aware adaptive graph clustering;
- **GTR**: geometry-gated recovery of compatible centerline components;
- deterministic instance, endpoint, distance, and topology evaluation;
- frozen protocol files for L515, CrackStructures and CrackEnsembles;
- a synthetic PAC-GTR smoke test and an L515 TopoCrackNet training entry point.

## Scope

This is intentionally a **code-only** repository. It does not contain or redistribute:

- CrackStructures, CrackEnsembles, or any other third-party dataset;
- mesh, image, point-cloud, or annotation files;
- model checkpoints, frontend probability archives, or trained weights;
- raw experiment outputs, author information, or manuscript files.

Users must obtain all datasets under their original licenses and prepare the
inputs independently. The TopoCrackNet trainer expects a prepared L515 array
split; PAC-GTR accepts arrays of 3D points and per-point crack probabilities.

## Installation

```bash
python -m venv .venv
python -m pip install -e ".[test,train]"
```

## TopoCrackNet: 2D crack segmentation

TopoCrackNet is a lightweight U-Net with region, centreline and boundary
outputs. The centreline and boundary branches are used only as training
supervision; inference uses the sigmoid probability of the region head.

The frozen L515 protocol is documented in
`configs/topocracknet_l515.json`. The split has 531 training images, 224
validation images and 245 test images. The dataset is not redistributed here.

Prepare a directory containing the six arrays below:

```text
/path/to/l515_scene_split/data/
  data_train.npy    mask_train.npy
  data_val.npy      mask_val.npy
  data_test.npy     mask_test.npy
```

Train with the fixed default configuration:

```bash
python scripts/train_topocracknet_l515.py \
  --data-root /path/to/l515_scene_split/data \
  --output-dir runs/topocracknet_l515
```

The script selects the checkpoint with the highest validation F1 and writes
`history.csv`, `summary.json` and the selected checkpoint to the supplied
output directory. The default `runs/` directory is excluded by `.gitignore`.

For inference in a separate Python workflow:

```python
import torch
from topocracknet import TopoCrackNet

model = TopoCrackNet()
checkpoint = torch.load("best.pt", map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval()
region_probability = model.predict_region(rgb_images)
```

`rgb_images` must have shape `[batch, 3, height, width]` and values in
`[0, 1]`.

## PAC-GTR: 3D probability-point-cloud post-processing

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

The two datasets do not share one parameterization. CrackEnsembles uses distances normalized by a robust per-sample diagonal, whereas CrackStructures uses metric distances in meters. Select the protocol explicitly:

```python
from pac_gtr import CrackEnsemblesProtocol, predict_components

records, point_mask = predict_components(
    points_xyz,
    crack_probabilities,
    protocol=CrackEnsemblesProtocol(),
    mode="full",  # baseline, pac, gate, gtr, or full
)
```

For metric CrackStructures inputs, use `CrackStructuresProtocol()`. Its frozen full branch uses threshold 0.25, PAC epsilon 0.035 m, and GTR maximum gap 0.04 m.

The complete machine-readable configurations are:

- `configs/crackensembles_protocol.json`
- `configs/crackstructures_protocol.json`

`full` combines PAC and GTR. The implementation is deterministic for fixed inputs.

## Reproducibility boundary

This repository exposes the TopoCrackNet training implementation and the
PAC-GTR post-processing and evaluation logic. It does not redistribute
datasets, model weights, raw prediction archives or manuscript numerical
outputs. Users should reproduce results with data obtained from the original
providers and should cite the underlying datasets.

CrackStructures is real multi-view data with manual centerline annotations. CrackEnsembles is semi-synthetic and uses procedurally derived centerline ground truth. Neither dataset is redistributed here.

## Relationship to ENSTRECT and CrackStructures

The 3D stage extends the ENSTRECT workflow at post-processing and evaluation.
The multi-view data and semantic-mapping workflow are available from
CrackStructures at <https://github.com/ben-z-original/crackstructures>; the
original ENSTRECT project is available at
<https://github.com/ben-z-original/enstrect>. No upstream dataset, asset,
checkpoint or complete source tree is copied into this release.

## License

The code in this repository is released under GPL-3.0-only to remain compatible with the upstream ENSTRECT project. See `LICENSE` and `THIRD_PARTY.md`.
