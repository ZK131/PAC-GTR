# Reproducing the current four-module method

Use Python 3.11 for the validated reconstruction environment. Install this checkout:

```bash
python -m pip install -e ".[reconstruction,image,test]"
python -m unittest discover -s tests -p "test_current_reconstruction.py"
```

## Inputs and calibration

Obtain CrackStructures, CrackEnsembles and L515 from their original providers.
Retain the providers' licenses. Use the calibrated CrackStructures/ENSTRECT
projection and visibility workflow to map image probabilities onto the same
registered surface samples; camera calibration is an input, not estimated by
this backend. No geometry, imagery, annotations or weights are included here.

The current callable is `crack_reconstruction.pipeline.reconstruct`. Prepare an
NPZ containing `xyz` and `normals` (N by 3), and `donor_probabilities` and
`native_probabilities` (N). Keep point order identical in both fields.

For CrackStructures, use six views and 1M points with coordinates in meters.
The donor field is the visible-view mean of learned DUTR outputs. The native
field starts with the frozen Omni visible-view mean, then applies
`recover_native_probabilities` using point-by-camera probabilities and stored
angles in the original convention (valid finite angles at least 130 degrees).
The helper preserves strong seeds and limits the weak-seed increment to 0.06.

The image modules contain the directional context model, auxiliary TinyUNet,
and 8,691-parameter ContextCEGate. The gate returns **logits**: apply sigmoid
before mapping. `visible_view_mean` aggregates already projected probabilities
and an explicit visibility mask; it does not replace camera projection.

## Learned artifacts

Obtain or train the frozen Omni prior, context model, CE auxiliary model, DUTR
gate and WFR rankers separately. They are intentionally outside this code-only
release. For the ranker, create eleven route features with `route_features.features`,
standardize and fit class-balanced logistic regression on the other physical
structures for each held-out structure. Never fit on the evaluated structure.
`ranker.export_ranker` serializes a fitted sklearn pipeline to JSON for inference;
the repository contains no trained coefficient values. Use the original
structure-specific rankers when checking the historical graphs.

The original gate trainer is available with explicit paths:

```bash
python -m crack_reconstruction.train_gate \
  --prepared-root /your/prepared/l515 \
  --context-checkpoint /your/weights/context.pt \
  --ce-checkpoint /your/weights/ce.pt \
  --output /your/runs/dutr_gate --device cuda:0
```

The prepared L515 directory contains `manifest.json` (`items`, including id,
split, RGB path and `mask256` path) and `omni_fold0_probs256/NNNN.npy`. Relative
image paths resolve against the caller's working directory; absolute paths are
also accepted. Split sizes are 531 train / 224 validation / 245 test. Training retains
the original loss, augmentations,28-epoch cosine schedule and validation-mass
selection. The frozen context checkpoint has key `model`; the auxiliary checkpoint
has key `model`. Gate checkpoints have key `gate`. Frozen artifacts are never
silently downloaded or replaced by random weights.

## Backend execution

```bash
python -m crack_reconstruction \
  --input /your/prepared/registered_fields.npz \
  --ranker /your/weights/heldout_structure_ranker.json \
  --output /your/runs/centerline_graph.json
```

This executes the actual donor and native PAC/WFR/GTR branches and supported
integration. Geometry is reconstructed from input fields, not loaded from a
prediction cache. Node identifiers, insertion order and route competition are
retained from the executed research source; definitions and source hashes are
listed in `SOURCE_PROVENANCE.json`. The original `pac_gtr` and `topocracknet`
packages remain for historical compatibility; use `crack_reconstruction` for
the current manuscript method.

The CrackEnsembles evaluation uses coordinates divided by the robust scene
diagonal,10K points, six specified views, and256-pixel images. Its input adapter
assigns raw Omni to the donor field and learned DUTR to the other field and
retains the empty-branch seed gate. It is a distinct normalized protocol, not
the metric CrackStructures configuration. Follow `current_crackensembles.json`;
the historical `CrackEnsemblesProtocol()` class is a legacy PAC/GTR API.

## Verification scope

The relocated integration functions were compared with all15 frozen
CrackStructures output graphs. A complete beam_segment1 reconstruction from
the 1M-point fields was additionally compared with its frozen graph (414 nodes,
406 edges). This release verification does not rerun model training or claim
new numerical experiments. Exact historical scores still require the same
externally supplied inputs, learned artifacts, evaluator and software environment.
The paper's 23.59-second timing is not replaced by a migration-test timing.

Use `--domain crackensembles` to activate the normalized small-cloud and exact-order MST adapter. Run one reconstruction per process.

The principal scorer is `metrics_principal.evaluate(reference, prediction, probability_tree, probabilities)`. It retains the original bilateral counts, 10 mm sampling and bidirectional **average** Chamfer in millimeters. The transfer scorer `metrics_transfer.evaluate` returns average Chamfer in the input coordinate unit; multiply by 1,000 only for metric inputs. It preserves undefined distances for empty positive predictions. These are distinct overlap/endpoint protocols.
