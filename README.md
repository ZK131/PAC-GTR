# Probability-aware multiview crack reconstruction

This code-only repository contains the current four-module reconstruction
method: directional uncertainty-guided image refinement (DUTR), probability-aware
clustering (PAC), adaptive weak-fragment recovery (WFR), and geometry-topology
refinement (GTR). The current implementation is in `crack_reconstruction/`.

## Current implementation

- Frozen context/auxiliary image model definitions, the 8,691-parameter DUTR gate,
  and the original gate training objective and validation selection.
- Visibility-aware probability aggregation and multiview/spatial native weak-seed recovery.
- Separate donor and native graph branches with the actual executed PAC, WFR,
  ranked route selection, local GTR and supported endpoint continuation.
- Cross-branch continuation, component grafting, backbone/pruning protection,
  local recovery and source-connected anchor restoration.
- Dataset-specific configurations and the principal and equal-arc evaluation code.
- A command-line reconstruction entry point, automated checks and source hashes.

The original `pac_gtr/` and `topocracknet/` APIs remain for compatibility with
the earlier code release. They are separate from the current four-module API.

## Install and run

Python 3.11 is the validated reconstruction environment. Obtain datasets and
learned artifacts separately, then install:

```bash
python -m pip install -e ".[reconstruction,image,test]"
python -m unittest discover -s tests -p "test_current_reconstruction.py"
python -m crack_reconstruction --help
```

Reconstruct from aligned coordinates, normals and the two prepared probability fields:

```bash
python -m crack_reconstruction \
  --input /your/prepared/registered_fields.npz \
  --ranker /your/weights/heldout_structure_ranker.json \
  --output /your/runs/centerline_graph.json
```

See [the reproduction guide](docs/REPRODUCING_CURRENT_METHOD.md) for calibration,
NPZ fields, image training, ranker fitting, probability roles and dataset adapters.
The current physical and normalized configurations are
`configs/current_crackstructures.json` and `configs/current_crackensembles.json`.
Do not substitute the legacy `configs/crackensembles_protocol.json` for the
current four-module adapter.

## Verification

The relocated integration code was checked against all 15 frozen CrackStructures
graphs. A complete reconstruction from the one-million-point beam_segment1
fields also matched the frozen graph, with 414 nodes and 406 edges. A normalized
CrackEnsembles reconstruction matched its frozen graph (64 nodes,60 edges).
Seven public
checks cover visibility, seed recovery, image-gate initialization, learned-ranker
serialization, empty seeds and the Chamfer definition. These are migration
checks, not new experiments or a claim that model training was repeated.

## Data and learned artifacts

The repository distributes source code, configurations and instructions. It
does not redistribute third-party datasets, photographs, surface samples,
annotations, trained model/ranker weights or raw prediction archives. Users
obtain these under the original providers' terms and supply their own prepared
inputs and learned artifacts. `docs/SOURCE_PROVENANCE.json` traces the released
definitions to the executed research source; `docs/VALIDATION_ENVIRONMENT.json`
identifies the environment used for release checks, not historical timing hardware.

## Upstream and license

The calibrated mapping and contraction workflow builds on
[ENSTRECT](https://github.com/ben-z-original/enstrect) and
[CrackStructures](https://github.com/ben-z-original/crackstructures).
Selected ENSTRECT-derived extraction code is included under GPL-3.0-only,
together with the authors' added reconstruction modules. See LICENSE and
THIRD_PARTY.md. No complete upstream project, dataset or pretrained weight
distribution is copied into this repository.
