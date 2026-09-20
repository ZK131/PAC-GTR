# Upload checklist

This folder is the complete code-only upload package for the public
repository `ZK131/PAC-GTR`.

Before uploading, verify that the repository contains only this folder's
contents. Do not add datasets, raw images, point clouds, annotations, trained
weights, prediction archives, experiment outputs, author information, or the
manuscript.

Recommended upload steps:

1. Replace the repository root contents with this folder's contents.
2. Keep `LICENSE`, `THIRD_PARTY.md`, `.gitignore`, `README.md`, and
   `pyproject.toml` at the repository root.
3. Run `python -m pip install -e ".[test,train]"` and `pytest -q` in a clean
   environment before tagging a release.
4. In the GitHub release description, state that this is a code-only release;
   users must obtain CrackStructures and CrackEnsembles from their original
   providers under the applicable licenses.

The paper should cite the repository URL and should not state that L515 data
or model checkpoints are available from this release.
