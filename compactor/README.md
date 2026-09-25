# hepml-compactor

A standalone package that turns large simulation files (ROOT format, from the Delphes detector
simulation) into compact, typed Parquet. It is deployed to a CPU server next to the data and needs no
machine-learning libraries.

## What it does

- **Declarative recipes.** A YAML recipe names the object collections, fields, storage filters and the
  event selection (for example "at least three jets with pT > 25 GeV and |η| < 2.5"). Units are
  explicit; unsupported ones fail instead of being reinterpreted.
- **Bounded memory.** Files are read in chunks of a fixed decoded size; objects are stored as lists per
  event.
- **Resumable and verifiable.** Each chunk is written atomically and recorded in a manifest with its
  hash. Rerunning continues an interrupted export; a changed input, recipe, extraction code or library
  version is refused rather than mixed in.
- **Stable event identities.** Event IDs derive from the source file's identity and entry number, so
  exports made with different recipes can be joined event by event.
- **Accumulators.** Per-event quantities of every input event (for example generator weight ratios)
  can be summed while selected events are written.

## Usage

```bash
python -m hepml_compact --extraction studies/cg_bbc/extraction_truth_tag.yaml \
  --config samples.yaml --sample <sample key> --outdir exports/backgrounds
```

`scripts/package_server.py` builds a versioned ZIP with the package, the recipes and pinned library
constraints (`constraints-py311.txt`); `python -m hepml_compact.environment constraints-py311.txt`
checks a server environment against them. `python -m hepml_compact.merging_counts` counts, per input
file, the events the event generator accepted, from the stored generator weights.

Tests: `tests/test_compact*.py`, `tests/test_object_export.py`, `tests/test_extraction_runner.py` and
`tests/test_merging_counts.py`, all on synthetic files.
