"""Count MLM-accepted events per ROOT file from the Weight branch alone.

With JetMatching:doVeto=off, Pythia writes matching-rejected events with zero weight
at the rejecting merging scale; only events rejected at every scale of
SysCalc:qCutList are dropped. Per event, the Weight branch then holds the nominal
weight (identical for every event of a run), one merging weight per scale
(nominal / N_generated when accepted, otherwise 0) and scale variations, which
Pythia evaluates at the central merging scale only.

No column position is assumed: HepMC sorts weight names as text, so the central
scale's column differs between productions. The central merging column is the
merging column whose zero pattern the scale variations share.

The implied merged cross section is only a consistency check against the manifest's
Pythia merged rate, never a normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from .config import load_compact_sample
from .root_reader import source_id_for

BRANCH = "Weight/Weight.Weight"
# Weights are stored as float32: a few units in the last place.
RELATIVE_TOLERANCE = 1e-6


class _Columns:
    """Per-column statistics accumulated over chunks."""

    def __init__(self, width: int):
        self.width = width
        self.entries = 0
        self.nonzero = np.zeros(width, dtype=np.int64)
        self.sums = np.zeros(width)
        self.first = None
        self.constant = np.ones(width, dtype=bool)
        self.value = np.full(width, np.nan)  # common non-zero value, if any
        self.value_constant = np.ones(width, dtype=bool)
        self.patterns = [hashlib.sha256() for _ in range(width)]

    def add(self, weights: np.ndarray) -> None:
        mask = weights != 0
        if self.first is None:
            self.first = weights[0].copy()
        self.constant &= (weights == self.first).all(axis=0)
        for column in range(self.width):
            values = weights[mask[:, column], column]
            if len(values):
                if np.isnan(self.value[column]):
                    self.value[column] = values[0]
                self.value_constant[column] &= bool(np.all(values == self.value[column]))
            self.patterns[column].update(np.packbits(mask[:, column]).tobytes())
            self.patterns[column].update(len(mask).to_bytes(8, "little"))
        self.entries += len(weights)
        self.nonzero += mask.sum(axis=0)
        self.sums += weights.sum(axis=0)


def identify(columns: _Columns) -> dict:
    """Nominal, merging and central merging columns, from the values alone."""
    sometimes_zero = (columns.nonzero > 0) & (columns.nonzero < columns.entries)
    nominal = [j for j in range(columns.width) if columns.constant[j] and columns.nonzero[j] == columns.entries]
    if len(nominal) != 1:
        raise ValueError(f"Expected one weight constant and non-zero in every event, found columns {nominal}")
    nominal = nominal[0]
    total = float(columns.first[nominal])
    merging = {}
    for j in range(columns.width):
        if j == nominal or not columns.value_constant[j] or np.isnan(columns.value[j]):
            continue
        ratio = total / float(columns.value[j])
        # A merging weight is nominal / N_generated, and N_generated >= stored events.
        if ratio >= columns.entries and abs(ratio - round(ratio)) <= RELATIVE_TOLERANCE * ratio:
            merging[j] = round(ratio)
    if not merging or len(set(merging.values())) != 1:
        raise ValueError(f"Expected merging weights sharing one generated count, found {merging}")
    digests = [pattern.hexdigest() for pattern in columns.patterns]
    shared = {j: [k for k in range(columns.width) if k not in merging and k != nominal and sometimes_zero[k]
                  and digests[k] == digests[j]] for j in merging}
    central = [j for j in merging if shared[j]]
    if len(central) != 1:
        raise ValueError(f"Expected one merging weight whose zeros the scale variations share, found {central}")
    central = central[0]
    return {
        "nominal_column": nominal,
        "nominal_weight": total,
        "n_generated": merging[central],
        "merging_columns": {str(j): int(columns.nonzero[j]) for j in merging},
        "central_column": central,
        "scale_variation_columns": shared[central],
        "accepted": int(columns.nonzero[central]),
    }


def count_file(path: Path, *, tree_name: str = "Delphes", step_size: str | int = "50 MB") -> dict:
    path = path.resolve(strict=True)
    with uproot.open(path) as root:
        uuid = str(root.file.uuid)
        tree = root[tree_name]
        columns = None
        for chunk in tree.iterate([BRANCH], step_size=step_size, library="ak"):
            jagged = chunk[BRANCH]
            widths = np.unique(ak.to_numpy(ak.num(jagged)))
            if len(widths) != 1 or (columns is not None and widths[0] != columns.width):
                raise ValueError(f"{path}: the number of weights varies between events")
            columns = columns or _Columns(int(widths[0]))
            columns.add(ak.to_numpy(jagged).astype(np.float64))
        if columns is None:
            raise ValueError(f"{path}: no events")
    record = {"path": str(path), "source_id": source_id_for(uuid, tree_name), "root_uuid": uuid,
              "entries": columns.entries, "n_weights": columns.width}
    record |= identify(columns)
    record["column_sums"] = columns.sums.tolist()
    record["column_nonzero"] = columns.nonzero.tolist()
    return record


def count_sample(files, sample: dict, **options) -> dict:
    records = []
    for path in files:
        start = time.monotonic()
        records.append(count_file(Path(path), **options))
        logging.info("%s: %d stored, %d accepted, column %d, %.0f s", path, records[-1]["entries"],
                     records[-1]["accepted"], records[-1]["central_column"], time.monotonic() - start)
    generated = sum(r["n_generated"] for r in records)
    result = {
        "kind": sample.get("kind"),
        "xs_pb": sample.get("xs_pb"),
        "n_stored": sum(r["entries"] for r in records),
        "n_accepted": sum(r["accepted"] for r in records),
        "n_generated": generated,
        "files": records,
    }
    # Check only: the Pythia merged rate of the pooled runs, from the files' own weights.
    implied = sum(r["nominal_weight"] * r["accepted"] for r in records) / generated
    result["implied_merged_xs"] = implied
    if sample.get("xs_pb"):
        result["xs_pb_over_implied"] = sample["xs_pb"] / implied
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path, help="Sample manifest, e.g. samples.ready.yaml")
    parser.add_argument("--output", required=True, type=Path, help="New JSON file; existing files are refused")
    parser.add_argument("--sample", action="append", help="Sample key (repeatable); default: every sample")
    parser.add_argument("--data-root", help="Override the manifest data_root")
    parser.add_argument("--tree", default="Delphes")
    parser.add_argument("--step-size", default="50 MB")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.output.exists():
        parser.error(f"{args.output} exists; choose a new file")
    import yaml

    step_size = int(args.step_size) if args.step_size.isdigit() else args.step_size
    keys = args.sample or list((yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}).get("samples", {}))
    result = {"schema": 1, "branch": BRANCH, "samples": {}}
    for key in keys:
        files, sample = load_compact_sample(args.config, key, args.data_root)
        result["samples"][key] = count_sample(files, sample, tree_name=args.tree, step_size=step_size)
        summary = result["samples"][key]
        logging.info("%s: %d of %d stored events accepted; manifest xs / implied %s", key, summary["n_accepted"],
                     summary["n_stored"], f"{summary.get('xs_pb_over_implied', float('nan')):.4f}")
    temporary = args.output.with_name(args.output.name + ".partial")
    temporary.write_text(json.dumps(result, indent=1), encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
