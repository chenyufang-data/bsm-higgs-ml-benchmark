"""Truth-tagged benchmark rows: one drawn tag outcome per event, with its split and weights.

Layout under a dataset root (the common background is stored once, not per point):
  background/{train,val,test}.parquet          every background event
  <coupling>/sig<mass>_{train,val,test}.parquet  the point's signal events
  <coupling>/sig<mass>.meta.json               per-sample weight totals and provenance
A point's split is its signal rows followed by the background rows, as in the direct
benchmarks. sample_weight = the sample's uniform stored weight x the pass probability;
fit_weight = the pass probability.

With the matching acceptance (validation.yaml: matching_acceptance), only events accepted
at the central merging scale are kept, and the uniform weight uses the accepted count of
the sample's ROOT files instead of the stored count.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from hepml_compact.merging_counts import _Columns, identify

from hepml.domain.flavour_overlap import outgoing_counts
from hepml.domain.truth_tagging import draw_outcomes, event_uniforms
from hepml.domain.weights import compute_sample_weight

SPLITS = ("train", "val", "test")
KINEMATICS = ("pt", "eta", "phi", "mass")
ROLES = ("b1", "b2", "c1")
JET_COLUMNS = [f"jets_{field}" for field in (*KINEMATICS, "btag", "truth_flavor")]


def export_shards(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    return [Path(manifest_path).parent / chunk["file"] for chunk in manifest["chunks"] if chunk["file"]]


def flat_jets(table):
    """Offsets and flat numpy arrays of the jet columns (one shared list layout)."""
    first = table["jets_pt"].combine_chunks()
    offsets = first.offsets.to_numpy().astype(np.int64)
    offsets = offsets - offsets[0]
    values = {}
    for column in JET_COLUMNS:
        array = table[column].combine_chunks()
        if not np.array_equal(array.offsets.to_numpy() - array.offsets.to_numpy()[0], offsets):
            raise ValueError(f"{column} is not aligned with jets_pt")
        values[column.removeprefix("jets_")] = array.flatten().to_numpy(zero_copy_only=False)
    return offsets, values


def truth_tag_rows(manifest_path, *, sample, stored_lumi, model, plugin, seed, split_of):
    """One row per event of a truth-tag export: a drawn passing outcome, its features, split and weights."""
    manifest = json.loads(Path(manifest_path).read_text())
    metadata = manifest["sample_metadata"]
    n_total = metadata.get("n_generated", sum(source["entries"] for source in manifest["sources"]))
    weight = compute_sample_weight(float(metadata["xs_pb"]), float(n_total), lumi=stored_lumi)
    frames = []
    for path in export_shards(manifest_path):
        table = pq.read_table(path, columns=["event_id", *JET_COLUMNS])
        ids = table["event_id"].to_numpy(zero_copy_only=False)
        offsets, jets = flat_jets(table)
        roles, pass_probability = draw_outcomes(offsets, jets["pt"], jets["eta"], jets["truth_flavor"], jets["btag"],
                                                model, event_uniforms(ids, seed))
        if (roles["b1"] < 0).any():
            raise ValueError(f"{path}: an event has fewer than three jets in acceptance")
        objects = {role: tuple(jets[field][roles[role]] for field in KINEMATICS) for role in ROLES}
        frame = plugin.features_from_objects(objects, retained=True).reset_index(drop=True)
        frame.insert(0, "event_id", ids)
        frames.append(frame.assign(sample_key=manifest["sample"], sample=sample,
                                   target=int(manifest["kind"] == "signal"), fit_weight=pass_probability,
                                   sample_weight=weight * pass_probability, xs_pb=float(metadata["xs_pb"]),
                                   n_events_total=float(n_total)))
    rows = pd.concat(frames, ignore_index=True)
    split = split_of.reindex(rows.event_id.to_numpy()).to_numpy()
    if pd.isna(split).any():
        raise ValueError(f"{manifest['sample']}: truth-tag events without a registry split")
    return rows.assign(split=split)


def load_merging_counts(path):
    """Per-sample records of `hepml_compact.merging_counts`, keyed by export sample key."""
    record = json.loads(Path(path).read_text())
    if record.get("schema") != 1 or record.get("branch") != "Weight/Weight.Weight":
        raise ValueError(f"{path}: not a merging-counts record")
    return record["samples"]


def matching_acceptance(manifest_path, counted):
    """Per event: accepted at the central merging scale, read from the column its file's count
    identified. The same identification repeated on the exported weights, per source, is returned
    for the pre-registered comparison."""
    central = {record["source_id"]: record["central_column"] for record in counted["files"]}
    ids, flags, columns = [], [], {}
    for path in export_shards(manifest_path):
        table = pq.read_table(path, columns=["event_id", "source_id", "truth_lhe_weights"])
        weights = table["truth_lhe_weights"].combine_chunks()
        values = weights.flatten().to_numpy()
        weights = values.reshape(len(weights), -1) if len(weights) else values.reshape(0, 0)
        sources = pd.Series(table["source_id"].to_numpy(zero_copy_only=False))
        column = sources.map(central)
        if column.isna().any():
            raise ValueError(f"{path}: events from a ROOT file without a merging count")
        flags.append(weights[np.arange(len(weights)), column.to_numpy(dtype=np.int64)] != 0)
        ids.append(table["event_id"].to_numpy(zero_copy_only=False))
        for source in sources.unique():
            rows = weights[(sources == source).to_numpy()].astype(np.float64)
            columns.setdefault(source, _Columns(rows.shape[1])).add(rows)
    identified = {source: identify(statistics) for source, statistics in columns.items()}
    return pd.Series(np.concatenate(flags), index=pd.Index(np.concatenate(ids))), identified


def hard_process_counts(manifest_path, outgoing_status):
    """Per event of a parton export (extraction_partons.yaml): flavour-class counts of its
    outgoing hard-process partons, indexed by event ID."""
    ids, frames = [], []
    for path in export_shards(manifest_path):
        table = pq.read_table(path, columns=["event_id", "truth_partons_pid", "truth_partons_status"])
        pid, status = (table[name].combine_chunks() for name in ("truth_partons_pid", "truth_partons_status"))
        offsets = pid.offsets.to_numpy().astype(np.int64)
        if not np.array_equal(status.offsets.to_numpy(), pid.offsets.to_numpy()):
            raise ValueError(f"{path}: parton status is not aligned with parton ID")
        frames.append(pd.DataFrame(outgoing_counts(offsets - offsets[0], pid.flatten().to_numpy(),
                                                   status.flatten().to_numpy(), outgoing_status)))
        ids.append(table["event_id"].to_numpy(zero_copy_only=False))
    frame = pd.concat(frames, ignore_index=True)
    frame.index = pd.Index(np.concatenate(ids))
    return frame


def apply_acceptance(rows, accepted, n_accepted):
    """Keep accepted events; their uniform weight becomes xs / n_accepted instead of xs / n_stored."""
    keep = accepted.reindex(rows.event_id.to_numpy())
    if keep.isna().any():
        raise ValueError("Rows without an acceptance flag")
    kept = rows[keep.to_numpy(dtype=bool)].reset_index(drop=True)
    return kept.assign(sample_weight=kept.sample_weight * kept.n_events_total / n_accepted,
                       n_events_total=float(n_accepted))


def weighted_quantiles(values, weights, quantiles):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(np.asarray(weights, dtype=float)[order])
    positions = np.searchsorted(cumulative, np.asarray(quantiles) * cumulative[-1], side="left")
    return np.asarray(values)[order][np.minimum(positions, len(order) - 1)]


def sampling_check(manifest_path, *, split_of, split, model, plugin, seed, bins):
    """Per feature, drawn rows against the all-outcome distribution: {feature: (chi2, dof)}.

    Events of one split only; bins at weighted deciles (for bins=10) of the drawn rows.
    """
    from hepml.domain.truth_tagging import enumerate_outcomes, sampling_chi2

    entry_features, drawn_features, entry_event, entry_probability, drawn_probability = [], [], [], [], []
    offset = 0
    for path in export_shards(manifest_path):
        table = pq.read_table(path, columns=["event_id", *JET_COLUMNS])
        keep = split_of.reindex(table["event_id"].to_numpy(zero_copy_only=False)).to_numpy() == split
        table = table.filter(keep)
        ids = table["event_id"].to_numpy(zero_copy_only=False)
        offsets, jets = flat_jets(table)
        entries, _ = enumerate_outcomes(offsets, jets["pt"], jets["eta"], jets["truth_flavor"], jets["btag"], model)
        roles, pass_probability = draw_outcomes(offsets, jets["pt"], jets["eta"], jets["truth_flavor"], jets["btag"],
                                                model, event_uniforms(ids, seed))
        for source, target in ((entries, entry_features), (roles, drawn_features)):
            target.append(plugin.features_from_objects(
                {role: tuple(jets[field][source[role]] for field in KINEMATICS) for role in ROLES}))
        entry_event.append(entries["event"] + offset)
        entry_probability.append(entries["probability"])
        drawn_probability.append(pass_probability)
        offset += len(ids)
    entry_features, drawn_features = pd.concat(entry_features), pd.concat(drawn_features)
    entry_event, entry_probability = np.concatenate(entry_event), np.concatenate(entry_probability)
    drawn_probability = np.concatenate(drawn_probability)
    results = {}
    for feature in drawn_features:
        drawn = drawn_features[feature].to_numpy(dtype=float)
        edges = np.unique(weighted_quantiles(drawn, drawn_probability, np.linspace(0, 1, bins + 1)[1:-1]))
        k = len(edges) + 1
        expected = np.bincount(entry_event * k + np.searchsorted(edges, entry_features[feature].to_numpy(dtype=float),
                                                                 side="right"),
                               weights=entry_probability, minlength=offset * k).reshape(offset, k)
        results[feature] = sampling_chi2(expected, drawn_probability, np.searchsorted(edges, drawn, side="right"))
    return results


class TruthTagBenchmark:
    """Read and write the shared-background layout above."""

    def __init__(self, root):
        self.root = Path(root)
        self._background = {}

    def background_path(self, split):
        return self.root / "background" / f"{split}.parquet"

    def signal_path(self, directory, mass, split):
        return self.root / directory / f"sig{mass}_{split}.parquet"

    def meta_path(self, directory, mass):
        return self.root / directory / f"sig{mass}.meta.json"

    def write_background(self, rows):
        (self.root / "background").mkdir(parents=True, exist_ok=False)
        for split in SPLITS:
            rows[rows.split == split].drop(columns="split").to_parquet(self.background_path(split), index=False)

    def write_signal(self, directory, mass, rows, meta):
        (self.root / directory).mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            rows[rows.split == split].drop(columns="split").to_parquet(self.signal_path(directory, mass, split),
                                                                        index=False)
        self.meta_path(directory, mass).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    def compare_background(self, rows):
        """Resuming: the stored background equals a fresh derivation, value for value."""
        for split in SPLITS:
            expected = rows[rows.split == split].drop(columns="split").reset_index(drop=True)
            pd.testing.assert_frame_equal(pd.read_parquet(self.background_path(split)), expected, check_dtype=False,
                                          check_exact=True)

    def compare_signal(self, directory, mass, rows, meta):
        """Resuming: a point's stored signal rows and metadata equal a fresh derivation."""
        for split in SPLITS:
            expected = rows[rows.split == split].drop(columns="split").reset_index(drop=True)
            pd.testing.assert_frame_equal(pd.read_parquet(self.signal_path(directory, mass, split)), expected,
                                          check_dtype=False, check_exact=True)
        if self.meta(directory, mass) != json.loads(json.dumps(meta, sort_keys=True)):
            raise ValueError(f"{directory} m{mass}: stored point metadata differs from its re-derivation")

    def background(self, split, columns=None):
        key = (split, None if columns is None else tuple(columns))
        if key not in self._background:
            self._background = {key: pd.read_parquet(self.background_path(split), columns=columns)}
        return self._background[key]

    def split(self, directory, mass, split, columns=None):
        signal = pd.read_parquet(self.signal_path(directory, mass, split), columns=columns)
        return pd.concat([signal, self.background(split, columns)], ignore_index=True)

    def meta(self, directory, mass):
        return json.loads(self.meta_path(directory, mass).read_text())

    def totals(self, directory, mass):
        """One row per sample carrying its full prepared yield: all evaluation normalization needs."""
        rows = [dict(event_id=f"total:{sample}", sample=sample, sample_key=item["sample_key"], target=item["target"],
                     sample_weight=item["full_weight_sum"])
                for sample, item in sorted(self.meta(directory, mass)["totals"].items())]
        return pd.DataFrame(rows)

    def files(self, directory, mass):
        return [self.meta_path(directory, mass), *(self.signal_path(directory, mass, s) for s in SPLITS)]

    def background_files(self):
        return [self.background_path(split) for split in SPLITS]
