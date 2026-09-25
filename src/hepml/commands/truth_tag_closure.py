"""Truth-tagging closure on simulation, as pre-registered in validation.yaml: truth_tagging.

Reads the truth-tag export (events with at least three jets in acceptance, no tag
requirement) beside the direct-tag export of the same ROOT files. Gates, in the
pre-registered order: exact export consistency, tag rates per flavour and pT,
yields per background, and bbc feature shapes. Test-split events are excluded
throughout and no model score is computed. The event registry extended to the
truth-tag events is written for later studies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from hepml_compact.config import load_profile
from hepml_compact.parquet_writer import sha256_file, validate_export
from scipy.stats import chi2 as chi2_distribution

from hepml.adapters.configuration import default_study_directory
from hepml.adapters.study_loader import load_study
from hepml.adapters.study_run import (
    finish_report,
    hash_files,
    load_registry,
    open_report,
    refuse_existing,
    registry_files,
    run_study,
    write_json,
    write_tables,
)
from hepml.adapters.truth_tag_report import save_truth_tag_figures
from hepml.domain.splits import extend_sample_membership
from hepml.domain.truth_tagging import (
    FLAVOUR_CLASSES,
    STATES,
    TaggingModel,
    direct_roles,
    enumerate_outcomes,
    state_probabilities,
)

STAGE = "truth_tag_closure"
KINEMATICS = ("pt", "eta", "phi", "mass")
ROLES = ("b1", "b2", "c1")
TAGGED_STATES = (1, 2, 3)  # b_only, c_only, both; "none" is fixed by the jet count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study")
    parser.add_argument("--truth-tag-root", required=True, help="Truth-tag compact exports (cg_bbc_truth_tag)")
    parser.add_argument("--direct-root", required=True, help="Direct-tag compact exports of the same ROOT files")
    parser.add_argument("--registry-dir", required=True, help="Frozen event registry to extend (e.g. Phase 9)")
    parser.add_argument("--registry-out", required=True, help="New directory for the extended registry")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    run_study(args.outdir, _run, args)


def validate_design(settings, direct_config, model):
    """The pre-registered design, and the direct selection it must reproduce."""
    expected = {"b_jets": (model.b_bit, 2, ["b1", "b2"]), "c_jets": (model.c_bit, 1, ["c1"])}
    selection = direct_config["selection"]
    if set(selection) != set(expected):
        raise ValueError("The direct selection is not the two-b plus one-c selection truth tagging reproduces")
    for group, (bit, count, roles) in expected.items():
        spec = selection[group]
        if (spec["collection"] != "jets" or spec["count"] != {"exact": count} or spec["sort_by"] != "pt"
                or spec["roles"] != roles or spec["filters"] != {"pt": {"gt": model.pt_min},
                                                                  "eta": {"abs_lt": model.abs_eta_max},
                                                                  "btag": {"eq": bit}}):
            raise ValueError(f"{group}: extraction.yaml differs from the truth-tagging acceptance and bits")
    closure = settings["closure"]
    if (settings["flavour"]["column"] != "jets_truth_flavor"
            or settings["roles"] != "b1_b2_by_descending_pt_c1_the_c_only_jet"
            or settings["splits"] != "by_event_registry_extension"
            or settings["mc_variance"] != "sum_entries_per_event_then_square"
            or closure["export_consistency"] != "exact_event_ids_and_jets_every_sample"
            or closure["shapes"]["bins"] < 2 or not 0 < closure["shapes"]["min_p_value"] < 1
            or not 0 < closure["tag_rates"]["min_p_value"] < 1 or closure["yield_max_abs_pull"] <= 0
            or sorted(closure["tag_rates"]["pt_edges"]) != closure["tag_rates"]["pt_edges"]
            or closure["tag_rates"]["pt_edges"][0] != model.pt_min):
        raise ValueError("Unsupported truth-tagging design")


def pair_exports(truth_root, direct_root):
    truth = {p.relative_to(truth_root).as_posix(): p for p in truth_root.glob("*/*.export.json")}
    direct = {p.relative_to(direct_root).as_posix(): p for p in direct_root.glob("*/*.export.json")}
    if not truth or set(truth) != set(direct):
        raise ValueError("Truth-tag and direct exports must hold the same samples")
    return [dict(name=name, truth=truth[name], direct=direct[name]) for name in sorted(truth)]


def shards(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    return [manifest_path.parent / chunk["file"] for chunk in manifest["chunks"] if chunk["file"]]


def flat_jets(table, fields):
    """Offsets and flat numpy arrays of jet list columns that share one layout."""
    first = table[f"jets_{fields[0]}"].combine_chunks()
    offsets = first.offsets.to_numpy().astype(np.int64)
    offsets = offsets - offsets[0]
    values = {}
    for field in fields:
        column = table[f"jets_{field}"].combine_chunks()
        if not np.array_equal(column.offsets.to_numpy() - column.offsets.to_numpy()[0], offsets):
            raise ValueError(f"jets_{field} is not aligned with jets_{fields[0]}")
        values[field] = column.flatten().to_numpy(zero_copy_only=False)
    return offsets, values


def role_objects(offsets, jets, events, positions):
    """{"b1": (pt, eta, phi, mass), ...} of role jets given as positions within their events."""
    return {role: tuple(jets[field][offsets[events] + positions[role]] for field in KINEMATICS) for role in ROLES}


def consistency(pair, split_of, model):
    """Gate 1 for one sample: the direct selection re-applied to the truth-tag export, non-test events."""
    direct_manifest = json.loads(pair["direct"].read_text())
    jet_columns = sorted(c for c in direct_manifest["columns"] if c.startswith("jets_"))
    truth_columns = json.loads(pair["truth"].read_text())["columns"]
    if not set(jet_columns) <= set(truth_columns):
        raise ValueError(f"{pair['name']}: truth-tag export lacks jet columns of the direct export")
    direct = pa.concat_tables([pq.read_table(p, columns=["event_id", *jet_columns, "role_b1", "role_b2", "role_c1"])
                               for p in shards(pair["direct"])])
    direct_split = split_of.reindex(direct["event_id"].to_numpy(zero_copy_only=False)).to_numpy()
    unregistered_direct = int(pd.isna(direct_split).sum())
    direct = direct.filter(pa.array(direct_split != "test"))
    reproduced = []
    for path in shards(pair["truth"]):
        table = pq.read_table(path, columns=["event_id", *jet_columns])
        offsets, jets = flat_jets(table, ["pt", "eta", "btag"])
        passes, roles = direct_roles(offsets, jets["pt"], jets["eta"], jets["btag"], model)
        split = split_of.reindex(table["event_id"].to_numpy(zero_copy_only=False)).to_numpy()
        keep = passes & (split != "test")  # unregistered passing events stay and count as extra
        chosen = table.filter(pa.array(keep))
        for role in ROLES:
            chosen = chosen.append_column(f"role_{role}", pa.array(roles[role][keep]))
        reproduced.append(chosen)
    truth = pa.concat_tables(reproduced)
    t_ids, d_ids = truth["event_id"].to_pylist(), direct["event_id"].to_pylist()
    t_index = {event: i for i, event in enumerate(t_ids)}
    common = [(t_index[event], i) for i, event in enumerate(d_ids) if event in t_index]
    t_take, d_take = (pa.array([pair_[k] for pair_ in common], type=pa.int64()) for k in (0, 1))
    mismatched = [column for column in jet_columns
                  if not truth[column].take(t_take).combine_chunks().equals(direct[column].take(d_take).combine_chunks())]
    role_mismatch = sum(int(np.sum(truth[f"role_{r}"].take(t_take).to_numpy() != direct[f"role_{r}"].take(d_take).to_numpy()))
                        for r in ROLES)
    return dict(sample=pair["name"], direct_events=len(d_ids), reproduced=len(common),
                missing=len(d_ids) - len(common), extra=len(t_ids) - len(common),
                unregistered_direct=unregistered_direct, jet_columns_compared=len(jet_columns),
                mismatched_columns=";".join(mismatched), role_mismatches=role_mismatch,
                passed=bool(len(common) == len(d_ids) == len(t_ids) and not mismatched and not unregistered_direct))


def extend_registry_to_exports(pairs, base, seed, registry_out):
    """Write the registry with every truth-tag event; registered assignments never move."""
    registry_out.mkdir(parents=True)
    path = registry_out / "assignments.parquet"
    schema = pa.schema([(name, pa.string()) for name in ("event_id", "sample_key", "split", "assignment_method")])
    by_sample = {key: group for key, group in base.groupby("sample_key", sort=False)}
    exported = {}
    for pair in pairs:
        key = json.loads(pair["truth"].read_text())["sample"]
        ids = [x for p in shards(pair["truth"]) for x in pq.read_table(p, columns=["event_id"])["event_id"].to_pylist()]
        exported[key] = ids
    rows, splits = [], {}
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for key in sorted(set(by_sample) | set(exported)):
            previous = by_sample.get(key, base.iloc[:0])
            if key in exported:
                extended = extend_sample_membership(previous, exported[key], key, seed=seed)
            else:
                extended = previous.sort_values("event_id")
            writer.write_table(pa.Table.from_pandas(extended.astype(str), schema=schema, preserve_index=False))
            counts = extended.split.value_counts()
            kept = previous.merge(extended, on=["event_id", "sample_key", "split", "assignment_method"])
            rows.append(dict(sample_key=key, registered=len(previous), frozen_rows_kept=len(kept),
                             exported=len(exported.get(key, [])), added=len(extended) - len(previous),
                             total=len(extended), **{name: int(counts.get(name, 0)) for name in ("train", "val", "test")}))
            if key in exported:
                splits[key] = pd.Series(extended.split.to_numpy(), index=pd.Index(extended.event_id.to_numpy()))
    write_json(registry_out / "registry.json", dict(
        status="complete", method="membership_extension_v1", seed=seed, assignments_sha256=sha256_file(path),
        rule="registered events keep their split; new events of a sample are sha256-ranked 80/10/10 among themselves",
        added_samples=sorted(set(exported) - set(by_sample))))
    return pd.DataFrame(rows), splits


class ShapeAccumulator:
    """Per-bin direct counts, truth-tag expectations and the per-event multinomial covariance."""

    def __init__(self, edges):
        self.edges = edges
        self.bins = len(edges) + 1
        self.direct = np.zeros(self.bins)
        self.expected = np.zeros(self.bins)
        self.second = np.zeros((self.bins, self.bins))

    def add(self, values, event, probability, direct, n_events):
        bins = np.searchsorted(self.edges, values, side="right")
        per_event = np.bincount(event * self.bins + bins, weights=probability,
                                minlength=n_events * self.bins).reshape(n_events, self.bins)
        self.expected += per_event.sum(axis=0)
        self.second += per_event.T @ per_event
        self.direct += np.bincount(bins[direct], minlength=self.bins)

    def result(self):
        difference = self.direct - self.expected
        covariance = np.diag(self.expected) - self.second
        chi2 = float(difference @ np.linalg.solve(covariance, difference))
        return chi2, self.bins, float(chi2_distribution.sf(chi2, self.bins)), np.sqrt(np.diag(covariance))


class TagRateAccumulator:
    """Tag states of acceptance jets per flavour class and pT bin, against the per-jet multinomial prediction."""

    def __init__(self, model, edges, min_expected):
        self.model, self.edges, self.min_expected = model, np.asarray(edges, dtype=float), min_expected
        cells = len(FLAVOUR_CLASSES) * len(self.edges)
        self.jets = np.zeros(cells)
        self.observed = np.zeros((cells, 4))
        self.expected = np.zeros((cells, 4))
        self.second = np.zeros((cells, 4, 4))
        self.other_bits = 0

    def add(self, pt, flavour, btag):
        """Jets already inside the tagging acceptance."""
        model, cells = self.model, len(self.jets)
        self.other_bits += int(np.sum((btag & ~(model.b_bit | model.c_bit)) != 0))
        cell = model.flavour_class(flavour) * len(self.edges) + np.searchsorted(self.edges, pt, side="right") - 1
        probabilities = state_probabilities(*model.efficiencies(flavour, pt))
        self.jets += np.bincount(cell, minlength=cells)
        self.observed += np.bincount(cell * 4 + model.observed_states(btag), minlength=cells * 4).reshape(cells, 4)
        for s in range(4):
            self.expected[:, s] += np.bincount(cell, weights=probabilities[:, s], minlength=cells)
            for u in range(4):
                self.second[:, s, u] += np.bincount(cell, weights=probabilities[:, s] * probabilities[:, u],
                                                    minlength=cells)

    def results(self, min_p_value):
        """Per-cell rows and, per flavour class, chi2 = D^T C^-1 D summed over pT bins."""
        rows, tests = [], []
        n_pt = len(self.edges)
        for c, name in enumerate(FLAVOUR_CLASSES):
            total, dof = 0.0, 0
            for b in range(n_pt):
                cell = c * n_pt + b
                used = [s for s in TAGGED_STATES if self.expected[cell, s] >= self.min_expected]
                if used:
                    difference = self.observed[cell, used] - self.expected[cell, used]
                    covariance = np.diag(self.expected[cell, used]) - self.second[cell][np.ix_(used, used)]
                    total += float(difference @ np.linalg.solve(covariance, difference))
                    dof += len(used)
                for s in TAGGED_STATES:
                    variance = self.expected[cell, s] - self.second[cell, s, s]
                    rows.append(dict(flavour=name, pt_low=self.edges[b],
                                     pt_high=self.edges[b + 1] if b + 1 < n_pt else np.inf,
                                     jets=int(self.jets[cell]), state=STATES[s], observed=int(self.observed[cell, s]),
                                     expected=self.expected[cell, s],
                                     pull=(self.observed[cell, s] - self.expected[cell, s]) / np.sqrt(variance)
                                     if variance > 0 else np.nan, used=s in used))
            p_value = float(chi2_distribution.sf(total, dof)) if dof else np.nan
            tests.append(dict(flavour=name, chi2=total, dof=dof, p_value=p_value,
                              passed=bool(dof and p_value >= min_p_value)))
        return pd.DataFrame(rows), pd.DataFrame(tests)


def _run(args):
    study = load_study(args.study or default_study_directory())
    settings = yaml.safe_load((study.directory / "validation.yaml").read_text())["truth_tagging"]
    closure = settings["closure"]
    model = TaggingModel.from_settings(settings)
    recipe = load_profile(study.directory / settings["recipe"])
    validate_design(settings, load_profile(study.directory).config, model)
    out, registry_out = Path(args.outdir).resolve(), Path(args.registry_out).resolve()
    refuse_existing(out, registry_out)
    truth_root, direct_root = Path(args.truth_tag_root).resolve(), Path(args.direct_root).resolve()
    registry_dir = Path(args.registry_dir).resolve()
    pairs = pair_exports(truth_root, direct_root)
    metadata = [p for pair in pairs for m in (pair["truth"], pair["direct"])
                for p in (m, m.with_name(m.name.replace(".export.json", ".meta.json")))]
    inputs = hash_files(sorted([*registry_files(registry_dir), *metadata, study.directory / "validation.yaml",
                                study.directory / "extraction.yaml", study.directory / settings["recipe"]]))
    open_report(out, dict(study=str(study.directory), arguments=vars(args), settings=settings, test_evaluated=False),
                inputs, preregister=True)

    # Contract: both exports are complete, intact, and read the same ROOT files with the recorded recipes.
    checks = []
    for pair in pairs:
        truth, direct = (json.loads(pair[side].read_text()) for side in ("truth", "direct"))
        validate_export(pair["truth"].parent, truth)
        validate_export(pair["direct"].parent, direct)
        same = (truth["status"] == direct["status"] == "complete" and truth["study_config"] == recipe.config
                and [s["source_id"] for s in truth["sources"]] == [s["source_id"] for s in direct["sources"]]
                and truth["cutflow"]["input"] == direct["cutflow"]["input"]
                and truth["sample_metadata"] == direct["sample_metadata"] and truth["extractor"] == direct["extractor"])
        checks.append(dict(check=f"{pair['name']}: exports intact, same ROOT files, rates and compactor code; "
                                 "recipe as in the repository", passed=bool(same)))
    print(f"{len(pairs)} export pairs validated", flush=True)

    # Gate 1: exact export consistency, with the frozen registry's splits.
    base = load_registry(registry_dir)
    seed = int(json.loads((registry_dir / "registry.json").read_text())["seed"])
    base_split = pd.Series(base.split.to_numpy(), index=pd.Index(base.event_id.to_numpy()))
    consistency_rows = [consistency(pair, base_split, model) for pair in pairs]
    consistency_table = pd.DataFrame(consistency_rows)
    gate_consistency = bool(consistency_table.passed.all())
    print(f"Gate 1 (export consistency): {'pass' if gate_consistency else 'FAIL'}", flush=True)
    if not gate_consistency:
        write_tables(out, dict(checks=pd.DataFrame(checks), consistency=consistency_table))
        raise ValueError("Gate 1 failed: the truth-tag export does not reproduce the direct export")

    # Registry extended to every truth-tag event (existing assignments fixed).
    extension, splits = extend_registry_to_exports(pairs, base, seed, registry_out)
    exported = extension.exported > 0
    checks.append(dict(check="extended registry keeps every frozen assignment and registers every truth-tag event",
                       passed=bool((extension.frozen_rows_kept == extension.registered).all()
                                   and (extension.total[exported] == extension.exported[exported]).all()
                                   and exported.sum() == len(pairs))))
    print(f"Registry extended to {int(extension.total.sum()):,} events", flush=True)

    # Gates 2-4 on non-test events of the backgrounds; yields of the signals are descriptive.
    rates = TagRateAccumulator(model, closure["tag_rates"]["pt_edges"], closure["tag_rates"]["min_expected"])
    yields, shapes, shape_bins = [], [], []
    features_by_sample = {}
    plugin = study.plugin
    for pair in pairs:
        manifest = json.loads(pair["truth"].read_text())
        key, kind = manifest["sample"], manifest["kind"]
        split = splits[key]
        background = kind == "background"
        direct_features, direct_count, pass_sum, pass_var, pass_sq, events = [], 0, 0.0, 0.0, 0.0, 0
        columns = ["event_id", *(f"jets_{f}" for f in (*KINEMATICS, "btag", "truth_flavor"))]
        tables = []
        for path in shards(pair["truth"]):
            table = pq.read_table(path, columns=columns)
            keep = split.reindex(table["event_id"].to_numpy(zero_copy_only=False)).to_numpy() != "test"
            table = table.filter(pa.array(keep))
            tables.append(table)
            offsets, jets = flat_jets(table, [*KINEMATICS, "btag", "truth_flavor"])
            if background:
                accepted = model.in_acceptance(jets["pt"], jets["eta"])
                rates.add(jets["pt"][accepted], jets["truth_flavor"][accepted], jets["btag"][accepted])
                passes, roles = direct_roles(offsets, jets["pt"], jets["eta"], jets["btag"], model)
                chosen = np.flatnonzero(passes)
                direct_features.append(plugin.features_from_objects(
                    role_objects(offsets, jets, chosen, {r: roles[r][chosen] for r in ROLES})))
        features_by_sample[key] = pd.concat(direct_features, ignore_index=True) if direct_features else None
        accumulators = None
        if background and features_by_sample[key] is not None and len(features_by_sample[key]) >= closure["shapes"]["bins"]:
            quantiles = np.linspace(0, 1, closure["shapes"]["bins"] + 1)[1:-1]
            accumulators = {feature: ShapeAccumulator(np.unique(np.quantile(values, quantiles)))
                            for feature, values in features_by_sample[key].items()}
        for table in tables:
            offsets, jets = flat_jets(table, [*KINEMATICS, "btag", "truth_flavor"])
            entries, per_event = enumerate_outcomes(offsets, jets["pt"], jets["eta"], jets["truth_flavor"],
                                                    jets["btag"], model)
            probability = per_event["pass_probability"]
            events += len(probability)
            direct_count += int(per_event["direct_pass"].sum())
            pass_sum += probability.sum()
            pass_var += (probability * (1 - probability)).sum()
            pass_sq += (probability**2).sum()
            if int(entries["direct"].sum()) != int(per_event["direct_pass"].sum()):
                raise ValueError(f"{key}: enumeration lost a direct-tag outcome")
            if accumulators:
                objects = {role: tuple(jets[field][entries[role]] for field in KINEMATICS) for role in ROLES}
                features = plugin.features_from_objects(objects)
                for feature, accumulator in accumulators.items():
                    accumulator.add(features[feature].to_numpy(dtype=float), entries["event"],
                                    entries["probability"], entries["direct"], len(probability))
        pull = (direct_count - pass_sum) / np.sqrt(pass_var)
        neff = pass_sum**2 / pass_sq
        yields.append(dict(sample=key, kind=kind, events=events, direct=direct_count, truth_tag=pass_sum,
                           pull=pull, neff_truth_tag=neff, gain=neff / pass_sum, gated=background,
                           passed=bool(abs(pull) <= closure["yield_max_abs_pull"])))
        if accumulators:
            gated = key == closure["shapes"]["primary_sample"]
            for feature, accumulator in accumulators.items():
                chi2, dof, p_value, sigma = accumulator.result()
                shapes.append(dict(sample=key, feature=feature, chi2=chi2, dof=dof, p_value=p_value, gated=gated,
                                   passed=bool(p_value >= closure["shapes"]["min_p_value"])))
                bounds = np.concatenate([[-np.inf], accumulator.edges, [np.inf]])
                for b in range(accumulator.bins):
                    shape_bins.append(dict(sample=key, feature=feature, bin=b, low=bounds[b], high=bounds[b + 1],
                                           direct=accumulator.direct[b], truth_tag=accumulator.expected[b],
                                           sigma=sigma[b], pull=(accumulator.direct[b] - accumulator.expected[b]) / sigma[b]))
        print(f"{key}: {events:,} non-test events, direct {direct_count:,}, truth-tag {pass_sum:,.1f} "
              f"(pull {pull:+.2f}), effective-MC gain {neff / pass_sum:.1f}", flush=True)

    # Gate 2: tag rates per flavour class, summed over pT bins.
    rate_rows, rate_tests = rates.results(closure["tag_rates"]["min_p_value"])
    checks.append(dict(check=f"BTag words hold only the b and c bits ({rates.other_bits} jets with other bits)",
                       passed=rates.other_bits == 0))

    tables = dict(checks=pd.DataFrame(checks), consistency=consistency_table, registry_extension=extension,
                  tag_rates=rate_rows, tag_rate_tests=rate_tests,
                  yields=pd.DataFrame(yields), shapes=pd.DataFrame(shapes), shape_bins=pd.DataFrame(shape_bins))
    gates = {
        "export_consistency": gate_consistency,
        "tag_rates": bool(tables["tag_rate_tests"].passed.all()),
        "yields": bool(tables["yields"].loc[tables["yields"].gated, "passed"].all()),
        "shapes": bool(tables["shapes"].loc[tables["shapes"].gated, "passed"].all()),
    }
    tables["gates"] = pd.DataFrame([dict(gate=name, passed=value) for name, value in gates.items()])
    write_tables(out, tables)
    if not tables["checks"].passed.all():
        raise ValueError("Truth-tagging contract checks failed")
    adopted = all(gates.values())
    background_yields = tables["yields"][tables["yields"].gated]
    summary = dict(schema_version=2, phases=["truth_tagging"], stage=STAGE, settings=settings,
                   gates=gates, adopted=adopted, registry=str(registry_out), base_registry=str(registry_dir),
                   truth_tag_root=str(truth_root), direct_root=str(direct_root),
                   gains={row.sample: row.gain for row in background_yields.itertuples()},
                   test_evaluated=False, status="checkpoint_ready",
                   note=("All pre-registered gates pass: truth tagging replaces direct tagging for every sample in "
                         "later cg_bbc studies, under new run IDs." if adopted else
                         "A pre-registered gate failed: truth tagging is not used until the cause is understood "
                         "and a new design is recorded."))
    write_json(out / "provenance/summary.json", summary)
    save_truth_tag_figures(out)
    finish_report(out, "05_truth_tagging.ipynb", inputs, artifacts=registry_files(registry_out))
    print("Gates: " + ", ".join(f"{name} {'pass' if value else 'FAIL'}" for name, value in gates.items())
          + f". Truth tagging {'adopted' if adopted else 'not adopted'}.", flush=True)
