"""Phase 9 with truth tagging, as pre-registered in validation.yaml: truth_tag_rerun.

The grid's design (masses, couplings, training coupling, seeds, recipe, evaluation and
cut reference) on truth-tagged rows: one drawn passing tag outcome per event, weighted
by its pass probability, with splits from the truth-tagging closure's registry. Results
go to their own report, runs and datasets; the direct-tag grid stays unchanged. The
test partition stays sealed.

--merging-counts applies the matching acceptance (validation.yaml: matching_acceptance).
Its gates run before any row is written; rows keep only events accepted at the central
merging scale, weighted by xs / N_accepted. The per-event checks against the registry and
the closure still run on all events first, as without the acceptance.

--parton-root adds the heavy-flavour overlap removal (validation.yaml: flavour_overlap) on
top of the acceptance: every background event needs a parton record matching its sample's
card, and events that another sample also generates are dropped with their weights.

--resume continues an interrupted run of the same command: identical settings and
inputs, the stored rows re-derived and compared, completed fits re-checked and read
back, and incomplete fit directories set aside (renamed) and fitted again.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from hepml_compact.parquet_writer import sha256_file
from scipy.stats import chi2 as chi2_distribution

from hepml.adapters.configuration import default_study_directory, fit_parameters, load_analysis
from hepml.adapters.dataset_files import discover_samples
from hepml.adapters.grid_report import save_grid_figures
from hepml.adapters.study_loader import load_study
from hepml.adapters.study_run import (
    comparable_config,
    completed_report,
    finish_report,
    hash_files,
    open_report,
    refuse_existing,
    registry_files,
    reopen_report,
    run_study,
    write_json,
    write_tables,
)
from hepml.adapters.truth_tag_datasets import (
    SPLITS,
    TruthTagBenchmark,
    apply_acceptance,
    hard_process_counts,
    load_merging_counts,
    matching_acceptance,
    sampling_check,
    truth_tag_rows,
)
from hepml.commands.grid_study import fit_and_evaluate, run_complete, validate_settings
from hepml.domain.artifacts import model_dirname
from hepml.domain.flavour_overlap import REMOVALS, matches_card, removed
from hepml.domain.truth_tagging import TaggingModel

TAGGING = "truth"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study")
    parser.add_argument("--grid-report", required=True, help="Completed direct-tag Phase 9 report: the design it follows")
    parser.add_argument("--closure-report", required=True, help="Completed truth-tagging closure that adopted the method")
    parser.add_argument("--dataset-dir", required=True, help="New root for the truth-tagged rows")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--merging-counts",
                        help="hepml_compact.merging_counts JSON: apply the pre-registered matching acceptance")
    parser.add_argument("--parton-root",
                        help="Parton export root (extraction_partons.yaml): apply the heavy-flavour overlap removal")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run of this exact command (same settings and inputs)")
    args = parser.parse_args(argv)
    run_study(args.outdir, _run, args)


def validate_rerun(rerun, closure_report):
    check = rerun["sampling_check"]
    if (rerun["rows"] != "one_sampled_outcome_per_event" or rerun["fit_weight"] != "pass_probability"
            or rerun["grid"] != "grid_study" or type(rerun["outcome_seed"]) is not int
            or rerun["binned_fit"] != {"stage_a": "binned_fit_study", "stage_b": "binned_fit_stage_b",
                                       "jets": "reconstructed"}
            or check["split"] != "val" or check["bins"] < 2 or not 0 < check["min_p_value"] < 1
            or Path(closure_report).name != rerun["closure_report"]):
        raise ValueError("Unsupported truth-tagged rerun design, or a closure report other than the registered one")


def validate_acceptance(design):
    gates = design["gates"]
    if (design["counts"] != "hepml_compact.merging_counts" or design["accepted"] != "central_merging_weight_nonzero"
            or design["central_weight"] != "merging_weight_sharing_the_scale_variation_zeros"
            or design["normalization"] != "xs_pythia_over_accepted_count"
            or not all(gates[name] is True for name in ("counts_cover_every_export_source",
                                                        "stored_counts_match_export_entries",
                                                        "export_identification_matches_counts"))
            or not 0 < gates["max_abs_xs_ratio_offset"] < 1):
        raise ValueError("Unsupported matching-acceptance design")


def count_gates(key, manifest_path, meta, counted, tolerance):
    """Gates 1 and 3 of matching_acceptance for one sample, before any row is derived."""
    sources = {s["source_id"]: s["entries"] for s in json.loads(Path(manifest_path).read_text())["sources"]}
    files = {record["source_id"]: record for record in counted["files"]}
    covered = set(sources) == set(files) and all(files[s]["entries"] == n for s, n in sources.items())
    covered = covered and counted["n_stored"] == meta["n_events_total"]
    ratio = float(meta["xs_pb"]) / counted["implied_merged_xs"]
    same_rate = bool(np.isclose(float(counted["xs_pb"]), float(meta["xs_pb"]), rtol=1e-9, atol=0))
    row = dict(sample=key, n_stored=counted["n_stored"], n_accepted=counted["n_accepted"],
               n_generated=counted["n_generated"], accepted_fraction=counted["n_accepted"] / counted["n_stored"],
               central_columns=",".join(str(f["central_column"]) for f in counted["files"]),
               xs_pb=float(meta["xs_pb"]), implied_merged_xs=counted["implied_merged_xs"], xs_over_implied=ratio)
    checks = [dict(check=f"{key}: every ROOT file of the export is counted, with its stored events", passed=covered),
              dict(check=f"{key}: Pythia merged xs / the files' implied merged rate = {ratio:.5f} (within "
                         f"{tolerance:.0%}), from the manifest the counts read",
                   passed=bool(abs(ratio - 1) <= tolerance and same_rate))]
    return row, checks


def identification_gate(key, counted, identified, accepted):
    """Gate 2: the exported weights identify the counted central merging weight in every file."""
    files = {record["source_id"]: record for record in counted["files"]}
    same = all(files[s]["central_column"] == r["central_column"] and files[s]["n_generated"] == r["n_generated"]
               and r["accepted"] <= files[s]["accepted"] for s, r in identified.items())
    return dict(check=f"{key}: exported weights identify the counted central merging weight in every file; "
                      f"{int(accepted.sum()):,} of {len(accepted):,} exported events accepted", passed=bool(same))


def validate_overlap(design, backgrounds):
    if (design["partons"] != "extraction_partons" or type(design["outgoing_status"]) is not int
            or not set(design["remove"].values()) <= set(REMOVALS) or not set(design["remove"]) <= set(design["cards"])):
        raise ValueError("Unsupported flavour-overlap design")
    missing = sorted(meta["key"] for meta in backgrounds.values() if meta["key"] not in design["cards"])
    if missing:
        raise ValueError(f"No flavour-overlap card for {missing}")


def split_lookup(path, sample_key):
    table = pq.read_table(path, columns=["event_id", "split"], filters=[("sample_key", "=", sample_key)])
    return pd.Series(table["split"].to_numpy(zero_copy_only=False),
                     index=pd.Index(table["event_id"].to_numpy(zero_copy_only=False)))


def _run(args):
    study = load_study(args.study or default_study_directory())
    analysis = load_analysis(study.directory / "analysis.yaml")
    cfg = analysis.evaluation
    validation = yaml.safe_load((study.directory / "validation.yaml").read_text())
    rerun, settings = validation["truth_tag_rerun"], validation["grid_study"]
    cut_settings = validation["reference_study"]["cut_reference"]
    closure_report, grid_report = Path(args.closure_report).resolve(), Path(args.grid_report).resolve()
    validate_rerun(rerun, closure_report)
    directories = validate_settings(settings)
    masses, couplings, seeds, training = (settings["masses"], settings["couplings"], settings["seeds"],
                                          settings["training_coupling"])
    features = list(study.plugin.FEATURES)

    # The direct grid fixes the design; the closure fixes the method, its exports and splits.
    grid = completed_report(grid_report, [9])
    if grid.get("stage") is not None or grid.get("tagging", "direct") != "direct":
        raise ValueError("Require the completed direct-tag Phase 9 grid report")
    params = fit_parameters(analysis)
    if (grid["settings"] != settings or grid["cut_reference"] != cut_settings or grid["fit_parameters"] != params
            or comparable_config(grid["evaluation_config"]) != comparable_config(asdict(cfg))):
        raise ValueError("Grid design, cut reference, recipe or evaluation settings differ from the direct grid")
    for record in grid["runs"]:
        if json.loads((Path(record["model_directory"]) / "metrics.json").read_text())["features"] != features:
            raise ValueError("Features differ from the direct grid")
    stored_lumi = float(json.loads((grid_report / "provenance/preregistration.json").read_text())["stored_lumi_pb_inv"])
    closure = completed_report(closure_report, ["truth_tagging"])
    if not closure["adopted"] or closure["settings"] != validation["truth_tagging"]:
        raise ValueError("The closure did not adopt truth tagging, or its design has changed since")
    model = TaggingModel.from_settings(validation["truth_tagging"])
    truth_root, registry_dir = Path(closure["truth_tag_root"]), Path(closure["registry"])
    registry_path = registry_dir / "assignments.parquet"
    if sha256_file(registry_path) != json.loads((registry_dir / "registry.json").read_text())["assignments_sha256"]:
        raise ValueError("The truth-tag registry changed after the closure")
    seed = rerun["outcome_seed"]
    acceptance, counts = None, None
    if args.merging_counts:
        acceptance = validation["matching_acceptance"]
        validate_acceptance(acceptance)
        counts = load_merging_counts(args.merging_counts)
    overlap = None
    if args.parton_root:
        if acceptance is None:
            raise ValueError("The flavour-overlap removal is registered on top of the matching acceptance: "
                             "give --merging-counts too")
        overlap = validation["flavour_overlap"]

    out, dataset_root, runs = (Path(getattr(args, name)).resolve() for name in ("outdir", "dataset_dir", "runs_dir"))
    if not args.run_prefix or Path(args.run_prefix).name != args.run_prefix or any(x in args.run_prefix for x in "/\\:"):
        raise ValueError("run-prefix must be a directory name")
    run_ids = {(mass, s): f"{args.run_prefix}-m{mass}-seed{s}" for mass in masses for s in seeds}
    if args.resume:
        if not (out / "provenance/preregistration.json").is_file() or not dataset_root.is_dir():
            raise ValueError("--resume needs the interrupted run's report and datasets")
    else:
        refuse_existing(out, dataset_root, *(runs / name for name in run_ids.values()))

    _, backgrounds = discover_samples(truth_root / "backgrounds")
    samples = {}
    for rho in couplings:
        signals, _ = discover_samples(truth_root / directories[rho])
        for mass in masses:
            meta = signals.get(str(mass))
            if meta is None or not np.isclose(float(meta["rho_tc"]), rho):
                raise ValueError(f"No truth-tag signal for m{mass}, rho_tc={rho}")
            samples[mass, rho] = meta
    manifests = {key: truth_root / "backgrounds" / meta["export_manifest"] for key, meta in backgrounds.items()}
    parton_manifests = {}
    if overlap is not None:
        validate_overlap(overlap, backgrounds)
        parton_root = Path(args.parton_root).resolve()
        _, parton_backgrounds = discover_samples(parton_root / "backgrounds")
        for key, meta in backgrounds.items():
            if key not in parton_backgrounds:
                raise ValueError(f"No parton export for {meta['key']} under {parton_root}")
            parton_manifests[key] = parton_root / "backgrounds" / parton_backgrounds[key]["export_manifest"]
    manifests.update({point: truth_root / directories[point[1]] / meta["export_manifest"]
                      for point, meta in samples.items()})
    metadata = [p for path in manifests.values() for p in (path, path.with_name(path.name.replace(".export.json",
                                                                                                    ".meta.json")))]
    counts_files = [Path(args.merging_counts).resolve()] if counts is not None else []
    counts_files += [p for path in parton_manifests.values()
                     for p in (path, path.with_name(path.name.replace(".export.json", ".meta.json")))]
    inputs = hash_files(sorted([*registry_files(registry_dir), *metadata, *counts_files,
                                study.directory / "validation.yaml",
                                study.directory / "analysis.yaml",
                                *(report / "provenance" / name for report in (grid_report, closure_report)
                                  for name in ("summary.json", "status.json"))]))
    preregistration = dict(study=str(study.directory),
                           arguments={key: value for key, value in vars(args).items() if key != "resume"},
                           analysis=asdict(analysis), settings=settings, rerun=rerun, cut_reference=cut_settings,
                           fit_parameters=params, grid_report=str(grid_report), closure_report=str(closure_report),
                           stored_lumi_pb_inv=stored_lumi, test_evaluated=False)
    if acceptance is not None:
        preregistration["matching_acceptance"] = acceptance
    if overlap is not None:
        preregistration["flavour_overlap"] = overlap
    if args.resume:
        resume_record = reopen_report(out, preregistration, inputs, dict(command="truth-tag-grid-study"))
    else:
        open_report(out, preregistration, inputs, preregister=True)
    checks = [dict(check="direct Phase 9 grid complete and unchanged; same design, recipe and evaluation", passed=True),
              dict(check="truth-tagging closure complete, adopted and unchanged; registry hash as recorded", passed=True)]
    acceptance_rows = []
    if acceptance is not None:
        tolerance = acceptance["gates"]["max_abs_xs_ratio_offset"]
        for name, meta in [*backgrounds.items(), *samples.items()]:
            if meta["key"] not in counts:
                raise ValueError(f"{meta['key']}: no merging count")
            row, gate_checks = count_gates(meta["key"], manifests[name], meta, counts[meta["key"]], tolerance)
            acceptance_rows.append(row)
            checks += gate_checks
        if not all(check["passed"] for check in checks):
            write_tables(out, dict(checks=pd.DataFrame(checks), matching_acceptance=pd.DataFrame(acceptance_rows)))
            raise ValueError("Pre-registered matching-acceptance gates failed")

    def accepted_rows(key, manifest_path, rows):
        """All-event rows -> accepted rows (unchanged without --merging-counts)."""
        if acceptance is None:
            return rows
        flags, identified = matching_acceptance(manifest_path, counts[key])
        checks.append(identification_gate(key, counts[key], identified, flags))
        if not checks[-1]["passed"]:
            write_tables(out, dict(checks=pd.DataFrame(checks), matching_acceptance=pd.DataFrame(acceptance_rows)))
            raise ValueError("Pre-registered matching-acceptance gates failed")
        kept = apply_acceptance(rows, flags, counts[key]["n_accepted"])
        row = next(r for r in acceptance_rows if r["sample"] == key)
        row.update(exported_events=len(flags), exported_accepted=int(flags.sum()), rows_kept=len(kept))
        return kept

    overlap_rows = []

    def without_overlap(name, all_ids, rows):
        """Gates on the parton records of every truth-tag event, then the pre-registered removal."""
        if overlap is None:
            return rows
        key = backgrounds[name]["key"]
        partons = hard_process_counts(parton_manifests[name], overlap["outgoing_status"])
        covered = partons.index.is_unique and set(partons.index) == set(all_ids)
        card = matches_card(partons, overlap["cards"][key], overlap["outgoing_partons"])
        checks.append(dict(check=f"{key}: every one of {len(all_ids):,} truth-tag events has one parton record",
                           passed=bool(covered)))
        checks.append(dict(check=f"{key}: every event's outgoing hard-process partons match its card "
                                 f"({int(card.sum()):,} of {len(card):,})", passed=bool(card.all())))
        if not checks[-1]["passed"] or not checks[-2]["passed"]:
            write_tables(out, dict(checks=pd.DataFrame(checks)))
            raise ValueError("Pre-registered flavour-overlap gates failed")
        rule = overlap["remove"].get(key)
        drop = removed(partons, rule) if rule else pd.Series(False, index=partons.index)
        kept = rows[~drop.reindex(rows.event_id.to_numpy()).to_numpy(dtype=bool)].reset_index(drop=True)
        overlap_rows.append(dict(sample=key, rule=rule or "keep all", events=len(partons),
                                 events_removed=int(drop.sum()), rows_before=len(rows), rows_after=len(kept),
                                 yield_removed=1 - float(kept.sample_weight.sum() / rows.sample_weight.sum())))
        return kept

    # 1. Rows: one drawn outcome per event, the registry's split, weight x pass probability.
    bench = TruthTagBenchmark(dataset_root)
    registry_counts = pq.read_table(registry_path, columns=["sample_key", "split"]).to_pandas().value_counts()
    closure_yields = pd.read_csv(closure_report / "tables/yields.csv").set_index("sample")
    common = dict(stored_lumi=stored_lumi, model=model, plugin=study.plugin, seed=seed)
    parts, totals, row_checks = [], {}, []
    for key in sorted(backgrounds):
        split_of = split_lookup(registry_path, key)
        rows = truth_tag_rows(manifests[key], sample=f"bkg_{key}", split_of=split_of, **common)
        row_checks.append((key, rows.split.value_counts(), float(rows.loc[rows.split != "test", "fit_weight"].sum())))
        all_ids = rows.event_id.to_numpy()
        rows = without_overlap(key, all_ids, accepted_rows(backgrounds[key]["key"], manifests[key], rows))
        parts.append(rows)
        totals[f"bkg_{key}"] = dict(target=0, sample_key=key, full_weight_sum=float(rows.sample_weight.sum()))
        print(f"{key}: {len(rows):,} truth-tagged rows", flush=True)
    background = pd.concat(parts, ignore_index=True)
    if args.resume:
        bench.compare_background(background)
    else:
        bench.write_background(background)
    del parts

    # Every event once, in its registry split; the pass probabilities are the closure's expectations.
    # Both concern all events, before the matching acceptance.
    for key, split_counts, non_test_fit in row_checks:
        same = all(int(split_counts.get(s, 0)) == int(registry_counts.get((key, s), 0)) for s in SPLITS)
        expected = closure_yields.loc[key, "truth_tag"]
        closure_match = bool(np.isclose(non_test_fit, expected, rtol=1e-9))
        checks.append(dict(check=f"{key}: one row per truth-tag event in its registry split; non-test pass "
                                 f"probabilities sum to the closure's {expected:,.1f}", passed=bool(same and closure_match)))
    del row_checks

    sampling = rerun["sampling_check"]
    key = sampling["sample"]
    results = sampling_check(manifests[key], split_of=split_lookup(registry_path, key), split=sampling["split"],
                             model=model, plugin=study.plugin, seed=seed, bins=sampling["bins"])
    sampling_rows = [dict(sample=key, feature=feature, chi2=chi2, dof=dof,
                          p_value=float(chi2_distribution.sf(chi2, dof)) if dof else np.nan)
                     for feature, (chi2, dof) in results.items()]
    for row in sampling_rows:
        row["passed"] = bool(row["dof"] and row["p_value"] >= sampling["min_p_value"])
    checks.append(dict(check=f"sampling check: drawn rows match all outcomes on {key} {sampling['split']} "
                             f"(lowest p = {min(r['p_value'] for r in sampling_rows):.3g})",
                       passed=all(r["passed"] for r in sampling_rows)))
    if not checks[-1]["passed"]:
        write_tables(out, dict(checks=pd.DataFrame(checks), sampling_check=pd.DataFrame(sampling_rows)))
        raise ValueError("Pre-registered sampling check failed")

    rates = []
    for (mass, rho), meta in samples.items():
        key = meta["key"]
        rows = truth_tag_rows(manifests[mass, rho], sample=f"sig{mass}", split_of=split_lookup(registry_path, key),
                              **common)
        split_counts = rows.split.value_counts()
        checks.append(dict(check=f"{key}: one row per truth-tag event in its registry split", passed=all(
            int(split_counts.get(s, 0)) == int(registry_counts.get((key, s), 0)) for s in SPLITS)))
        rows = accepted_rows(key, manifests[mass, rho], rows)
        point_totals = {**totals, f"sig{mass}": dict(target=1, sample_key=key,
                                                     full_weight_sum=float(rows.sample_weight.sum()))}
        point_meta = dict(lumi=stored_lumi, features=features, tagging=TAGGING, rows=rerun["rows"], outcome_seed=seed,
                          signal={k: v for k, v in meta.items() if not k.startswith("_")}, totals=point_totals,
                          k_factors_applied_to_stored_weights=False)
        if acceptance is not None:
            point_meta["matching_acceptance"] = dict(n_accepted=counts[key]["n_accepted"],
                                                     backgrounds={m["key"]: counts[m["key"]]["n_accepted"]
                                                                  for m in backgrounds.values()})
        if args.resume:
            bench.compare_signal(directories[rho], mass, rows, point_meta)
        else:
            bench.write_signal(directories[rho], mass, rows, point_meta)
        n_denominator = counts[key]["n_accepted"] if acceptance is not None else meta["n_events_total"]
        rates.append(dict(mass=mass, rho_tc=rho, sample=key, xs_pb=meta["xs_pb"],
                          xs_source=json.dumps(meta.get("xs_source")), n_root=meta["n_events_total"],
                          n_accepted=n_denominator, n_selected=len(rows),
                          expected_direct_selected=float(rows.fit_weight.sum()),
                          selection_efficiency=float(rows.fit_weight.sum()) / n_denominator,
                          validation_signal_mc=int((rows.split == "val").sum()),
                          expected_signal_before_bdt=point_totals[f"sig{mass}"]["full_weight_sum"]
                          * cfg.lumi_pb_inv / stored_lumi * cfg.signal_k_factor))
    print(f"Truth-tagged rows {'re-derived and identical to the stored ones' if args.resume else 'written'} "
          f"for {len(samples)} points", flush=True)
    if args.resume:
        checks.append(dict(check="resume: every stored row and point metadata equals its re-derivation", passed=True))

    # On resume, completed fits are re-checked and read back; incomplete ones are set aside and refitted.
    reused, resumed = frozenset(), None
    if args.resume:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        reused = frozenset(point for point, run_id in run_ids.items()
                           if run_complete(runs / run_id / "models" / model_dirname(point[0]), couplings, directories))
        set_aside = []
        for point, run_id in sorted(run_ids.items()):
            if point not in reused and (runs / run_id).exists():
                target = runs / f"{run_id}.incomplete-{stamp}"
                (runs / run_id).rename(target)
                set_aside.append(str(target))
        resumed = dict(json.loads(resume_record.read_text()),
                       reused_runs=sorted(run_ids[point] for point in reused),
                       refitted_runs=sorted(run_ids[point] for point in run_ids if point not in reused),
                       incomplete_runs_set_aside=set_aside)
        write_json(resume_record, resumed)
        print(f"Resuming: {len(reused)} completed fits reused, {len(run_ids) - len(reused)} to fit", flush=True)

    # 2. The grid's fits, point evaluations and cut reference, on these rows.
    bootstrap = dict(replicates=settings["bootstrap_replicates"], seed=settings["bootstrap_seed"])
    grid_rows, auc_rows, support_rows, runs_record, cut_rows = fit_and_evaluate(
        masses=masses, couplings=couplings, seeds=seeds, training=training, directories=directories,
        features=features, params=params, cfg=cfg, stored_lumi=stored_lumi, bootstrap=bootstrap, runs=runs,
        run_ids=run_ids, fit_policy=rerun["fit_weight"], cut_settings=cut_settings, checks=checks,
        sample_keys={point: meta["key"] for point, meta in samples.items()},
        training_data=lambda mass: (bench.split(directories[training], mass, "train"),
                                    bench.split(directories[training], mass, "val")),
        validation_data=lambda mass, rho: bench.split(directories[rho], mass, "val"),
        totals=lambda mass, rho: bench.totals(directories[rho], mass), fit_weights="fit_weight", reuse=reused)

    written = [p for name in run_ids.values() for p in (runs / name).rglob("*preds_test*")]
    checks.append(dict(check="no_test_scores_read_or_written", passed=not written))
    registry_table = (registry_counts.rename("events").reset_index()
                      .pivot(index="sample_key", columns="split", values="events").fillna(0).astype(int).reset_index())
    tables = dict(checks=pd.DataFrame(checks), rates=pd.DataFrame(rates), registry_extension=registry_table,
                  grid=pd.DataFrame(grid_rows + cut_rows), auc=pd.DataFrame(auc_rows),
                  sampling_check=pd.DataFrame(sampling_rows),
                  **({"matching_acceptance": pd.DataFrame(acceptance_rows)} if acceptance is not None else {}),
                  **({"flavour_overlap": pd.DataFrame(overlap_rows)} if overlap is not None else {}),
                  process_support=pd.concat(support_rows, ignore_index=True) if support_rows
                  else pd.DataFrame(columns=["mass", "rho_tc", "seed", "objective"]))
    write_tables(out, tables)
    if not tables["checks"].passed.all():
        raise ValueError("Truth-tagged grid contract checks failed")
    frozen = {f"m{r['mass']}": str(Path(r["model_directory"]) / "points") for r in runs_record
              if r["seed"] == settings["primary_seed"]}
    summary = dict(schema_version=2, phases=[9], tagging=TAGGING, settings=settings, rerun=rerun,
                   cut_reference=cut_settings, evaluation_config=asdict(cfg), fit_parameters=params, runs=runs_record,
                   registry=str(registry_dir), dataset=str(dataset_root), dataset_layout="truth_tag_shared_background_v1",
                   reference_report=grid["reference_report"], direct_grid_report=str(grid_report),
                   closure_report=str(closure_report), primary_seed=settings["primary_seed"],
                   frozen_threshold_directories=frozen, resumed=resumed, test_evaluated=False,
                   matching_acceptance=(dict(acceptance, counts=str(counts_files[0])) if acceptance is not None
                                        else None),
                   flavour_overlap=(dict(overlap, parton_root=str(Path(args.parton_root).resolve()))
                                    if overlap is not None else None),
                   status="checkpoint_ready",
                   note="Truth-tagged validation grid, separate from the direct-tag grid. Thresholds of the primary "
                        "seed are frozen per point for one later test evaluation; alternative hypotheses are never "
                        "combined.")
    write_json(out / "provenance/summary.json", summary)
    save_grid_figures(out)
    artifacts = [p for r in runs_record for p in (runs / r["run_id"]).rglob("*") if p.is_file()]
    artifacts += [p for p in dataset_root.rglob("*") if p.is_file()]
    finish_report(out, "04_physics_results.ipynb", inputs, sorted(set(artifacts)))
