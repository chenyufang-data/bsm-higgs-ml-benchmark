"""Phase 9: mass-specific BDTs trained at one coupling and evaluated at every (mass, rho_tc) point.

Every point is scored on its own genuine signal sample with its own cross section,
and its operating points come from that point's validation events. The test
partition stays sealed for a separate final evaluation of the frozen thresholds.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
from hepml_compact.contracts import part_glob_pattern
from hepml_compact.parquet_writer import sha256_file

from hepml.adapters.configuration import default_study_directory, fit_parameters, load_analysis
from hepml.adapters.dataset_files import discover_samples, load_sample_frame, prepare_registered_reference
from hepml.adapters.grid_report import save_grid_figures
from hepml.adapters.inference import load_model
from hepml.adapters.study_loader import load_study
from hepml.adapters.study_run import (
    REPOSITORY,
    comparable_config,
    completed_report,
    finish_report,
    hash_files,
    load_registry,
    open_report,
    refuse_existing,
    registry_files,
    run_study,
    write_json,
    write_tables,
    write_validation,
)
from hepml.adapters.xgboost_model import train_xgb
from hepml.application.baselines import cut_reference
from hepml.application.evaluation import (
    check_yield_closure,
    evaluation_weights,
    point_rows,
    roc_tables,
    selected_support,
    validate_model,
)
from hepml.application.training import make_xyw
from hepml.domain.artifacts import BenchmarkFiles, model_dirname
from hepml.domain.metrics import OBJECTIVES, safe_auc
from hepml.domain.splits import extend_registry

IDENTITY = ["event_id", "sample_key", "sample", "target", "sample_weight"]
# Files a completed fit writes, in its model directory and in each point directory.
RUN_FILES = ["model.ubj", "metrics.json", "operating_points.json", "learning_curves.json", "roc.parquet",
             "preds_train.parquet", "preds_val.parquet", "threshold_curves.parquet", "threshold_bands.parquet",
             "process_curves.parquet", "threshold_stability.csv", "frozen_train.csv", "frozen_val.csv"]
POINT_FILES = ["operating_points.json", "roc.parquet", "preds_val.parquet", "threshold_curves.parquet",
               "threshold_bands.parquet", "process_curves.parquet", "threshold_stability.csv", "frozen_val.csv"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study")
    parser.add_argument("--reference-report", required=True,
                        help="Completed Phase 2 report: anchor, registry, stored luminosity and validated recipe")
    parser.add_argument("--compact-root", required=True, help="Folder holding one compact directory per coupling")
    parser.add_argument("--registry-dir", required=True, help="New registry: the Phase 2 registry plus every grid sample")
    parser.add_argument("--dataset-dir", required=True, help="New root; one prepared benchmark per coupling below it")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    run_study(args.outdir, _run, args)


def validate_settings(settings):
    masses, couplings, seeds = settings["masses"], settings["couplings"], settings["seeds"]
    directories = {float(key): value for key, value in settings["coupling_directories"].items()}
    if (not masses or len(set(masses)) != len(masses) or not couplings or len(set(couplings)) != len(couplings)
            or set(directories) != set(couplings) or len(set(directories.values())) != len(directories)
            or settings["training_coupling"] not in couplings or settings["fit_policy"] != "none"
            or not seeds or len(set(seeds)) != len(seeds) or settings["primary_seed"] not in seeds
            or settings["bootstrap_replicates"] < 2):
        raise ValueError("Unsupported grid-study design")
    return directories


def _run(args):
    study = load_study(args.study or default_study_directory())
    analysis = load_analysis(study.directory / "analysis.yaml")
    cfg = analysis.evaluation
    validation = yaml.safe_load((study.directory / "validation.yaml").read_text())
    settings, cut_settings = validation["grid_study"], validation["reference_study"]["cut_reference"]
    directories = validate_settings(settings)
    masses, couplings, seeds = settings["masses"], settings["couplings"], settings["seeds"]
    training = settings["training_coupling"]
    features = list(study.plugin.FEATURES)
    reference_report = Path(args.reference_report).resolve()
    phase2 = completed_report(reference_report, [2])
    if comparable_config(phase2["evaluation_config"]) != comparable_config(asdict(cfg)):
        raise ValueError("Evaluation settings (luminosity, k-factors, support rules) differ from Phase 2")
    params = fit_parameters(analysis)
    for record in phase2["runs"]:
        metrics = json.loads((Path(record["model_directory"]) / "metrics.json").read_text())
        if any(metrics["xgb"]["params"].get(key) != value for key, value in params.items()) or metrics["features"] != features:
            raise ValueError("Features or hyperparameters differ from the validated Phase 2 recipe")
    phase2_args = json.loads((reference_report / "provenance/preregistration.json").read_text())["arguments"]
    anchor = Path(phase2_args["anchor_dataset"])
    anchor = (anchor if anchor.is_absolute() else REPOSITORY / anchor).resolve()
    anchor_mass, anchor_files = phase2_args["anchor_mass"], BenchmarkFiles(anchor)
    stored_lumi = float(json.loads(anchor_files.meta(anchor_mass).read_text())["lumi"])

    compact = Path(args.compact_root).resolve()
    samples = {}
    for rho in couplings:
        signals, _ = discover_samples(compact / directories[rho])
        for mass in masses:
            meta = signals.get(str(mass))
            if meta is None or not np.isclose(float(meta["rho_tc"]), rho):
                raise ValueError(f"No compact signal for m{mass}, rho_tc={rho} in {compact / directories[rho]}")
            samples[mass, rho] = meta
    out, registry_dir, dataset_root, runs = (Path(getattr(args, name)).resolve()
                                             for name in ("outdir", "registry_dir", "dataset_dir", "runs_dir"))
    if not args.run_prefix or Path(args.run_prefix).name != args.run_prefix or any(x in args.run_prefix for x in "/\\:"):
        raise ValueError("run-prefix must be a directory name")
    run_ids = {(mass, seed): f"{args.run_prefix}-m{mass}-seed{seed}" for mass in masses for seed in seeds}
    refuse_existing(out, registry_dir, dataset_root, *(runs / name for name in run_ids.values()))

    parent_registry = Path(phase2["registry"]).resolve()
    inputs = hash_files([*registry_files(parent_registry), anchor_files.dataset(anchor_mass),
                         *sorted(anchor_files.splits.glob("*")), reference_report / "provenance/summary.json",
                         reference_report / "provenance/status.json"])
    records = []
    for (mass, rho), meta in samples.items():
        directory = compact / directories[rho]
        export_path = directory / meta["export_manifest"]
        records.append(dict(directory=str(directory), meta=meta, export=json.loads(export_path.read_text()),
                            export_sha256=sha256_file(export_path)))
        inputs[str(export_path)] = records[-1]["export_sha256"]
        for path in directory.glob(part_glob_pattern(meta["_stem"])):
            inputs[str(path)] = sha256_file(path)
    preregistration = dict(study=str(study.directory), arguments=vars(args), analysis=asdict(analysis),
                           settings=settings, cut_reference=cut_settings, fit_parameters=params,
                           reference_report=str(reference_report), stored_lumi_pb_inv=stored_lumi,
                           test_evaluated=False)
    open_report(out, preregistration, inputs, records, preregister=True)
    checks = [dict(check="phase2_report_complete_and_unchanged", passed=True),
              dict(check="evaluation_config_and_k_factors_match_phase2", passed=True),
              dict(check="features_and_hyperparameters_match_the_validated_phase2_recipe", passed=True)]

    # 1. Registry: the Phase 2 registry plus every grid sample, without moving any existing event.
    parent = load_registry(parent_registry)
    registry, added = parent.copy(), []
    for (mass, rho), meta in samples.items():
        if meta["key"] not in set(registry.sample_key):
            frame = load_sample_frame(compact / directories[rho], meta, meta["key"], features, stored_lumi, study)
            registry = extend_registry(registry, frame[["event_id", "sample_key"]], seed=settings["registry_seed"])
            added.append(meta["key"])
    columns = ["event_id", "sample_key", "split", "assignment_method"]
    kept = registry[registry.sample_key.isin(parent.sample_key.unique())][columns]
    if not kept.sort_values("event_id").reset_index(drop=True).equals(
            parent[columns].sort_values("event_id").reset_index(drop=True)):
        raise ValueError("Registry extension moved an existing assignment")
    registry_dir.mkdir(parents=True)
    registry.to_parquet(registry_dir / "assignments.parquet", index=False)
    write_json(registry_dir / "registry.json", dict(
        status="complete", method="anchored_hash_rank_v1", seed=settings["registry_seed"], added_samples=added,
        parent=dict(directory=str(parent_registry),
                    assignments_sha256=json.loads((parent_registry / "registry.json").read_text())["assignments_sha256"]),
        assignments_sha256=sha256_file(registry_dir / "assignments.parquet"),
        samples=sorted(registry.sample_key.unique()), study=str(out)))
    checks.append(dict(check="registry_extension_preserves_every_existing_assignment", passed=True))
    print(f"Registry extended by {len(added)} samples; preparing {len(samples)} benchmarks", flush=True)

    # 2. One prepared benchmark per point: common registry, common background, unchanged stored weights.
    full, validation_frames, training_splits, background_projection, rates = {}, {}, {}, None, []
    for (mass, rho), meta in samples.items():
        point_full, splits = prepare_registered_reference(study, compact / directories[rho], meta, anchor, registry,
                                                          dataset_root / directories[rho], anchor_mass=anchor_mass,
                                                          lumi=stored_lumi)
        projection = pd.concat([part.loc[part.target == 0, ["event_id", "sample_weight"]].assign(split=name)
                                for name, part in splits.items()]).sort_values("event_id").reset_index(drop=True)
        if background_projection is None:
            background_projection = projection
        pd.testing.assert_frame_equal(background_projection, projection)
        full[mass, rho] = point_full[IDENTITY].copy()
        validation_frames[mass, rho] = splits["val"]
        if rho == training:
            training_splits[mass] = splits
        signal_sum = float(point_full.loc[point_full.target == 1, "sample_weight"].sum())
        rates.append(dict(mass=mass, rho_tc=rho, sample=meta["key"], xs_pb=meta["xs_pb"],
                          xs_source=json.dumps(meta.get("xs_source")), n_root=meta["n_events_total"],
                          n_selected=meta["n_selected"], selection_efficiency=meta["n_selected"] / meta["n_events_total"],
                          validation_signal_mc=int((splits["val"].target == 1).sum()),
                          expected_signal_before_bdt=signal_sum * cfg.lumi_pb_inv / stored_lumi * cfg.signal_k_factor))
    checks.append(dict(check="common_background_assignments_and_weights_at_every_point", passed=True))
    previous = BenchmarkFiles(Path(phase2["dataset"]))
    for mass in sorted(set(phase2["settings"]["masses"]) & set(masses)):
        rho = phase2["settings"]["rho_tc"]
        if (mass, rho) in validation_frames:
            saved = pd.read_parquet(previous.split("val", mass))
            pd.testing.assert_frame_equal(saved[IDENTITY + features],
                                          validation_frames[mass, rho][IDENTITY + features].reset_index(drop=True))
    checks.append(dict(check="phase2_points_reproduce_phase2_validation_events_and_features", passed=True))

    # 3-4. Fits at the training coupling, every point scored on its own validation events, cut reference.
    bootstrap = dict(replicates=settings["bootstrap_replicates"], seed=settings["bootstrap_seed"])
    grid_rows, auc_rows, support_rows, runs_record, cut_rows = fit_and_evaluate(
        masses=masses, couplings=couplings, seeds=seeds, training=training, directories=directories,
        features=features, params=params, cfg=cfg, stored_lumi=stored_lumi, bootstrap=bootstrap, runs=runs,
        run_ids=run_ids, fit_policy=settings["fit_policy"], cut_settings=cut_settings, checks=checks,
        sample_keys={point: meta["key"] for point, meta in samples.items()},
        training_data=lambda mass: (training_splits[mass]["train"], training_splits[mass]["val"]),
        validation_data=lambda mass, rho: validation_frames[mass, rho], totals=lambda mass, rho: full[mass, rho])

    written = [p for name in run_ids.values() for p in (runs / name).rglob("*preds_test*")]
    checks.append(dict(check="no_test_scores_read_or_written",
                       passed=not written and not any(Path(p).name.startswith("preds_test") for p in inputs)))
    registry_counts = (registry[registry.sample_key.isin(added)].groupby(["sample_key", "split"]).size()
                       .unstack(fill_value=0).reset_index())
    grid = pd.DataFrame(grid_rows + cut_rows)
    tables = dict(checks=pd.DataFrame(checks), rates=pd.DataFrame(rates), registry_extension=registry_counts,
                  grid=grid, auc=pd.DataFrame(auc_rows),
                  process_support=pd.concat(support_rows, ignore_index=True) if support_rows
                  else pd.DataFrame(columns=["mass", "rho_tc", "seed", "objective"]))
    write_tables(out, tables)
    if not tables["checks"].passed.all():
        raise ValueError("Grid contract checks failed")
    frozen = {f"m{r['mass']}": str(Path(r["model_directory"]) / "points") for r in runs_record
              if r["seed"] == settings["primary_seed"]}
    summary = dict(schema_version=2, phases=[9], settings=settings, cut_reference=cut_settings,
                   evaluation_config=asdict(cfg), fit_parameters=params, runs=runs_record,
                   registry=str(registry_dir), dataset=str(dataset_root), reference_report=str(reference_report),
                   primary_seed=settings["primary_seed"], frozen_threshold_directories=frozen,
                   test_evaluated=False, status="checkpoint_ready",
                   note="Validation-only grid. Thresholds of the primary seed are frozen per point for one later test "
                        "evaluation; sensitivities of alternative hypotheses are never combined.")
    write_json(out / "provenance/summary.json", summary)
    save_grid_figures(out)
    artifacts = [p for r in runs_record for p in (runs / r["run_id"]).rglob("*") if p.is_file()]
    artifacts += [p for p in dataset_root.rglob("*") if p.is_file()] + [p for p in registry_dir.glob("*") if p.is_file()]
    finish_report(out, "04_physics_results.ipynb", inputs, sorted(set(artifacts)))


def run_complete(model_dir, couplings, directories):
    """Whether a fit's model directory holds every file a completed fit writes, and nothing else."""
    expected = {model_dir / name for name in RUN_FILES}
    expected |= {model_dir / "points" / directories[rho] / name for rho in couplings for name in POINT_FILES}
    return model_dir.is_dir() and {p for p in model_dir.rglob("*") if p.is_file()} == expected


def saved_run_rows(model_dir, *, mass, seed, run_id, couplings, directories, training, features, cfg, stored_lumi,
                   validation_data, totals):
    """A completed fit's table rows, re-derived from its files after re-checking them.

    The reloaded model must reproduce every saved validation score on the same rows, and
    each point's yield closure must hold; the rows equal those fit_and_evaluate writes.
    """
    if not run_complete(model_dir, couplings, directories):
        raise ValueError(f"{run_id}: incomplete run directory")
    _, _, predict = load_model(model_dir)
    train = pd.read_parquet(model_dir / "preds_train.parquet", columns=["target", "bdt_score"])
    train_auc = safe_auc(train.target.to_numpy(), train.bdt_score)
    grid_rows, auc_rows, support_rows = [], [], []
    for rho in couplings:
        point_dir = model_dir / "points" / directories[rho]
        frame = validation_data(mass, rho)
        saved = pd.read_parquet(point_dir / "preds_val.parquet")
        if list(saved.columns) != [*IDENTITY, "bdt_score"] or len(saved) != len(frame) or not all(
                (saved[column].to_numpy() == frame[column].to_numpy()).all() for column in IDENTITY):
            raise ValueError(f"{run_id} rho{rho:g}: saved scores are not on this point's validation rows")
        X, _, _ = make_xyw(frame, features)
        np.testing.assert_allclose(predict(X), saved.bdt_score, rtol=1e-6, atol=2e-7)
        weights, _ = evaluation_weights(totals(mass, rho), saved, stored_lumi=stored_lumi, target_lumi=cfg.lumi_pb_inv,
                                        signal_k_factor=cfg.signal_k_factor,
                                        background_k_factor=cfg.background_k_factor)
        check_yield_closure(totals(mass, rho), frame, weights, cfg, stored_lumi=stored_lumi)
        metadata = json.loads((point_dir / "operating_points.json").read_text())
        grid_rows += point_rows(metadata, mass=mass, rho_tc=rho, seed=seed, run_id=run_id, method="bdt",
                                evaluation="genuine sample")
        auc_rows.append(dict(mass=mass, rho_tc=rho, seed=seed, auc_weighted=metadata["auc_weighted"],
                             auc_unweighted=metadata["auc_unweighted"],
                             train_auc_unweighted=train_auc if rho == training else None))
        record = SimpleNamespace(metadata=metadata, processes=pd.read_parquet(point_dir / "process_curves.parquet"))
        for objective, table in selected_support(record, saved, cfg, stored_lumi=stored_lumi):
            support_rows.append(table.assign(mass=mass, rho_tc=rho, seed=seed, objective=objective))
    return grid_rows, auc_rows, support_rows


def fit_and_evaluate(*, masses, couplings, seeds, training, directories, features, params, cfg, stored_lumi,
                     bootstrap, runs, run_ids, fit_policy, cut_settings, checks, sample_keys, training_data,
                     validation_data, totals, fit_weights=None, reuse=frozenset()):
    """Fits at the training coupling, every point scored on its own validation events, then the cut reference.

    training_data(mass) -> (train, val) and validation_data(mass, rho) -> frame give prepared rows;
    totals(mass, rho) gives the frame whose per-sample weight sums are the full prepared yields.
    fit_weights names a per-row fit-weight column; None fits unweighted. reuse names (mass, seed)
    fits completed by an interrupted run: they are re-checked and read back, not refitted.
    """
    grid_rows, auc_rows, support_rows, runs_record, cut_rows = [], [], [], [], []
    for mass in masses:
        train, val = training_data(mass)
        X_train, y_train, _ = make_xyw(train, features)
        X_val, y_val, _ = make_xyw(val, features)
        weights = ({} if fit_weights is None else
                   dict(fit_weights_train=train[fit_weights].to_numpy(dtype=float),
                        fit_weights_val=val[fit_weights].to_numpy(dtype=float)))
        for seed in seeds:
            started = time.perf_counter()
            run_id = run_ids[mass, seed]
            model_dir = runs / run_id / "models" / model_dirname(mass)
            if (mass, seed) in reuse:
                rows = saved_run_rows(model_dir, mass=mass, seed=seed, run_id=run_id, couplings=couplings,
                                      directories=directories, training=training, features=features, cfg=cfg,
                                      stored_lumi=stored_lumi, validation_data=validation_data, totals=totals)
                grid_rows += rows[0]
                auc_rows += rows[1]
                support_rows += rows[2]
                runs_record.append(dict(mass=mass, seed=seed, run_id=run_id, model_directory=str(model_dir),
                                        training_coupling=training))
                checks += [dict(check=f"{run_id}: reload_score_closure_and_yield_closure", passed=True),
                           dict(check=f"{run_id}: yield_closure_at_every_coupling", passed=True),
                           dict(check=f"{run_id}: reused from the interrupted run; files complete, every saved "
                                      "validation score reproduced by the reloaded model", passed=True)]
                print(f"Reused {run_id}: {len(couplings)} couplings re-checked in "
                      f"{time.perf_counter() - started:.0f} s", flush=True)
                continue
            model_dir.mkdir(parents=True)
            model = train_xgb(X_train, y_train, X_val, y_val, seed=seed, **params, record_training=True,
                              **weights)
            model.get_booster().save_model(model_dir / "model.ubj")
            predictions = {name: frame[IDENTITY].assign(bdt_score=model.predict_proba(X)[:, 1])
                           for name, frame, X in (("train", train, X_train), ("val", val, X_val))}
            record = validate_model(totals(mass, training), predictions, cfg, stored_lumi=stored_lumi, **bootstrap)
            check_yield_closure(totals(mass, training), val, record.weights, cfg, stored_lumi=stored_lumi)
            metrics = dict(mass=mass, rho_tc=training, features=features, evaluation=record.metadata,
                           lumi_pb_inv=cfg.lumi_pb_inv, fit_policy=fit_policy, test_evaluated=False,
                           xgb=dict(best_iteration=int(model.best_iteration), params=model.get_params(),
                                    weight_mode=fit_policy))
            write_json(model_dir / "metrics.json", metrics)
            write_json(model_dir / "operating_points.json", record.metadata)
            write_json(model_dir / "learning_curves.json", model.evals_result())
            write_validation(model_dir, record)
            roc_tables(totals(mass, training), predictions, cfg, stored_lumi=stored_lumi).to_parquet(
                model_dir / "roc.parquet", index=False)
            for name, frame in predictions.items():
                frame.to_parquet(model_dir / f"preds_{name}.parquet", index=False)
            _, _, reload_predict = load_model(model_dir)
            np.testing.assert_allclose(reload_predict(X_val), predictions["val"].bdt_score, rtol=1e-6, atol=2e-7)
            checks.append(dict(check=f"{run_id}: reload_score_closure_and_yield_closure", passed=True))
            runs_record.append(dict(mass=mass, seed=seed, run_id=run_id, model_directory=str(model_dir),
                                    training_coupling=training))
            for rho in couplings:
                point_dir = model_dir / "points" / directories[rho]
                point_dir.mkdir(parents=True)
                frame = validation_data(mass, rho)
                X, y, _ = make_xyw(frame, features)
                scored = {"val": frame[IDENTITY].assign(bdt_score=model.predict_proba(X)[:, 1])}
                point = validate_model(totals(mass, rho), scored, cfg, stored_lumi=stored_lumi, **bootstrap)
                check_yield_closure(totals(mass, rho), frame, point.weights, cfg, stored_lumi=stored_lumi)
                write_validation(point_dir, point)
                write_json(point_dir / "operating_points.json", dict(point.metadata, mass=mass, rho_tc=rho,
                                                                     sample=sample_keys[mass, rho],
                                                                     training_coupling=training))
                roc_tables(totals(mass, rho), scored, cfg, stored_lumi=stored_lumi).to_parquet(
                    point_dir / "roc.parquet", index=False)
                scored["val"].to_parquet(point_dir / "preds_val.parquet", index=False)
                grid_rows += point_rows(point.metadata, mass=mass, rho_tc=rho, seed=seed, run_id=run_id, method="bdt",
                                        evaluation="genuine sample")
                auc_rows.append(dict(mass=mass, rho_tc=rho, seed=seed, auc_weighted=point.metadata["auc_weighted"],
                                     auc_unweighted=point.metadata["auc_unweighted"],
                                     train_auc_unweighted=safe_auc(y_train, predictions["train"].bdt_score)
                                     if rho == training else None))
                for objective, table in selected_support(point, scored["val"], cfg, stored_lumi=stored_lumi):
                    support_rows.append(table.assign(mass=mass, rho_tc=rho, seed=seed, objective=objective))
            checks.append(dict(check=f"{run_id}: yield_closure_at_every_coupling", passed=True))
            print(f"Completed {run_id}: {len(couplings)} couplings evaluated in "
                  f"{time.perf_counter() - started:.0f} s; test sealed", flush=True)

    # 4. Bounded cut reference at every point (descriptive, independent of the BDT seeds).
    for rho in couplings:  # the order in which the points were prepared
        for mass in masses:
            points, _, _, _ = cut_reference(totals(mass, rho), validation_data(mass, rho), mass, cut_settings, cfg,
                                            stored_lumi=stored_lumi)
            for objective, point in points.items():
                item = dict.fromkeys(["S", "B", "S_over_B", *OBJECTIVES, "background_neff", "signal_efficiency",
                                      "background_efficiency"])
                item.update({k: v for k, v in point.get("metrics", {}).items() if k in item})
                cut_rows.append(dict(item, mass=mass, rho_tc=rho, seed=None, method="cuts", objective=objective,
                                     status=point["status"], threshold=None,
                                     candidate_id=(point.get("candidate") or {}).get("candidate_id")))

    return grid_rows, auc_rows, support_rows, runs_record, cut_rows
