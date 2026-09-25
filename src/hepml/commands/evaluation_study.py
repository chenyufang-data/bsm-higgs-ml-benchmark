"""Phases 4/7/8: fixed-split weight-policy pilot and validation operating points."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from hepml_compact.parquet_writer import sha256_file

from hepml.adapters.configuration import default_study_directory, load_analysis
from hepml.adapters.dataset_files import discover_samples, load_sample_frame
from hepml.adapters.evaluation_report import stability_figure, threshold_figure
from hepml.adapters.research import verify_stage
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
    write_validation,
)
from hepml.adapters.xgboost_model import train_xgb
from hepml.application.evaluation import evaluation_weights, selected_support, validate_model
from hepml.application.training import build_fit_weights, make_xyw
from hepml.domain.artifacts import BenchmarkFiles, model_dirname
from hepml.domain.metrics import normalized_weights, sample_normalization
from hepml.domain.weights import balanced_hypothesis_weights


def prior_audit(study, compact_root, registry, full, train, settings, stored_lumi, evaluation_config):
    """Two-mass objective-measure audit only; no parameterized model is fitted."""
    rho = settings["prior_audit_rho"]
    validation = yaml.safe_load((study.directory / "validation.yaml").read_text())
    directory = compact_root / validation["coupling_directories"][str(rho)]
    signals, _ = discover_samples(directory)
    physical = normalized_weights(train, sample_normalization(full, train))
    background = train.loc[train.target == 0, ["event_id", "sample", "target"]].copy()
    background["physics_weight"] = physical[train.target == 0] * evaluation_config.background_k_factor
    frames, rates, records = [], [], []
    for mass, meta in signals.items():
        export_path = directory / meta["export_manifest"]
        records.append(dict(directory=str(directory.resolve()), metadata=meta,
                            export=json.loads(export_path.read_text()), export_sha256=sha256_file(export_path)))
        rates.append(dict(mass=int(mass), rho=rho, xs_pb=meta["xs_pb"],
                          n_root=meta["n_events_total"], n_selected=meta["n_selected"],
                          selection_efficiency=meta["n_selected"] / meta["n_events_total"]))
    for mass in settings["prior_audit_masses"]:
        meta = signals[str(mass)]
        sample = load_sample_frame(directory, meta, meta["key"], list(study.plugin.FEATURES), stored_lumi, study)
        sample["target"] = 1
        registered = registry[registry.sample_key == meta["key"]]
        if set(registered.event_id) != set(sample.event_id):
            raise ValueError("Prior audit signal differs from shared registry")
        part = sample[sample.event_id.isin(registered.loc[registered.split == "train", "event_id"])].copy()
        part["physics_weight"] = (normalized_weights(part, sample_normalization(sample, part))
                                  * evaluation_config.signal_k_factor)
        part = part[["event_id", "sample", "target", "physics_weight"]]
        frames.append(pd.concat([part, background], ignore_index=True).assign(hypothesis=mass))
    pooled = pd.concat(frames, ignore_index=True)
    prior = {mass: 1 / len(settings["prior_audit_masses"]) for mass in settings["prior_audit_masses"]}
    pooled["fit_weight"] = balanced_hypothesis_weights(pooled.target, pooled.physics_weight, pooled.hypothesis,
                                                      prior, event_ids=pooled.event_id)
    audit = pooled.groupby(["hypothesis", "target", "sample"]).agg(
        rows=("event_id", "size"), sum_fit_weight=("fit_weight", "sum"),
        physics_weight_per_hypothesis=("physics_weight", "sum")).reset_index()
    sums = pooled.groupby(["target", "hypothesis"]).fit_weight.sum()
    expected = pooled.event_id.nunique() / 2 / len(prior)
    if not np.allclose(sums, expected):
        raise ValueError("Class-conditional hypothesis priors do not match")
    return audit, pd.DataFrame(rates), records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study")
    parser.add_argument("--benchmark", required=True, help="Existing benchmark data/ directory")
    parser.add_argument("--registry-dir", required=True)
    parser.add_argument("--compact-root", required=True)
    parser.add_argument("--pilot-run", help="Optional saved unweighted seed-42 run to re-evaluate")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--run-prefix", required=True)
    args = parser.parse_args(argv)
    return run_study(args.outdir, _run, args)


def _run(args):
    study = load_study(args.study or default_study_directory())
    analysis = load_analysis(study.directory / "analysis.yaml")
    cfg = analysis.evaluation
    settings = yaml.safe_load((study.directory / "validation.yaml").read_text())["evaluation_study"]
    mass, features = settings["pilot_mass"], list(study.plugin.FEATURES)
    benchmark, out = Path(args.benchmark).resolve(), Path(args.outdir).resolve()
    registry_dir, runs_dir = Path(args.registry_dir).resolve(), Path(args.runs_dir).resolve()
    refuse_existing(out)
    registry = load_registry(registry_dir)
    files = BenchmarkFiles(benchmark)
    full = pd.read_parquet(files.dataset(mass), columns=["event_id", "sample_key", "sample", "target", "sample_weight"])
    train = pd.read_parquet(files.split("train", mass))
    val = pd.read_parquet(files.split("val", mass))
    split_meta = json.loads(files.meta(mass).read_text())
    stored_lumi = float(split_meta["lumi"])
    for name, frame in (("train", train), ("val", val)):
        expected = registry.loc[registry.event_id.isin(full.event_id) & (registry.split == name), "event_id"]
        if set(frame.event_id) != set(expected) or frame.event_id.duplicated().any():
            raise ValueError(f"{name}: pilot and common registry assignments differ")
    run_ids = [f"{args.run_prefix}-{policy}-seed{seed}" for policy in settings["fit_policies"] for seed in settings["seeds"]]
    refuse_existing(*(runs_dir / run_id for run_id in run_ids))
    source_config = dict(study=str(study.directory.resolve()), arguments=vars(args), analysis=asdict(analysis),
                         study_settings=settings, stored_lumi_pb_inv=stored_lumi)
    input_hashes = hash_files([files.dataset(mass), files.split("train", mass), files.split("val", mass),
                               files.meta(mass), *registry_files(registry_dir)])
    open_report(out, source_config, input_hashes)
    audit, rates, compact_records = prior_audit(study, Path(args.compact_root), registry, full, train,
                                               settings, stored_lumi, cfg)
    provenance_path = out / "provenance/provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["inputs"] = compact_records
    write_json(provenance_path, provenance)
    audit.to_csv(out / "tables/hypothesis_fit_priors.csv", index=False)
    rates["k_factor"] = cfg.signal_k_factor
    rates["corrected_xs_pb"] = rates.xs_pb * cfg.signal_k_factor
    rates["expected_signal_500fb"] = rates.corrected_xs_pb * rates.selection_efficiency * cfg.lumi_pb_inv
    rates["fails_legacy_min_expected_signal_50"] = rates.expected_signal_500fb < 50
    rates.to_csv(out / "tables/constraint_audit.csv", index=False)
    X_train, y_train, _ = make_xyw(train, features)
    X_val, y_val, _ = make_xyw(val, features)
    records, selected_rows, support_rows, fit_rows, checks, seed_rows = [], [], [], [], [], []
    checks.append(dict(check="two_mass_class_conditional_priors", passed=True))
    training = asdict(analysis.training)
    training.pop("scan")
    training.pop("seed")
    pilot = Path(args.pilot_run).resolve() if args.pilot_run else None
    if pilot:
        verify_stage(pilot)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out / "figures/threshold_curves.pdf") as pdf, PdfPages(out / "figures/threshold_stability.pdf") as stability_pdf:
        for policy in settings["fit_policies"]:
            fit_train = build_fit_weights(train, policy, reference=full)
            fit_val = build_fit_weights(val, policy, reference=full)
            weights_for_fit = np.ones(len(train)) if fit_train is None else fit_train
            fit_frame = train.assign(fit_weight=weights_for_fit)
            for sample, group in fit_frame.groupby("sample"):
                fit_rows.append(dict(policy=policy, sample=sample, target=int(group.target.iloc[0]),
                                     raw_count=len(group), sum_fit_weight=group.fit_weight.sum(),
                                     sum_physical_weight=group.sample_weight.sum(),
                                     physical_neff=group.sample_weight.sum()**2/(group.sample_weight**2).sum()))
            for seed in settings["seeds"]:
                run_id = f"{args.run_prefix}-{policy}-seed{seed}"
                model_dir = runs_dir / run_id / "models" / model_dirname(mass)
                model_dir.mkdir(parents=True)
                reused = policy == "none" and seed == 42 and pilot is not None
                if reused:
                    old_dir = pilot / "models" / model_dirname(mass)
                    old_metrics = json.loads((old_dir / "metrics.json").read_text())
                    if old_metrics["features"] != features or old_metrics["xgb"]["weight_mode"] != "none":
                        raise ValueError("Pilot features or fitting policy differs")
                    if any(old_metrics["xgb"]["params"].get(key) != value for key, value in training.items()):
                        raise ValueError("Pilot hyperparameters differ; omit --pilot-run to refit")
                    saved_train = pd.read_parquet(old_dir / "preds_train.parquet")
                    saved_val = pd.read_parquet(old_dir / "preds_val.parquet")
                    for original, saved in ((train, saved_train), (val, saved_val)):
                        pd.testing.assert_frame_equal(original[["event_id", "sample_weight"]], saved[["event_id", "sample_weight"]])
                    p_train, p_val = saved_train.bdt_score.to_numpy(), saved_val.bdt_score.to_numpy()
                    shutil.copy2(old_dir / "model.ubj", model_dir / "model.ubj")
                    xgb_metadata = old_metrics["xgb"]
                else:
                    model = train_xgb(X_train, y_train, X_val, y_val, seed=seed, **training,
                                      fit_weights_train=fit_train, fit_weights_val=fit_val)
                    p_train, p_val = model.predict_proba(X_train)[:, 1], model.predict_proba(X_val)[:, 1]
                    model.get_booster().save_model(model_dir / "model.ubj")
                    xgb_metadata = dict(best_iteration=int(model.best_iteration), params=model.get_params(), weight_mode=policy)
                columns = ["event_id", "sample_key", "sample", "target", "sample_weight"]
                predicted_train, predicted_val = train[columns].assign(bdt_score=p_train), val[columns].assign(bdt_score=p_val)
                record = validate_model(full, {"train": predicted_train, "val": predicted_val}, cfg,
                                        stored_lumi=stored_lumi, replicates=settings["bootstrap_replicates"],
                                        seed=settings["bootstrap_seed"])
                metadata, curve, bands, stability = record.metadata, record.curve, record.bands, record.stability
                metrics = dict(mass=mass, features=features, xgb=xgb_metadata, evaluation=metadata,
                               lumi_pb_inv=cfg.lumi_pb_inv, fit_policy=policy,
                               reused_model_from=str(pilot) if reused else None, test_evaluated=False)
                write_json(model_dir / "metrics.json", metrics)
                write_validation(model_dir, record)
                predicted_train.to_parquet(model_dir / "preds_train.parquet", index=False)
                predicted_val.to_parquet(model_dir / "preds_val.parquet", index=False)
                write_json(model_dir / "operating_points.json", metadata)
                support = dict(selected_support(record, predicted_val, cfg, stored_lumi=stored_lumi))
                for objective, point in metadata["operating_points"].items():
                    row = dict(run_id=run_id, policy=policy, seed=seed, objective=objective,
                               status=point["status"], threshold=point["threshold"], plateau_threshold=point["plateau_threshold"],
                               auc_weighted=metadata["auc_weighted"], auc_unweighted=metadata["auc_unweighted"])
                    row.update({name: None for name in ("S", "B", "S_over_B", "Z_Asimov", "Z_syst_5pct", "Z_syst_10pct",
                                                        "background_neff", "signal_efficiency", "background_efficiency")})
                    row["eligible"] = False
                    if point["status"] == "valid":
                        row.update(point["metrics"])
                        support_rows.append(support[objective].assign(run_id=run_id, objective=objective))
                    selected_rows.append(row)
                fig = threshold_figure(curve, bands, metadata["operating_points"], title=f"{run_id}: validation, {cfg.lumi_pb_inv/1000:g} fb^-1")
                pdf.savefig(fig)
                plt.close(fig)
                fig = stability_figure(stability, title=run_id)
                stability_pdf.savefig(fig)
                plt.close(fig)
                records.append(dict(run_id=run_id, policy=policy, seed=seed, model_directory=str(model_dir), reused=reused))
                seed_rows.append(dict(run_id=run_id, seed=seed, policy=policy, **{key: metadata[key] for key in ("auc_weighted", "auc_unweighted")}))
                # Closure and fit-weight isolation are evaluated at fixed scores, not after refitting.
                double_cfg = type(cfg)(**dict(asdict(cfg), lumi_pb_inv=2*cfg.lumi_pb_inv))
                double_weights, _ = evaluation_weights(full, predicted_val, stored_lumi=stored_lumi,
                                                        target_lumi=double_cfg.lumi_pb_inv,
                                                        signal_k_factor=cfg.signal_k_factor,
                                                        background_k_factor=cfg.background_k_factor)
                checks.append(dict(check=f"{run_id}: luminosity_scales_once",
                                   passed=bool(np.allclose(double_weights, 2*record.weights))))
                checks.append(dict(check=f"{run_id}: stored_physical_weights_unchanged",
                                   passed=bool(np.array_equal(predicted_val.sample_weight, val.sample_weight))))
                print(f"Completed {run_id}: validation AUC={metadata['auc_weighted']:.4f}; test sealed", flush=True)
    pd.DataFrame(selected_rows).to_csv(out / "tables/operating_points.csv", index=False)
    support_columns = ["run_id", "objective", "threshold", "sample", "target", "mc_count", "N_eff", "sum_weight",
                       "sum_weight_squared", "validation_mc", "efficiency_lower", "efficiency_upper",
                       "yield_lower", "yield_upper"]
    support_table = (pd.concat(support_rows, ignore_index=True)[support_columns] if support_rows
                     else pd.DataFrame(columns=support_columns))
    support_table.to_csv(out / "tables/process_support.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(out / "tables/fit_weight_audit.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out / "tables/seed_comparison.csv", index=False)
    selected = pd.DataFrame(selected_rows)
    policy_comparison = selected[selected.objective == cfg.primary_objective].groupby("policy").agg(
        valid_seeds=("eligible", "sum"), auc_mean=("auc_weighted", "mean"), auc_std=("auc_weighted", "std"),
        primary_Z_mean=(cfg.primary_objective, "mean"), primary_Z_std=(cfg.primary_objective, "std"),
        threshold_min=("threshold", "min"), threshold_max=("threshold", "max")).reset_index()
    policy_comparison.to_csv(out / "tables/policy_comparison.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "tables/checks.csv", index=False)
    if not all(row["passed"] for row in checks):
        raise ValueError("Evaluation contract checks failed")
    summary = dict(schema_version=2, phases=[4,7,8], runs=records, evaluation_config=asdict(cfg),
                   settings=settings, test_evaluated=False, benchmark=str(benchmark), registry=str(registry_dir),
                   status="checkpoint_ready", primary_fit_policy=settings["primary_fit_policy"],
                   note="Weight-policy adoption awaits checkpoint review; no test-derived model or threshold choice.")
    write_json(out / "provenance/summary.json", summary)
    for run in records:
        write_json(Path(run["model_directory"]) / "provenance.json",
                   dict(report=str(out), settings=source_config, inputs=input_hashes))
    finish_report(out, "02_bdt_baselines.ipynb", input_hashes,
                  [path for run in records for path in Path(run["model_directory"]).glob("*") if path.is_file()])


if __name__ == "__main__":
    main()
