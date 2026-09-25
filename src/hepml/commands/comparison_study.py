"""Phase 5: paired mass-specific versus parameterized comparison on identical validation events."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hepml.adapters.comparison_report import save_comparison_figures
from hepml.adapters.configuration import default_study_directory, fit_parameters, load_analysis
from hepml.adapters.inference import load_model
from hepml.adapters.study_loader import load_study
from hepml.adapters.study_run import (
    comparable_config,
    completed_report,
    finish_report,
    hash_files,
    open_report,
    refuse_existing,
    run_study,
    write_json,
    write_tables,
    write_validation,
)
from hepml.adapters.xgboost_model import measured_train_xgb
from hepml.application.comparison import (
    align_predictions,
    combine_outcomes,
    joint_paired_bootstrap,
    noninferiority,
    paired_metrics,
    selection_support,
    working_point,
)
from hepml.application.conditioning import anchored_class_weights
from hepml.application.evaluation import (
    check_yield_closure,
    evaluation_weights,
    point_rows,
    roc_tables,
    validate_model,
)
from hepml.application.training import make_xyw
from hepml.domain.artifacts import BenchmarkFiles, model_dirname
from hepml.domain.metrics import safe_auc, yield_metrics

IDENTITY = ["event_id", "sample_key", "sample", "target", "sample_weight"]
CONTROL_POLICY = "anchored-reference-class-totals"
SEED_RULE = "every_seed_non_inferior"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study")
    parser.add_argument("--reference-report", required=True, help="Completed Phase 2 mass-specific report")
    parser.add_argument("--parameterized-report", required=True, action="append",
                        help="Completed Phase 3 report built on it; repeat once per compared seed")
    parser.add_argument("--runs-dir", required=True, help="Parent of the new loss-weight control runs")
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    run_study(args.outdir, _run, args)


def efficiency_grid(settings):
    """Pre-registered working points plus a fixed grid for the efficiency-ratio curve."""
    points = [settings["background_efficiency"], *settings["secondary_background_efficiencies"]]
    grid = {f"{x:g}": float(x) for x in np.round(np.arange(0.05, 0.601, 0.05), 2)}
    grid.update({f"{x:g}": float(x) for x in points})
    return sorted(grid.values())


def validate_settings(settings, phase2, phase3):
    """phase3 maps each parameterized fit's seed to its report summary."""
    masses, seeds = settings["masses"], settings["seeds"]
    points = [settings["background_efficiency"], *settings["secondary_background_efficiencies"]]
    margins = settings["noninferiority"]
    trained = set(phase2["settings"]["masses"]).intersection(*(set(s["settings"]["masses"]) for s in phase3.values()))
    if (settings["reference"] != "mass_specific_bdt" or settings["candidate"] != "parameterized_bdt"
            or settings["primary_metric"] != "signal_efficiency_at_background_efficiency"
            or not masses or len(set(masses)) != len(masses) or not set(masses) <= trained
            or not set(settings["controls"]) <= {"cut_reference", "loss_weight_control"}
            or any(not 0 < x < 1 for x in points) or settings["bootstrap_replicates"] < 2
            or not 0.5 < margins["one_sided_confidence"] < 1
            or margins["max_relative_signal_efficiency_loss"] < 0 or margins["max_weighted_auc_loss"] < 0):
        raise ValueError("Unsupported paired-comparison design")
    if any(s["settings"]["rho_tc"] != settings["rho_tc"] for s in [phase2, *phase3.values()]):
        raise ValueError("Coupling differs from the Phase 2/3 inputs")
    if (len(set(seeds)) != len(seeds) or sorted(seeds) != sorted(phase3)
            or not set(seeds) <= set(phase2["settings"]["seeds"])):
        raise ValueError("Each compared seed needs exactly one mass-specific and one parameterized fit")
    if len(seeds) > 1 and settings.get("seed_rule") != SEED_RULE:
        raise ValueError(f"Several seeds need the pre-registered seed rule {SEED_RULE!r}")
    conditioning = [s["conditioning"] for s in phase3.values()]
    if any(c != conditioning[0] for c in conditioning):
        raise ValueError("Parameterized fits of different seeds use different conditioning")


def _run(args):
    study = load_study(args.study or default_study_directory())
    analysis = load_analysis(study.directory / "analysis.yaml")
    cfg = analysis.evaluation
    settings = yaml.safe_load((study.directory / "validation.yaml").read_text())["comparison_study"]
    reference_report = Path(args.reference_report).resolve()
    phase2 = completed_report(reference_report, [2])
    phase3, parameterized_reports = {}, {}
    for path in args.parameterized_report:
        report = Path(path).resolve()
        summary = completed_report(report, [3])
        if Path(summary["reference_report"]).resolve() != reference_report:
            raise ValueError(f"{report} was built on a different Phase 2 report")
        seed = summary["settings"]["seed"]
        if seed in phase3:
            raise ValueError(f"Two parameterized reports share seed {seed}")
        phase3[seed], parameterized_reports[seed] = summary, report
    current = comparable_config(asdict(cfg))
    if any(comparable_config(s["evaluation_config"]) != current for s in [phase2, *phase3.values()]):
        raise ValueError("Evaluation settings (luminosity, k-factors, support rules) differ from Phases 2/3")
    validate_settings(settings, phase2, phase3)
    masses, seeds = settings["masses"], settings["seeds"]
    conditioning = phase3[seeds[0]]["conditioning"]
    reference_mass, feature = conditioning["reference_mass"], conditioning["feature"]
    prior = {float(key): value for key, value in conditioning["prior"].items()}
    control_masses = [m for m in masses if m != reference_mass] if "loss_weight_control" in settings["controls"] else []
    out, runs = Path(args.outdir).resolve(), Path(args.runs_dir).resolve()
    if not args.run_prefix or Path(args.run_prefix).name != args.run_prefix or any(x in args.run_prefix for x in "/\\:"):
        raise ValueError("run-prefix must be a directory name")
    control_ids = {(m, s): f"{args.run_prefix}-m{m}-loss-control-seed{s}" for m in control_masses for s in seeds}
    refuse_existing(out, *(runs / name for name in control_ids.values()))

    features = list(study.plugin.FEATURES)
    shared_runs = {seed: Path(phase3[seed]["shared_model_directory"]) for seed in seeds}
    reference_dirs = {(r["mass"], r["seed"]): Path(r["model_directory"]) for r in phase2["runs"]}
    candidate_dirs = {(r["mass"], seed): Path(r["model_directory"]) for seed in seeds for r in phase3[seed]["runs"]}
    # One budget for every compared and control fit: the recorded hyperparameters of the
    # Phase 2 fits, which may differ from analysis.yaml through a recorded override.
    first = json.loads((reference_dirs[masses[0], seeds[0]] / "metrics.json").read_text())["xgb"]["params"]
    params = {key: first[key] for key in fit_parameters(analysis)}
    for mass in masses:
        for seed in seeds:
            recorded = json.loads((reference_dirs[mass, seed] / "metrics.json").read_text())
            shared = json.loads((shared_runs[seed] / "metrics.json").read_text())
            for metrics in (recorded, shared):
                if any(metrics["xgb"]["params"].get(key) != value for key, value in params.items()):
                    raise ValueError("Hyperparameters differ from the Phase 2/3 fits; comparisons need one budget")
            if recorded["features"] != features or conditioning["physics_features"] != features:
                raise ValueError("Feature contract differs from the Phase 2/3 fits")

    datasets = BenchmarkFiles(Path(phase2["dataset"]))
    full, splits, stored_lumi, paths = {}, {}, None, []
    for mass in sorted(set(masses) | ({reference_mass} if control_masses else set())):
        meta = json.loads(datasets.meta(mass).read_text())
        if stored_lumi is not None and float(meta["lumi"]) != stored_lumi:
            raise ValueError("Stored luminosity differs between masses")
        stored_lumi = float(meta["lumi"])
        full[mass] = pd.read_parquet(datasets.dataset(mass), columns=IDENTITY)
        splits[mass] = {name: pd.read_parquet(datasets.split(name, mass)) for name in ("train", "val")}
        paths += [datasets.meta(mass), datasets.dataset(mass), *(datasets.split(name, mass) for name in ("train", "val"))]
    for mass in masses:
        paths += [directory / "preds_val.parquet" for (m, _), directory in reference_dirs.items() if m == mass]
        paths += [candidate_dirs[mass, seed] / "preds_val.parquet" for seed in seeds]
    for seed in seeds:
        run = shared_runs[seed]
        paths += [run / "conditioning.json", run / "metrics.json", *(run / f"expanded_{n}.parquet" for n in ("train", "val"))]
    for report in (reference_report, *parameterized_reports.values()):
        paths += [report / "provenance/summary.json", report / "provenance/status.json"]
    inputs = hash_files(sorted(set(paths)))
    preregistration = dict(study=str(study.directory), arguments=vars(args), analysis=asdict(analysis),
                           settings=settings, reference_report=str(reference_report),
                           parameterized_reports={str(s): str(p) for s, p in parameterized_reports.items()},
                           control_masses=control_masses, control_policy=CONTROL_POLICY, fit_parameters=params,
                           efficiency_grid=efficiency_grid(settings), test_evaluated=False)
    open_report(out, preregistration, inputs, preregister=True)
    checks = [dict(check="phase2_and_phase3_reports_complete_and_unchanged", passed=True),
              dict(check="phase3_reports_built_on_this_phase2_report", passed=True),
              dict(check="evaluation_config_and_k_factors_match_phases_2_and_3", passed=True),
              dict(check="features_and_hyperparameters_match_phases_2_and_3", passed=True)]

    # Loss-weight control: the mass-specific fit with the parameterized model's class-loss ratio, per seed.
    controls, control_dirs, control_rows, resources, runs_record = {}, {}, [], [], []
    phase2_bootstrap = dict(replicates=phase2["settings"]["bootstrap_replicates"], seed=phase2["settings"]["bootstrap_seed"])
    for (mass, seed), run_id in control_ids.items():
        train, val = splits[mass]["train"], splits[mass]["val"]
        fit = {}
        for name, frame in (("train", train), ("val", val)):
            anchor = int((splits[reference_mass][name].target == 1).sum())
            fit[name] = anchored_class_weights(frame.target.to_numpy(), anchor)
            saved = pd.read_parquet(shared_runs[seed] / f"expanded_{name}.parquet",
                                    columns=["event_id", feature, "fit_weight"])
            saved = saved[saved[feature] == mass]
            joined = frame[["event_id"]].assign(control=fit[name]).merge(saved, on="event_id", validate="one_to_one")
            if len(joined) != len(frame) or not np.allclose(joined.fit_weight / prior[mass], joined.control,
                                                            rtol=1e-12, atol=0):
                raise ValueError(f"m{mass} seed{seed}: control loss weights differ from the parameterized model's")
        checks.append(dict(check=f"m{mass} seed{seed}: control_matches_parameterized_loss_weights", passed=True))
        model_dir = runs / run_id / "models" / model_dirname(mass)
        model_dir.mkdir(parents=True)
        X_train, y_train, _ = make_xyw(train, features)
        X_val, y_val, _ = make_xyw(val, features)
        model, cost = measured_train_xgb(X_train, y_train, X_val, y_val, seed=seed, **params, record_training=True,
                                         fit_weights_train=fit["train"], fit_weights_val=fit["val"])
        resources.append(dict(stage=run_id, **cost, trees=model.get_booster().num_boosted_rounds()))
        model.get_booster().save_model(model_dir / "model.ubj")
        predictions = {name: frame[IDENTITY].assign(bdt_score=model.predict_proba(X)[:, 1])
                       for name, frame, X in (("train", train, X_train), ("val", val, X_val))}
        for name, frame in predictions.items():
            frame.to_parquet(model_dir / f"preds_{name}.parquet", index=False)
        record = validate_model(full[mass], predictions, cfg, stored_lumi=stored_lumi, **phase2_bootstrap)
        check_yield_closure(full[mass], val, record.weights, cfg, stored_lumi=stored_lumi)
        loss_weights = dict(reference_mass=reference_mass,
                            **{f"{name}_signal_weight": float(fit[name][splits[mass][name].target.to_numpy() == 1][0])
                               for name in fit})
        metrics = dict(mass=mass, rho_tc=settings["rho_tc"], features=features, evaluation=record.metadata,
                       lumi_pb_inv=cfg.lumi_pb_inv, fit_policy=CONTROL_POLICY, loss_weights=loss_weights,
                       test_evaluated=False, xgb=dict(best_iteration=int(model.best_iteration),
                                                      params=model.get_params(), weight_mode=CONTROL_POLICY))
        write_json(model_dir / "metrics.json", metrics)
        write_json(model_dir / "operating_points.json", record.metadata)
        write_json(model_dir / "learning_curves.json", model.evals_result())
        _, _, reload_predict = load_model(model_dir)
        np.testing.assert_allclose(reload_predict(X_val), predictions["val"].bdt_score, rtol=1e-6, atol=2e-7)
        checks.append(dict(check=f"{run_id}: reload_score_closure_and_yield_closure", passed=True))
        write_validation(model_dir, record)
        roc_tables(full[mass], predictions, cfg, stored_lumi=stored_lumi).to_parquet(model_dir / "roc.parquet", index=False)
        write_json(model_dir / "provenance.json", dict(report=str(out), inputs=inputs, preregistration=preregistration))
        controls[mass, seed], control_dirs[mass, seed] = predictions["val"], model_dir
        control_rows += point_rows(record.metadata, mass=mass, run_id=run_id, seed=seed, policy=CONTROL_POLICY)
        runs_record.append(dict(mass=mass, seed=seed, run_id=run_id, model_directory=str(model_dir)))
        print(f"Completed {run_id}: weighted validation AUC={record.metadata['auc_weighted']:.4f}; test sealed", flush=True)

    # Paired comparison on identical validation events, weighted exactly as in Phases 2/3.
    grid = efficiency_grid(settings)
    points = [settings["background_efficiency"], *settings["secondary_background_efficiencies"]]
    groups, metric_rows, point_table, seed_rows, compared = {}, [], [], [], []
    for mass in masses:
        for seed in seeds:
            scored = {"reference": pd.read_parquet(reference_dirs[mass, seed] / "preds_val.parquet"),
                      "candidate": pd.read_parquet(candidate_dirs[mass, seed] / "preds_val.parquet")}
            if (mass, seed) in controls:
                scored["control"] = controls[mass, seed]
            frame = align_predictions(scored)
            checks.append(dict(check=f"m{mass} seed{seed}: identical_validation_events_classes_and_weights",
                               passed=True))
            weights, _ = evaluation_weights(full[mass], frame, stored_lumi=stored_lumi, target_lumi=cfg.lumi_pb_inv,
                                            signal_k_factor=cfg.signal_k_factor,
                                            background_k_factor=cfg.background_k_factor)
            check_yield_closure(full[mass], frame, weights, cfg, stored_lumi=stored_lumi)
            checks.append(dict(check=f"m{mass} seed{seed}: validation_yield_and_k_factor_closure", passed=True))
            models = list(scored)
            groups[mass, seed] = (frame, models, weights)
            y = frame.target.to_numpy()
            totals = {target: float(weights[y == target].sum()) for target in (0, 1)}
            run_ids = dict(reference=reference_dirs[mass, seed].parents[1].name,
                           candidate=f"{shared_runs[seed].name}-m{mass}", control=control_ids.get((mass, seed)))
            for row in paired_metrics(frame, models, weights, background_efficiencies=grid).to_dict("records"):
                scores = frame[row["model"]].to_numpy()
                metric_rows.append(dict(mass=mass, seed=seed, run_id=run_ids[row["model"]], **row,
                                        auc_unweighted=safe_auc(y, scores)))
                for target in points:
                    point = working_point(y, scores, weights, background_efficiency=target)
                    yields = yield_metrics(point["signal_efficiency"] * totals[1], target * totals[0])
                    point_table.append(dict(mass=mass, seed=seed, model=row["model"], **point,
                                            **selection_support(y, scores, weights, point["threshold"]),
                                            **{key: yields[key] for key in ("S", "B", "S_over_B", "Z_Asimov",
                                                                            "Z_syst_5pct", "Z_syst_10pct")}))
            compared.append(dict(mass=mass, seed=seed, reference=str(reference_dirs[mass, seed]),
                                 candidate=str(candidate_dirs[mass, seed]),
                                 control=str(control_dirs[mass, seed]) if (mass, seed) in control_dirs else None))
        # Every trained mass-specific seed, including seeds outside the comparison (descriptive).
        frame, _, weights = groups[mass, seeds[0]]
        for (m, other_seed), directory in sorted(reference_dirs.items()):
            if m != mass:
                continue
            other = align_predictions({"first": frame[["event_id", "sample", "target", "sample_weight"]].assign(
                bdt_score=frame["reference"]), "seed": pd.read_parquet(directory / "preds_val.parquet")})
            spread = paired_metrics(other, ["seed"], weights, background_efficiencies=points).iloc[0].to_dict()
            seed_rows.append(dict(mass=mass, seed=other_seed, run_id=directory.parents[1].name,
                                  **{k: v for k, v in spread.items() if k != "model"}))
    print(f"Paired bootstrap: {settings['bootstrap_replicates']} replicas shared across models, seeds and masses",
          flush=True)
    replicas = joint_paired_bootstrap(groups, background_efficiencies=grid, replicates=settings["bootstrap_replicates"],
                                      seed=settings["bootstrap_seed"])
    replicas.insert(0, "mass", [key[0] for key in replicas.group])
    replicas.insert(1, "seed", [key[1] for key in replicas.group])
    replicas = replicas.drop(columns="group")

    # Non-inferiority per mass and seed; a mass passes only if every seed does.
    margins = settings["noninferiority"]
    confidence = margins["one_sided_confidence"]
    primary = f"signal_efficiency_at_{settings['background_efficiency']:g}"
    criteria = [(primary, "primary", True, margins["max_relative_signal_efficiency_loss"]),
                ("auc_weighted", "primary", False, margins["max_weighted_auc_loss"])]
    criteria += [(f"signal_efficiency_at_{x:g}", "secondary", True, margins["max_relative_signal_efficiency_loss"])
                 for x in settings["secondary_background_efficiencies"]]
    contrasts = [("reference", "candidate", "gate"),
                 ("reference", "control", "descriptive: loss weighting only"),
                 ("control", "candidate", "descriptive: parameterization at matched loss weights")]
    estimates = pd.DataFrame(metric_rows)
    tests, seed_gate, gate_rows = [], [], []
    for mass in masses:
        for seed in seeds:
            point = estimates[(estimates.mass == mass) & (estimates.seed == seed)].set_index("model")
            replica = replicas[(replicas.mass == mass) & (replicas.seed == seed)]
            for baseline, compared_model, purpose in contrasts:
                if compared_model not in point.index or baseline not in point.index:
                    continue
                for metric, role, relative, margin in criteria:
                    wide = replica.pivot(index="replicate", columns="model", values=metric)
                    result = noninferiority(point.loc[baseline, metric], point.loc[compared_model, metric],
                                            wide[baseline], wide[compared_model], margin=margin, relative=relative,
                                            confidence=confidence)
                    tests.append(dict(mass=mass, seed=seed, baseline=baseline, compared=compared_model,
                                      purpose=purpose, metric=metric, role=role if purpose == "gate" else "descriptive",
                                      baseline_value=point.loc[baseline, metric],
                                      compared_value=point.loc[compared_model, metric], **result))
            gate = [t for t in tests if (t["mass"], t["seed"]) == (mass, seed)
                    and t["purpose"] == "gate" and t["role"] == "primary"]
            seed_gate.append(dict(mass=mass, seed=str(seed), **{t["metric"]: t["outcome"] for t in gate},
                                  outcome=combine_outcomes(t["outcome"] for t in gate)))
        per_seed = [row["outcome"] for row in seed_gate if row["mass"] == mass]
        gate_rows.append(dict(mass=mass, seed="all", outcome=combine_outcomes(per_seed)))
    overall = combine_outcomes(row["outcome"] for row in gate_rows)

    own = [pd.read_csv(reference_report / "tables/operating_points.csv").assign(model="reference")]
    own += [pd.read_csv(report / "tables/operating_points.csv").assign(model="candidate")
            for report in parameterized_reports.values()]
    own = pd.concat(own, ignore_index=True)
    own = own[own.run_id.isin([reference_dirs[m, s].parents[1].name for m in masses for s in seeds]
                              + [f"{shared_runs[s].name}-m{m}" for m in masses for s in seeds])]
    if control_rows:
        own = pd.concat([own, pd.DataFrame(control_rows).assign(model="control")], ignore_index=True)
    if "cut_reference" in settings["controls"]:
        cuts = pd.read_csv(reference_report / "tables/cut_operating_points.csv")
        own = pd.concat([own, cuts[cuts.mass.isin(masses)].assign(model="cut_reference")], ignore_index=True)
    written = [p for name in control_ids.values() for p in (runs / name).rglob("*preds_test*")]
    read = [p for p in inputs if Path(p).name.startswith("preds_test")]
    checks.append(dict(check="no_test_scores_read_or_written", passed=not written and not read))
    loss_weights = loss_weight_table(masses, splits, shared_runs[seeds[0]], feature, reference_mass, prior,
                                     {mass for mass, _ in controls})
    tables = dict(checks=pd.DataFrame(checks), loss_weights=loss_weights,
                  paired_metrics=estimates, working_points=pd.DataFrame(point_table),
                  bootstrap_replicas=replicas, noninferiority=pd.DataFrame(tests),
                  gate=pd.DataFrame(seed_gate + gate_rows),
                  own_operating_points=own, reference_seed_spread=pd.DataFrame(seed_rows),
                  resources=pd.DataFrame(resources, columns=["stage", "seconds", "rss_before_bytes",
                                                             "peak_sampled_rss_bytes", "peak_rss_delta_bytes",
                                                             "sampling_interval_seconds", "trees"]))
    write_tables(out, tables)
    if not tables["checks"].passed.all():
        raise ValueError("Paired-comparison contract checks failed")
    summary = dict(schema_version=2, phases=[5], runs=runs_record, compared=compared, settings=settings,
                   evaluation_config=asdict(cfg), reference_report=str(reference_report),
                   parameterized_reports={str(s): str(p) for s, p in parameterized_reports.items()},
                   control_policy=CONTROL_POLICY, fit_parameters=params,
                   seed_rule=SEED_RULE if len(seeds) > 1 else None,
                   gate={str(row["mass"]): row["outcome"] for row in gate_rows},
                   gate_by_seed={str(m): {r["seed"]: r["outcome"] for r in seed_gate if r["mass"] == m}
                                 for m in masses},
                   outcome=overall, test_evaluated=False, status="checkpoint_ready",
                   note="Validation-only paired comparison at the pre-registered working point. "
                        "Test remains sealed; interpolation to unseen masses is a later gate.")
    write_json(out / "provenance/summary.json", summary)
    save_comparison_figures(out)
    artifacts = [p for record in runs_record for p in Path(record["model_directory"]).glob("*") if p.is_file()]
    finish_report(out, "03_parameterized_bdt.ipynb", inputs, artifacts)
    print(f"Phase 5 outcome: {overall} ({summary['gate_by_seed']})", flush=True)


def loss_weight_table(masses, splits, shared_run, feature, reference_mass, prior, control_masses):
    """Training rows and class-loss totals each model used at each compared mass (seed-independent)."""
    rows = []
    shared = pd.read_parquet(shared_run / "expanded_train.parquet", columns=["event_id", feature, "fit_weight"])
    for mass in masses:
        train = splits[mass]["train"]
        n_signal, n_background = int((train.target == 1).sum()), int((train.target == 0).sum())
        at_mass = shared[shared[feature] == mass].merge(train[["event_id", "target"]], on="event_id")
        models = {"reference": (n_signal, n_background),
                  "candidate": (at_mass.loc[at_mass.target == 1, "fit_weight"].sum() / prior[mass],
                                at_mass.loc[at_mass.target == 0, "fit_weight"].sum() / prior[mass])}
        if mass in control_masses:
            anchor = int((splits[reference_mass]["train"].target == 1).sum())
            models["control"] = (float(anchored_class_weights(train.target.to_numpy(), anchor)[train.target == 1].sum()),
                                 n_background)
        for model, (signal, background) in models.items():
            rows.append(dict(mass=mass, model=model, signal_events=n_signal, background_events=n_background,
                             signal_loss_total=float(signal), background_loss_total=float(background),
                             signal_to_background_loss=float(signal) / float(background),
                             unique_training_events=int(shared.event_id.nunique()) if model == "candidate"
                             else n_signal + n_background))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
