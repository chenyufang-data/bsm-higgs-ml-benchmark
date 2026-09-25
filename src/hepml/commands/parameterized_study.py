"""Phase 3: one two-mass classifier with explicit priors and conditional inference."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from hepml_compact.parquet_writer import sha256_file
from sklearn.metrics import log_loss

from hepml.adapters.configuration import default_study_directory, fit_parameters, load_analysis
from hepml.adapters.inference import load_conditional_model
from hepml.adapters.parameterized_report import save_parameterized_figures
from hepml.adapters.study_loader import load_study
from hepml.adapters.study_run import (
    finish_report,
    open_report,
    refuse_existing,
    run_study,
    write_json,
    write_tables,
    write_validation,
)
from hepml.adapters.xgboost_model import measured_train_xgb, train_xgb
from hepml.application.conditioning import (
    anchored_class_weights,
    assert_split_isolation,
    condition_frame,
    expand_hypotheses,
)
from hepml.application.evaluation import (
    check_yield_closure,
    evaluation_weights,
    point_rows,
    roc_tables,
    selected_support,
    validate_model,
)
from hepml.application.training import make_xyw
from hepml.domain.artifacts import BenchmarkFiles
from hepml.domain.metrics import safe_auc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study")
    parser.add_argument(
        "--reference-report", required=True, help="Completed Phase 2 report with immutable dataset hashes"
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--run-dir", required=True, help="New shared model run; no model copies per mass")
    parser.add_argument("--seed", type=int, help="Training seed (default: parameterized_study.seed)")
    parser.add_argument("--training-override", action="append", default=[], metavar="KEY=VALUE",
                        help="Recorded change to one analysis.yaml training setting; must match the Phase 2 fits")
    args = parser.parse_args(argv)
    run_study(args.outdir, _run, args)


def _run(args):
    study = load_study(args.study or default_study_directory())
    analysis = load_analysis(study.directory / "analysis.yaml")
    cfg = analysis.evaluation
    settings = yaml.safe_load((study.directory / "validation.yaml").read_text())["parameterized_study"]
    if args.seed is not None:
        settings = dict(settings, seed=args.seed)  # recorded in the preregistration and summary
    masses, reference_mass = settings["masses"], settings["reference_mass"]
    if (
        len(masses) < 2
        or len(set(masses)) != len(masses)
        or reference_mass not in masses
        or settings["fit_policy"] != "reference-unweighted"
        or settings["hypothesis_prior"] != "uniform"
        or settings["background_assignment"] != "replicate"
        or settings["bootstrap_replicates"] < 2
        or settings["learning_curve_step"] < 1
    ):
        raise ValueError("Unsupported conditional-study design")
    reference_report = Path(args.reference_report).resolve()
    previous = json.loads((reference_report / "provenance/summary.json").read_text())
    status = json.loads((reference_report / "provenance/status.json").read_text())
    if status["status"] != "complete" or previous["phases"] != [2]:
        raise ValueError("Require a completed Phase 2 reference study")
    if (
        previous["settings"]["fit_policy"] != "none"
        or previous["settings"]["rho_tc"] != settings["rho_tc"]
        or set(previous["settings"]["masses"]) != set(masses)
    ):
        raise ValueError("Conditional design differs from the approved reference inputs/policy")
    recorded_cfg = dict(previous["evaluation_config"])
    recorded_cfg["background_systematics"] = tuple(recorded_cfg["background_systematics"])
    current_cfg = dict(asdict(cfg), background_systematics=tuple(cfg.background_systematics))
    if recorded_cfg != current_cfg:
        raise ValueError("Evaluation settings differ from Phase 2")
    for name, digest in status["artifacts"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"Reference artifact changed: {name}")
    params = fit_parameters(analysis, args.training_override)
    for record in previous["runs"]:
        recorded = json.loads((Path(record["model_directory"]) / "metrics.json").read_text())["xgb"]["params"]
        if any(recorded.get(key) != value for key, value in params.items()):
            raise ValueError("Hyperparameters differ from the Phase 2 reference fits; use the same --training-override")
    out, run = Path(args.outdir).resolve(), Path(args.run_dir).resolve()
    refuse_existing(out, run)
    datasets = BenchmarkFiles(Path(previous["dataset"]))
    features = list(study.plugin.FEATURES)
    conditioning = dict(
        schema_version=1,
        feature=settings["hypothesis_feature"],
        unit=settings["hypothesis_unit"],
        support=masses,
        prior={str(m): 1 / len(masses) for m in masses},
        physics_features=features,
        fixed_rho_tc=settings["rho_tc"],
        reference_mass=reference_mass,
        fit_policy=settings["fit_policy"],
        class_totals="unweighted_reference_mass_split_counts",
        background_assignment="replicate",
        interpolation_validated=False,
    )
    model_features = [*features, conditioning["feature"]]
    full, by_split, test_ids, inputs = {}, {"train": {}, "val": {}}, set(), {}
    stored_lumi = None
    # Identity/normalization aggregates may include test; no test observables or scores are used.
    for mass in masses:
        meta_path = datasets.meta(mass)
        meta = json.loads(meta_path.read_text())
        if meta["features"] != features or (stored_lumi is not None and stored_lumi != meta["lumi"]):
            raise ValueError("Reference feature or stored-luminosity contract changed")
        stored_lumi = float(meta["lumi"])
        path = datasets.dataset(mass)
        full[mass] = pd.read_parquet(path, columns=["event_id", "sample_key", "sample", "target", "sample_weight"])
        inputs[str(path)] = sha256_file(path)
        inputs[str(meta_path)] = sha256_file(meta_path)
        assignment_path = datasets.assignments(mass)
        assignments = pd.read_parquet(assignment_path)
        test_ids.update(assignments.loc[assignments.split == "test", "event_id"])
        inputs[str(assignment_path)] = sha256_file(assignment_path)
        for name in by_split:
            path = datasets.split(name, mass)
            frame = pd.read_parquet(path)
            if set(frame.event_id) != set(assignments.loc[assignments.split == name, "event_id"]):
                raise ValueError("Reference split assignments changed")
            by_split[name][mass] = frame
            inputs[str(path)] = sha256_file(path)
    for path in [reference_report / "provenance/status.json", reference_report / "provenance/summary.json"]:
        inputs[str(path)] = sha256_file(path)
    preregistration = dict(
        study=str(study.directory),
        arguments=vars(args),
        analysis=asdict(analysis),
        settings=settings,
        conditioning=conditioning,
        reference_report=str(reference_report),
        fit_parameters=params,
        test_evaluated=False,
    )
    open_report(out, preregistration, inputs, preregister=True)
    run.mkdir(parents=True)
    pooled, audits = {}, []
    for name, frames in by_split.items():
        pooled[name], audit = expand_hypotheses(
            frames, physics_features=features, conditioning=conditioning, reference_mass=reference_mass, split=name
        )
        audits.append(audit)
        pooled[name][
            [
                "event_id",
                "sample_key",
                "hypothesis_id",
                conditioning["feature"],
                "generated_mass",
                "target",
                "split",
                "sample_weight",
                "fit_weight",
            ]
        ].to_parquet(run / f"expanded_{name}.parquet", index=False)
    assert_split_isolation(pooled["train"], pooled["val"], test_ids)
    pd.concat(audits, ignore_index=True).to_csv(out / "tables/hypothesis_priors.csv", index=False)
    populations = [
        dict(
            split=name,
            unique_events=frame.event_id.nunique(),
            expanded_rows=len(frame),
            unique_background=frame.loc[frame.target == 0, "event_id"].nunique(),
            fit_weight_sum=frame.fit_weight.sum(),
            reference_rows=len(by_split[name][reference_mass]),
        )
        for name, frame in pooled.items()
    ]
    pd.DataFrame(populations).to_csv(out / "tables/populations.csv", index=False)
    checks = [
        dict(check="class_conditional_hypothesis_priors_and_replica_conservation", passed=True),
        dict(check="physical_event_split_isolation", passed=True),
    ]
    X_train, y_train, _ = make_xyw(pooled["train"], model_features)
    X_val, y_val, _ = make_xyw(pooled["val"], model_features)
    w_train, w_val = pooled["train"].fit_weight.to_numpy(), pooled["val"].fit_weight.to_numpy()
    seed = settings["seed"]
    probe = train_xgb(
        X_train[:, -1:],
        y_train,
        X_val[:, -1:],
        y_val,
        seed=seed,
        max_depth=2,
        n_estimators=settings["probe_estimators"],
        early_stopping_rounds=5,
        fit_weights_train=w_train,
        fit_weights_val=w_val,
    )
    probe_rows = []
    for name, X, y, w in [("train", X_train, y_train, w_train), ("val", X_val, y_val, w_val)]:
        scores = probe.predict_proba(X[:, -1:])[:, 1]
        auc = safe_auc(y, scores, w)
        probe_rows.append(dict(split=name, weighted_auc=auc, raw_auc=safe_auc(y, scores)))
        if not np.isclose(auc, 0.5, atol=1e-10, rtol=0):
            raise ValueError("Mass-only classifier detects weighted prior leakage")
    pd.DataFrame(probe_rows).to_csv(out / "tables/mass_only_probe.csv", index=False)
    checks.append(dict(check="mass_only_weighted_auc_is_chance", passed=True))
    fit_options = dict(seed=seed, record_training=True, fit_weights_train=w_train, fit_weights_val=w_val)
    pilot, pilot_cost = measured_train_xgb(
        X_train, y_train, X_val, y_val, **dict(params, n_estimators=settings["pilot_estimators"]), **fit_options
    )
    costs = [dict(stage="resource_pilot", **pilot_cost, trees=pilot.get_booster().num_boosted_rounds())]
    del pilot
    print("Conditioning checks passed; resource pilot complete; fitting one shared model", flush=True)
    model, cost = measured_train_xgb(X_train, y_train, X_val, y_val, **params, **fit_options)
    costs.append(dict(stage="shared_model", **cost, trees=model.get_booster().num_boosted_rounds()))
    pd.DataFrame(costs).to_csv(out / "tables/resources.csv", index=False)
    model.get_booster().save_model(run / "model.ubj")
    write_json(run / "learning_curves.json", model.evals_result())
    metrics = dict(
        features=model_features,
        conditioning=conditioning,
        lumi_pb_inv=cfg.lumi_pb_inv,
        fit_policy=settings["fit_policy"],
        test_evaluated=False,
        evaluations={},
        xgb=dict(
            best_iteration=int(model.best_iteration), params=model.get_params(), weight_mode=settings["fit_policy"]
        ),
    )
    write_json(run / "metrics.json", metrics)
    _, reload_score = load_conditional_model(run)
    rows, seed_rows, support_rows, records, learning_rows, bg_scores = [], [], [], [], [], {}
    total_trees = model.get_booster().num_boosted_rounds()
    iterations = sorted(
        set(
            [
                1,
                int(model.best_iteration) + 1,
                total_trees,
                *range(settings["learning_curve_step"], total_trees, settings["learning_curve_step"]),
            ]
        )
    )
    for mass in masses:
        evaluation_dir = run / f"m{mass}"
        evaluation_dir.mkdir()
        prediction = {}
        for name, frames in by_split.items():
            frame = frames[mass]
            conditioned = condition_frame(frame, mass, conditioning)
            X, y, _ = make_xyw(conditioned, model_features)
            scores = model.predict_proba(X)[:, 1]
            np.testing.assert_allclose(reload_score(frame, mass), scores, rtol=1e-6, atol=2e-7)
            prediction[name] = frame[["event_id", "sample_key", "sample", "target", "sample_weight"]].assign(
                bdt_score=scores
            )
            prediction[name].to_parquet(evaluation_dir / f"preds_{name}.parquet", index=False)
            physical, _ = evaluation_weights(
                full[mass],
                prediction[name],
                stored_lumi=stored_lumi,
                target_lumi=cfg.lumi_pb_inv,
                signal_k_factor=cfg.signal_k_factor,
                background_k_factor=cfg.background_k_factor,
            )
            reference_signal_count = (frames[reference_mass].target == 1).sum()
            fit = anchored_class_weights(y, reference_signal_count)
            for count in iterations:
                p = model.predict_proba(X, iteration_range=(0, count))[:, 1]
                learning_rows.append(
                    dict(
                        mass=mass,
                        split=name,
                        trees=count,
                        auc_weighted=safe_auc(y, p, physical),
                        auc_unweighted=safe_auc(y, p),
                        fit_logloss=log_loss(y, p, sample_weight=fit, labels=[0, 1]),
                    )
                )
        checks.append(dict(check=f"m{mass}: conditional_model_reload_closure", passed=True))
        roc_tables(full[mass], prediction, cfg, stored_lumi=stored_lumi).to_parquet(
            evaluation_dir / "roc.parquet", index=False
        )
        record = validate_model(
            full[mass],
            prediction,
            cfg,
            stored_lumi=stored_lumi,
            replicates=settings["bootstrap_replicates"],
            seed=settings["bootstrap_seed"],
        )
        evaluation = record.metadata
        check_yield_closure(full[mass], prediction["val"], record.weights, cfg, stored_lumi=stored_lumi)
        checks.append(dict(check=f"m{mass}: unique_event_physics_rate_closure", passed=True))
        write_validation(evaluation_dir, record)
        write_json(
            evaluation_dir / "metrics.json",
            dict(evaluation=evaluation, shared_model_directory=str(run), xgb=metrics["xgb"]),
        )
        metrics["evaluations"][str(mass)] = evaluation
        for objective, table in selected_support(record, prediction["val"], cfg, stored_lumi=stored_lumi):
            support_rows.append(table.assign(mass=mass, objective=objective))
        run_id = f"{run.name}-m{mass}"
        rows.extend(point_rows(evaluation, mass=mass, seed=seed, run_id=run_id, policy=settings["fit_policy"]))
        seed_rows.append(
            dict(
                mass=mass,
                seed=seed,
                auc_weighted=evaluation["auc_weighted"],
                auc_unweighted=evaluation["auc_unweighted"],
            )
        )
        records.append(dict(mass=mass, seed=seed, run_id=run_id, model_directory=str(evaluation_dir)))
        bg_scores[mass] = (
            prediction["val"]
            .loc[prediction["val"].target == 0, ["event_id", "bdt_score"]]
            .set_index("event_id")
            .sort_index()
        )
        print(
            f"m{mass}: weighted validation AUC={evaluation['auc_weighted']:.4f}; same shared model, test sealed",
            flush=True,
        )
    write_json(run / "metrics.json", metrics)
    write_json(run / "conditioning.json", conditioning)
    write_json(run / "provenance.json", dict(report=str(out), inputs=inputs, preregistration=preregistration))
    shift = np.abs(bg_scores[masses[0]].bdt_score.to_numpy() - bg_scores[masses[1]].bdt_score.to_numpy())
    effect = dict(
        masses=masses[:2],
        common_background_events=len(shift),
        mean_absolute_score_shift=float(shift.mean()),
        median_absolute_score_shift=float(np.median(shift)),
        mass_split_count=int(
            model.get_booster()[: int(model.best_iteration) + 1]
            .get_score(importance_type="weight").get(f"f{len(features)}", 0)
        ),
    )
    write_json(out / "provenance/conditioning_effect.json", effect)
    tables = dict(
        checks=pd.DataFrame(checks),
        operating_points=pd.DataFrame(rows),
        seed_comparison=pd.DataFrame(seed_rows),
        process_support=pd.concat(support_rows, ignore_index=True)
        if support_rows
        else pd.DataFrame(columns=["mass", "objective"]),
        per_mass_learning=pd.DataFrame(learning_rows),
    )
    write_tables(out, tables)
    summary = dict(
        schema_version=2,
        phases=[3],
        runs=records,
        shared_model_directory=str(run),
        settings=settings,
        conditioning=conditioning,
        evaluation_config=asdict(cfg),
        fit_parameters=params,
        training_overrides=args.training_override,
        reference_report=str(reference_report),
        test_evaluated=False,
        status="checkpoint_ready",
        note="Conditioning and reload contracts passed. Paired comparison, additional seeds and interpolation remain later gates.",
    )
    write_json(out / "provenance/summary.json", summary)
    save_parameterized_figures(out)
    finish_report(out, "03_parameterized_bdt.ipynb", inputs, [p for p in run.rglob("*") if p.is_file()])


if __name__ == "__main__":
    main()
