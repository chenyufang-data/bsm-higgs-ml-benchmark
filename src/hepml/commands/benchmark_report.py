"""Method benchmark for presentation: cut-based selection, BDT, LorentzNet and ParT at chosen points.

Reads completed reports and saved models only (the truth-tagged grid and both Phase 10a pilots);
no model is refitted for the results. Everything is on validation events at the grid's training
coupling. Significances are reported relative to the cut-based selection, never as absolute
values, and no event yields are written. The test partition stays sealed.

Contents per mass: Z_Asimov at each method's own validation optimum relative to the cut-based
optimum; weighted AUC; ROC curves; SHAP values of the BDT's features; permutation importance of
the twelve raw role-jet inputs (pT, eta, phi, mass of b1, b2, c1) for all three models, the
common ground for comparing them; parameters, fit time (networks from their pilots; one timed
BDT refit, not used for any result) and inference time.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hepml.adapters.configuration import default_study_directory, fit_parameters, load_analysis
from hepml.adapters.study_loader import load_study
from hepml.adapters.study_run import (
    completed_report,
    finish_report,
    hash_files,
    open_report,
    refuse_existing,
    run_study,
    write_json,
    write_tables,
)
from hepml.adapters.truth_tag_datasets import TruthTagBenchmark
from hepml.application.evaluation import evaluate_validation, evaluation_weights
from hepml.application.training import make_xyw
from hepml.commands.binned_fit_study import IDENTITY
from hepml.domain.metrics import background_efficiency_at

KINEMATICS = ("pt", "eta", "phi", "mass")
ROLES = ("b1", "b2", "c1")
RAW_INPUTS = [f"{q}{role}" for role in ROLES for q in KINEMATICS]
SIGNAL_EFFICIENCIES = np.round(np.linspace(0.05, 0.95, 19), 2)
NETWORKS = {"lorentznet": "LorentzNet", "particle_transformer": "ParT"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study")
    parser.add_argument("--grid-report", required=True)
    parser.add_argument("--lorentznet-report", required=True)
    parser.add_argument("--part-report", required=True)
    parser.add_argument("--masses", type=int, nargs="+", default=[200, 400])
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--shap-events", type=int, default=4000, help="Per class, for the BDT's SHAP values")
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--device", help="cuda or cpu; default: cuda when available")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    run_study(args.outdir, _run, args)


def _scores(path, key, column):
    saved = pd.read_parquet(path)
    return key.merge(saved[["event_id", "sample_key", column]], on=["event_id", "sample_key"],
                     validate="one_to_one")[column].to_numpy(dtype=float)


def _auc(target, score, weights):
    return float(roc_auc_score(target, score, sample_weight=weights))


def _run(args):
    import shap
    import torch
    import xgboost as xgb

    from hepml.adapters.benchmark_report import save_benchmark_figures, write_benchmark_markdown
    from hepml.adapters.inference import load_model
    from hepml.adapters.jet_networks import jet_nodes, load_network, predict
    from hepml.adapters.xgboost_model import train_xgb

    study = load_study(args.study or default_study_directory())
    analysis = load_analysis(study.directory / "analysis.yaml")
    cfg = analysis.evaluation
    grid_report = Path(args.grid_report).resolve()
    grid = completed_report(grid_report, [9])
    pilots = {"lorentznet": Path(args.lorentznet_report).resolve(),
              "particle_transformer": Path(args.part_report).resolve()}
    for name, report in pilots.items():
        summary = completed_report(report, [10])
        if Path(summary["grid_report"]).resolve() != grid_report or summary["design"]["primary"]["compare"] != name:
            raise ValueError(f"{report} is not the {name} pilot on this grid")
    if not set(args.masses) <= set(summary["design"]["masses"]):
        raise ValueError("Every mass needs both network pilots")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    training = float(grid["settings"]["training_coupling"])
    directory = {float(k): v for k, v in grid["settings"]["coupling_directories"].items()}[training]
    seeds = summary["design"]["seeds"]
    bench = TruthTagBenchmark(grid["dataset"])
    bdt_runs = {(r["mass"], r["seed"]): Path(r["model_directory"]) for r in grid["runs"]}
    network_runs, network_designs = {}, {}
    for name, report in pilots.items():
        frozen = json.loads((report / "provenance/preregistration.json").read_text())
        runs = Path(frozen["arguments"]["runs_dir"]).resolve()
        fits = pd.read_csv(report / "tables/fits.csv")
        network_runs.update({(name, r.mass, r.seed): runs / r.run_id for r in fits[fits.model == name].itertuples()})
        network_designs[name] = frozen["design"]
    out = Path(args.outdir).resolve()
    refuse_existing(out)
    features = list(study.plugin.FEATURES)
    columns = list(dict.fromkeys([*IDENTITY, "fit_weight", *RAW_INPUTS, *features]))  # pT and eta are both

    paths = [study.directory / "analysis.yaml", grid_report / "tables/grid.csv",
             *(r / "provenance" / n for r in (grid_report, *pilots.values()) for n in ("summary.json", "status.json")),
             *(r / "tables/primary.csv" for r in pilots.values()), *(r / "tables/fits.csv" for r in pilots.values()),
             bench.background_path("val"), bench.background_path("train")]
    for mass in args.masses:
        paths += [bench.meta_path(directory, mass), bench.signal_path(directory, mass, "val"),
                  bench.signal_path(directory, mass, "train"),
                  *(bdt_runs[mass, s] / name for s in seeds for name in ("model.ubj", "metrics.json")),
                  *(bdt_runs[mass, s] / "points" / directory / "preds_val.parquet" for s in seeds),
                  *(network_runs[n, mass, s] / f for n in NETWORKS for s in seeds
                    for f in ("model.pt", "network.json", f"preds_val_{directory}.parquet"))]
    inputs = hash_files(sorted(set(paths)))
    open_report(out, dict(study=str(study.directory), arguments=vars(args), evaluation_config=asdict(cfg),
                          grid_report=str(grid_report), pilots={k: str(v) for k, v in pilots.items()},
                          torch=torch.__version__, shap=shap.__version__, xgboost=xgb.__version__,
                          device=device, test_evaluated=False), inputs)
    checks = [dict(check="grid and both network pilots complete, unchanged and on the same grid", passed=True)]
    grid_table = pd.read_csv(grid_report / "tables/grid.csv")
    rng = np.random.default_rng(args.seed)
    rows, roc_rows, shap_rows, permutation_rows, cost_rows = [], [], [], [], []

    for mass in args.masses:
        frame = bench.split(directory, mass, "val", columns)
        full, lumi = bench.totals(directory, mass), float(bench.meta(directory, mass)["lumi"])
        weights, _ = evaluation_weights(full, frame, stored_lumi=lumi, target_lumi=cfg.lumi_pb_inv,
                                        signal_k_factor=cfg.signal_k_factor, background_k_factor=cfg.background_k_factor)
        target, key = frame.target.to_numpy(), frame[["event_id", "sample_key"]]

        # The cut-based selection at its own validation optimum, as the grid chose it.
        cut = grid_table[(grid_table.method == "cuts") & (grid_table.mass == mass)
                         & np.isclose(grid_table.rho_tc, training) & (grid_table.objective == "Z_Asimov")].iloc[0]
        baseline = float(cut.Z_Asimov)
        rows.append(dict(mass=mass, method="cut_based", seed=None, z_asimov_ratio=1.0, auc=np.nan,
                         signal_efficiency=float(cut.signal_efficiency),
                         background_efficiency=float(cut.background_efficiency)))

        scores = {("bdt", s): _scores(bdt_runs[mass, s] / "points" / directory / "preds_val.parquet", key, "bdt_score")
                  for s in seeds}
        scores.update({(n, s): _scores(network_runs[n, mass, s] / f"preds_val_{directory}.parquet", key, "score")
                       for n in NETWORKS for s in seeds})
        for (method, seed), score in scores.items():
            point = evaluate_validation(full, frame.assign(bdt_score=score), cfg, stored_lumi=lumi)[0]
            metrics = point["operating_points"]["Z_Asimov"]["metrics"]
            rows.append(dict(mass=mass, method=method, seed=seed, z_asimov_ratio=metrics["Z_Asimov"] / baseline,
                             auc=_auc(target, score, weights), signal_efficiency=metrics["signal_efficiency"],
                             background_efficiency=metrics["background_efficiency"]))
            for efficiency in SIGNAL_EFFICIENCIES:
                roc_rows.append(dict(mass=mass, method=method, seed=seed, signal_efficiency=efficiency,
                                     background_efficiency=background_efficiency_at(score, target, weights, efficiency)[0]))

        # SHAP of the primary-seed BDT on a class-balanced validation sample of its own features.
        seed = grid["primary_seed"]
        model_dir = bdt_runs[mass, seed]
        _, _, bdt_predict = load_model(model_dir)
        best = json.loads((model_dir / "metrics.json").read_text())["xgb"]["best_iteration"]
        booster = xgb.Booster()
        booster.load_model((model_dir / "model.ubj").as_posix())
        X = make_xyw(frame, features)[0]
        closure = float(np.max(np.abs(bdt_predict(X) - scores["bdt", seed])))
        checks.append(dict(check=f"m{mass}: the reloaded BDT reproduces its saved validation scores (max {closure:.1e})",
                           passed=closure < 1e-6))
        chosen = np.concatenate([rng.choice(np.flatnonzero(target == label),
                                            min(args.shap_events, int((target == label).sum())), replace=False)
                                 for label in (0, 1)])
        values = shap.TreeExplainer(booster[: best + 1]).shap_values(X[chosen])
        shap_rows += [dict(mass=mass, feature=f, mean_abs_shap=float(np.abs(values[:, i]).mean()))
                      for i, f in enumerate(features)]

        # Permutation importance of the raw role-jet inputs, the same for every model.
        def bdt_score(table):
            objects = {role: tuple(table[f"{q}{role}"].to_numpy() for q in KINEMATICS) for role in ROLES}
            derived = study.plugin.features_from_objects(objects)
            return bdt_predict(make_xyw(derived.assign(target=table.target.to_numpy()), features)[0]).astype(float)

        rebuilt = float(np.max(np.abs(bdt_score(frame) - scores["bdt", seed])))
        checks.append(dict(check=f"m{mass}: BDT features rebuilt from the raw role-jet inputs reproduce its scores "
                                 f"(max {rebuilt:.1e})", passed=rebuilt < 1e-5))
        scorers = {"bdt": bdt_score}
        for name in NETWORKS:
            network, _ = load_network(network_runs[name, mass, seed], device=device)
            nodes = network_designs[name]["nodes"]

            def network_score(table, network=network, nodes=nodes):
                return predict(network, *jet_nodes(table, unit_gev=nodes["momentum_unit_gev"], beams=nodes["beams"]),
                               device=device)

            scorers[name] = network_score
            rebuilt = float(np.max(np.abs(network_score(frame) - scores[name, seed])))
            checks.append(dict(check=f"m{mass}: the reloaded {name} reproduces its saved scores (max {rebuilt:.1e})",
                               passed=rebuilt < 1e-5))
        for method, scorer in scorers.items():
            nominal = _auc(target, scores[method, seed], weights)
            started = time.perf_counter()
            scorer(frame)
            cost_rows.append(dict(mass=mass, method=method, inference_ms_per_1k_events=1e3 * (time.perf_counter() - started)
                                  / len(frame) * 1e3, device="cpu" if method == "bdt" else device))
            for column in RAW_INPUTS:
                drops = []
                for _ in range(args.permutation_repeats):
                    shuffled = frame.copy()
                    shuffled[column] = rng.permutation(shuffled[column].to_numpy())
                    drops.append(nominal - _auc(target, scorer(shuffled), weights))
                permutation_rows.append(dict(mass=mass, method=method, input=column, auc_drop=float(np.mean(drops)),
                                             auc_drop_spread=float(np.std(drops))))
        print(f"m{mass}: done", flush=True)

    # Cost: parameters and fit time. Networks from their pilots; the BDT from one timed refit.
    fits = pd.concat([pd.read_csv(r / "tables/fits.csv").assign(pilot=n) for n, r in pilots.items()])
    for name in NETWORKS:
        rows_n = fits[fits.model == name]
        cost_rows.append(dict(method=name, parameters=int(rows_n.parameters.iloc[0]),
                              fit_seconds=float(rows_n.seconds.mean()), fit_device=device, best_epoch=float(rows_n.best_epoch.mean())))
    mass = args.masses[0]
    train = bench.split(directory, mass, "train", columns)
    val = bench.split(directory, mass, "val", columns)
    started = time.perf_counter()
    model = train_xgb(*make_xyw(train, features)[:2], *make_xyw(val, features)[:2], seed=grid["primary_seed"],
                      **fit_parameters(analysis), fit_weights_train=train.fit_weight.to_numpy(float),
                      fit_weights_val=val.fit_weight.to_numpy(float))
    trees = int(model.best_iteration) + 1
    cost_rows.append(dict(method="bdt", parameters=trees * (2 ** (fit_parameters(analysis)["max_depth"] + 1) - 1),
                          parameter_note=f"{trees:,} trees, depth {fit_parameters(analysis)['max_depth']}",
                          fit_seconds=time.perf_counter() - started, fit_device="cpu"))
    checks.append(dict(check="no test rows read", passed=not any("test" in Path(p).name for p in inputs)))

    tables = dict(checks=pd.DataFrame(checks), methods=pd.DataFrame(rows), roc=pd.DataFrame(roc_rows),
                  shap=pd.DataFrame(shap_rows), permutation=pd.DataFrame(permutation_rows),
                  costs=pd.DataFrame(cost_rows),
                  paired_auc=pd.concat([pd.read_csv(r / "tables/primary.csv").assign(pilot=n)
                                        for n, r in pilots.items()], ignore_index=True))
    write_tables(out, tables)
    if not tables["checks"].passed.all():
        raise ValueError("Benchmark checks failed: " + "; ".join(tables["checks"].loc[~tables["checks"].passed, "check"]))
    write_json(out / "provenance/summary.json", dict(
        schema_version=2, phases=[10], stage="benchmark", masses=args.masses, coupling=training, seeds=seeds,
        grid_report=str(grid_report), pilots={k: str(v) for k, v in pilots.items()}, test_evaluated=False,
        status="complete", note="Validation events; significances relative to the cut-based selection only."))
    save_benchmark_figures(out)
    write_benchmark_markdown(out)
    finish_report(out, None, inputs)
