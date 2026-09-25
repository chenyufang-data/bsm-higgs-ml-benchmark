"""Phase 10a pilots, as pre-registered in validation.yaml: jet_network_pilot (LorentzNet and
the Deep Sets control) and particle_transformer_pilot (the upstream ParT).

Networks over the b1, b2 and c1 jets, fitted on the rows, splits and fit weights of the named
truth-tagged grid, and compared on validation with that grid's BDTs of the same seeds; ParT
also with the LorentzNet pilot's networks of the same seeds. The descriptive stage C reuses
the registered stage C design; its BDT value must reproduce the named stage C report. The
test partition stays sealed: no test row is read.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from hepml.adapters.configuration import default_study_directory, load_analysis
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
from hepml.application.binned_fit import asimov_significance, closest_cb_mass, log_bins, score_boundaries
from hepml.application.evaluation import evaluate_validation, evaluation_weights
from hepml.commands.binned_fit_study import IDENTITY, merged_templates, role_arrays
from hepml.domain.metrics import background_efficiency_at

# Each registered design: its networks and the one its primary decision is about.
DESIGNS = {"jet_network_pilot": dict(models={"lorentznet", "deep_sets"}, compare="lorentznet"),
           "particle_transformer_pilot": dict(models={"particle_transformer"}, compare="particle_transformer")}
ROC_SIGNAL_EFFICIENCIES = np.round(np.linspace(0.05, 0.95, 19), 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study")
    parser.add_argument("--grid-report", required=True, help="The registered truth-tagged grid report")
    parser.add_argument("--stage-c-report", required=True, help="Completed stage C of that grid")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", help="cuda or cpu; default: cuda when available")
    parser.add_argument("--design", choices=sorted(DESIGNS), default="jet_network_pilot")
    parser.add_argument("--lorentznet-report", help="Completed LorentzNet pilot (particle_transformer_pilot only)")
    args = parser.parse_args(argv)
    run_study(args.outdir, _run, args)


def validate_design(name, design, grid, grid_report):
    implementation = design.get("implementation")
    if implementation and importlib.metadata.version(implementation["package"]) != implementation["version"]:
        raise ValueError(f"The registered design needs {implementation['package']} {implementation['version']}")
    if (Path(grid_report).name != design["grid_report"] or grid.get("tagging") != "truth"
            or not grid.get("matching_acceptance") or not grid.get("flavour_overlap")
            or not set(design["masses"]) <= set(grid["settings"]["masses"])
            or not set(design["seeds"]) <= set(grid["settings"]["seeds"])
            or set(design["models"]) != DESIGNS[name]["models"]
            or design["primary"]["compare"] != DESIGNS[name]["compare"]
            or design["nodes"]["jets"] != ["b1", "b2", "c1"] or design["fit_weight"] != "pass_probability"
            or design["nodes"]["beams"] != (name == "jet_network_pilot")
            or grid["rerun"]["fit_weight"] != "pass_probability"
            or design["checkpoint"] != "best_fit_weighted_validation_auc"
            or design["primary"]["metric"] != "weighted_validation_auc"):
        raise ValueError("The network pilot needs the registered truth-tagged grid (acceptance and overlap removal) "
                         "and the registered design")


def stage_c_z(frame, score, weights, mass, settings, stage_c, edges):
    """Stage C's primary fit for any score: its categories x the closest-pairing c-b mass.
    Returns (Z, status); Z is NaN when a score cannot form the registered categories, for
    example when one heavy background event spans two boundaries."""
    background = frame.target.to_numpy() == 0
    roles = role_arrays(frame)
    x = closest_cb_mass(roles["b1"], roles["b2"], roles["c1"], mass)[0] / mass
    efficiencies = stage_c["category_background_efficiencies"]
    try:
        thresholds = score_boundaries(score[background], weights[background], efficiencies)
    except ValueError as error:
        return float("nan"), f"categories not formed: {error}"
    category = np.searchsorted(thresholds, score, side="right")
    s, b, v = [], [], []
    for k in range(len(efficiencies) + 1):
        for row in merged_templates(x, frame, weights, category == k, edges, settings["binning"]["min_background_neff"]):
            s.append(row["S"])
            b.append(row["B"])
            v.append(row["mc_variance"])
    return asimov_significance(s, b, normalization=settings["normalization_uncertainty"],
                               shape=settings["primary_shape_uncertainty"], mc_variance=v), "valid"


def paired_bootstrap(target, weights, scores, reference, *, replicates, seed):
    """AUC differences (model - reference) per seed for each replica of the validation events."""
    rng = np.random.default_rng(seed)
    differences = np.empty((replicates, len(scores)))
    for replica in range(replicates):
        resampled = weights * np.bincount(rng.integers(0, len(target), len(target)), minlength=len(target))
        for column, seed_key in enumerate(scores):
            differences[replica, column] = (roc_auc_score(target, scores[seed_key], sample_weight=resampled)
                                            - roc_auc_score(target, reference[seed_key], sample_weight=resampled))
    return differences


def _run(args):
    import torch

    from hepml.adapters.jet_networks import ROLE_COLUMNS, fit_network, jet_nodes, load_network, predict, save_network

    study = load_study(args.study or default_study_directory())
    cfg = load_analysis(study.directory / "analysis.yaml").evaluation
    validation = yaml.safe_load((study.directory / "validation.yaml").read_text())
    design, settings, stage_c = (validation[name] for name in (args.design, "binned_fit_study",
                                                                "binned_fit_stage_c"))
    models = sorted(design["models"])
    grid_report, stage_c_report = Path(args.grid_report).resolve(), Path(args.stage_c_report).resolve()
    grid = completed_report(grid_report, [9])
    validate_design(args.design, design, grid, grid_report)
    stage_c_summary = completed_report(stage_c_report, [9])
    if stage_c_summary.get("stage") != "binned_fit_C" or Path(stage_c_summary["grid_report"]).resolve() != grid_report:
        raise ValueError("--stage-c-report must be the completed stage C of this grid")
    lorentznet_runs, extra_inputs = {}, []
    if "secondary" in design:
        if not args.lorentznet_report:
            raise ValueError("The secondary comparison needs --lorentznet-report")
        reference_report = Path(args.lorentznet_report).resolve()
        reference = completed_report(reference_report, [10])
        frozen = yaml.safe_load((reference_report / "provenance/preregistration.json").read_text())
        if (reference.get("stage") != "jet_network_pilot" or reference_report.name != design["lorentznet_report"]
                or Path(reference["grid_report"]).resolve() != grid_report
                or not set(design["masses"]) <= set(reference["design"]["masses"])
                or not set(design["seeds"]) <= set(reference["design"]["seeds"])):
            raise ValueError("--lorentznet-report must be the registered LorentzNet pilot on this grid")
        fits = pd.read_csv(reference_report / "tables/fits.csv")
        reference_runs = Path(frozen["arguments"]["runs_dir"]).resolve()
        lorentznet_runs = {(r.mass, r.seed): reference_runs / r.run_id for r in fits[fits.model == "lorentznet"].itertuples()}
        extra_inputs = [reference_report / "provenance" / name for name in ("summary.json", "status.json")]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    masses, seeds, couplings = design["masses"], design["seeds"], grid["settings"]["couplings"]
    directories = {float(k): v for k, v in grid["settings"]["coupling_directories"].items()}
    training_coupling = float(grid["settings"]["training_coupling"])
    bench = TruthTagBenchmark(grid["dataset"])
    bdt_runs = {(r["mass"], r["seed"]): Path(r["model_directory"]) for r in grid["runs"]}
    out, runs = Path(args.outdir).resolve(), Path(args.runs_dir).resolve()
    if not args.run_prefix or Path(args.run_prefix).name != args.run_prefix:
        raise ValueError("run-prefix must be a directory name")
    run_ids = {(m, kind, s): f"{args.run_prefix}-m{m}-{kind}-seed{s}" for m in masses for kind in models for s in seeds}
    refuse_existing(out, *(runs / run_id for run_id in run_ids.values()))

    splits = ("train", "val")
    paths = [study.directory / "validation.yaml", study.directory / "analysis.yaml",
             *(report / "provenance" / name for report in (grid_report, stage_c_report)
               for name in ("summary.json", "status.json")),
             grid_report / "tables/grid.csv", grid_report / "tables/auc.csv", stage_c_report / "tables/comparison.csv",
             *(bench.background_path(split) for split in splits), *extra_inputs]
    for mass in masses:
        for rho in couplings:
            paths += [bench.meta_path(directories[rho], mass),
                      *(bench.signal_path(directories[rho], mass, split) for split in splits),
                      *(bdt_runs[mass, seed] / "points" / directories[rho] / "preds_val.parquet" for seed in seeds),
                      *(lorentznet_runs[mass, seed] / f"preds_val_{directories[rho]}.parquet"
                        for seed in seeds if (mass, seed) in lorentznet_runs)]
    inputs = hash_files(sorted(set(paths)))
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    open_report(out, dict(study=str(study.directory), arguments=vars(args), design=design, stage_c=stage_c,
                          binned_fit=settings, evaluation_config=asdict(cfg), grid_report=str(grid_report),
                          stage_c_report=str(stage_c_report), torch=torch.__version__, device=device,
                          device_name=device_name, test_evaluated=False), inputs, preregister=True)
    checks = [dict(check="grid and stage C reports complete and unchanged; registered design", passed=True)]
    nodes = dict(unit_gev=design["nodes"]["momentum_unit_gev"], beams=design["nodes"]["beams"])
    columns = [*IDENTITY, "fit_weight", *ROLE_COLUMNS, "mcb1", "mcb2"]
    edges = log_bins(settings["binning"]["x_min"], settings["binning"]["x_max"], settings["binning"]["log_step"])

    def arrays(frame):
        return (*jet_nodes(frame, **nodes), frame.target.to_numpy(), frame.fit_weight.to_numpy())

    # 1. Fits at the grid's training coupling, on the rows the BDTs were fitted on.
    fit_rows, curve_rows, artifacts = [], [], []
    for mass in masses:
        train, val = (bench.split(directories[training_coupling], mass, split, columns) for split in splits)
        for seed in seeds:
            bdt = pd.read_parquet(bdt_runs[mass, seed] / "points" / directories[training_coupling] / "preds_val.parquet",
                                  columns=["event_id", "sample_key"])
            same = len(bdt) == len(val) and set(zip(bdt.event_id, bdt.sample_key)) == set(zip(val.event_id, val.sample_key))
            checks.append(dict(check=f"m{mass} seed {seed}: the BDT was scored on exactly these validation rows",
                               passed=bool(same)))
        train_arrays, val_arrays = arrays(train), arrays(val)
        for kind in models:
            for seed in seeds:
                run_id = run_ids[mass, kind, seed]
                model, record = fit_network(kind, design["models"][kind], design["training"], train_arrays, val_arrays,
                                            seed=seed, device=device)
                save_network(runs / run_id, model, record)
                for rho in couplings:
                    frame = bench.split(directories[rho], mass, "val", columns)
                    score = predict(model, *jet_nodes(frame, **nodes), device=device)
                    frame[["event_id", "sample_key"]].assign(score=score).to_parquet(
                        runs / run_id / f"preds_val_{directories[rho]}.parquet", index=False)
                reloaded, _ = load_network(runs / run_id, device=device)
                saved = pd.read_parquet(runs / run_id / f"preds_val_{directories[training_coupling]}.parquet").score
                reproduced = predict(reloaded, val_arrays[0], val_arrays[1], device=device)
                checks.append(dict(check=f"{run_id}: the reloaded network reproduces its saved validation scores",
                                   passed=bool(np.allclose(reproduced, saved, rtol=1e-6, atol=1e-7))))
                fit_rows.append(dict(mass=mass, model=kind, seed=seed, run_id=run_id,
                                     **{k: v for k, v in record.items() if k not in ("history", "config", "training", "seed", "kind")}))
                curve_rows += [dict(mass=mass, model=kind, seed=seed, **row) for row in record["history"]]
                artifacts += [p for p in (runs / run_id).rglob("*") if p.is_file()]
                print(f"{run_id}: best epoch {record['best_epoch']} of {record['epochs_run']}, "
                      f"{record['seconds']:.0f} s", flush=True)

    # 2. Validation metrics at every coupling, for the networks and the grid's BDTs alike.
    grid_table, auc_table = pd.read_csv(grid_report / "tables/grid.csv"), pd.read_csv(grid_report / "tables/auc.csv")
    stage_c_table = pd.read_csv(stage_c_report / "tables/comparison.csv")
    metric_rows, roc_rows, primary_rows, bootstrap_rows = [], [], [], []
    closure = dict(auc=[], z=[], stage_c=[])
    for mass in masses:
        for rho in couplings:
            frame = bench.split(directories[rho], mass, "val", columns)
            full, stored_lumi = bench.totals(directories[rho], mass), float(bench.meta(directories[rho], mass)["lumi"])
            weights, _ = evaluation_weights(full, frame, stored_lumi=stored_lumi, target_lumi=cfg.lumi_pb_inv,
                                            signal_k_factor=cfg.signal_k_factor,
                                            background_k_factor=cfg.background_k_factor)
            target, key = frame.target.to_numpy(), frame[["event_id", "sample_key"]]
            scores = {}
            for seed in seeds:
                bdt = pd.read_parquet(bdt_runs[mass, seed] / "points" / directories[rho] / "preds_val.parquet",
                                      columns=["event_id", "sample_key", "bdt_score"])
                scores["bdt", seed] = key.merge(bdt, on=["event_id", "sample_key"], validate="one_to_one").bdt_score.to_numpy()
                saved = {kind: runs / run_ids[mass, kind, seed] for kind in models}
                if (mass, seed) in lorentznet_runs:
                    saved["lorentznet"] = lorentznet_runs[mass, seed]
                for kind, directory in saved.items():
                    net = pd.read_parquet(directory / f"preds_val_{directories[rho]}.parquet")
                    scores[kind, seed] = key.merge(net, on=["event_id", "sample_key"], validate="one_to_one").score.to_numpy()
            for (model, seed), score in scores.items():
                auc = float(roc_auc_score(target, score, sample_weight=weights))
                point = evaluate_validation(full, frame.assign(bdt_score=score), cfg, stored_lumi=stored_lumi)[0]
                operating = point["operating_points"][cfg.primary_objective]
                z_c, stage_c_status = stage_c_z(frame, score, weights, mass, settings, stage_c, edges)
                row = dict(mass=mass, rho_tc=rho, model=model, seed=seed, auc_weighted=auc,
                           auc_unweighted=float(roc_auc_score(target, score)), status=operating["status"],
                           **{cfg.primary_objective: operating["metrics"].get(cfg.primary_objective, np.nan),
                              "background_neff_at_optimum": operating["metrics"].get("background_neff", np.nan)},
                           Z_stage_c=z_c, stage_c_status=stage_c_status)
                for efficiency in design["descriptive_signal_efficiencies"]:
                    eff, neff = background_efficiency_at(score, target, weights, efficiency)
                    row[f"background_efficiency_at_{efficiency:g}"], row[f"background_neff_at_{efficiency:g}"] = eff, neff
                metric_rows.append(row)
                if model == "bdt":
                    reference = auc_table[(auc_table.mass == mass) & np.isclose(auc_table.rho_tc, rho)
                                          & (auc_table.seed == seed)].auc_weighted.iloc[0]
                    closure["auc"].append(abs(auc - reference))
                    grid_row = grid_table[(grid_table.method == "bdt") & (grid_table.mass == mass)
                                          & np.isclose(grid_table.rho_tc, rho) & (grid_table.seed == seed)
                                          & (grid_table.objective == cfg.primary_objective)].iloc[0]
                    closure["z"].append(abs(row[cfg.primary_objective] / grid_row[cfg.primary_objective] - 1))
                    if seed == grid["primary_seed"]:
                        saved = stage_c_table[(stage_c_table.mass == mass) & np.isclose(stage_c_table.rho_tc, rho)]
                        closure["stage_c"].append(abs(z_c / saved.Z_binned.iloc[0] - 1))
                if np.isclose(rho, training_coupling):
                    for efficiency in ROC_SIGNAL_EFFICIENCIES:
                        eff, neff = background_efficiency_at(score, target, weights, efficiency)
                        roc_rows.append(dict(mass=mass, model=model, seed=seed, signal_efficiency=efficiency,
                                             background_efficiency=eff, background_neff=neff))
            if not np.isclose(rho, training_coupling):
                continue
            # 3. The primary comparison, paired with the BDT of the same seed on the same resamples.
            primary = design["primary"]
            comparisons = [(kind, "bdt") for kind in models]
            if "secondary" in design:
                comparisons.append((primary["compare"], design["secondary"]["against"]))
            for kind, against in comparisons:
                model_scores = {seed: scores[kind, seed] for seed in seeds}
                reference = {seed: scores[against, seed] for seed in seeds}
                # The same seed gives the same resamples for every comparison at this mass.
                replicas = paired_bootstrap(target, weights, model_scores, reference,
                                            replicates=primary["bootstrap_replicates"],
                                            seed=[primary["bootstrap_seed"], mass])
                nominal = np.array([roc_auc_score(target, model_scores[s], sample_weight=weights)
                                    - roc_auc_score(target, reference[s], sample_weight=weights) for s in seeds])
                tail = (1 - primary["interval"]) / 2
                low, high = np.quantile(replicas.mean(axis=1), [tail, 1 - tail])
                primary_rows.append(dict(mass=mass, model=kind, against=against,
                                         registered=kind == primary["compare"] and against == "bdt",
                                         **{f"delta_auc_seed{s}": d for s, d in zip(seeds, nominal)},
                                         delta_auc_mean=float(nominal.mean()), interval_low=float(low),
                                         interval_high=float(high), improvement=bool(nominal.mean() > 0 and low > 0
                                                                                     and (nominal > 0).all())))
                bootstrap_rows += [dict(mass=mass, model=kind, against=against, replica=r, delta_auc_mean=float(d))
                                   for r, d in enumerate(replicas.mean(axis=1))]
        print(f"m{mass}: metrics and paired bootstrap done", flush=True)

    checks += [
        dict(check=f"BDT weighted AUCs reproduce the grid's (max difference {max(closure['auc']):.1e})",
             passed=bool(max(closure["auc"]) < 1e-9)),
        dict(check=f"BDT {cfg.primary_objective} at the validation optimum reproduces the grid's "
                   f"(max relative {max(closure['z']):.1e})", passed=bool(max(closure["z"]) < 1e-9)),
        dict(check=f"stage C on the primary-seed BDT reproduces the stage C report (max relative "
                   f"{max(closure['stage_c']):.1e})", passed=bool(max(closure["stage_c"]) < 1e-9)),
        dict(check="no test rows read and no test scores written",
             passed=not any("test" in Path(p).name for p in inputs)
             and not any("test" in p.name for p in artifacts)),
    ]
    tables = dict(checks=pd.DataFrame(checks), fits=pd.DataFrame(fit_rows), learning_curves=pd.DataFrame(curve_rows),
                  metrics=pd.DataFrame(metric_rows), roc=pd.DataFrame(roc_rows), primary=pd.DataFrame(primary_rows),
                  bootstrap=pd.DataFrame(bootstrap_rows))
    write_tables(out, tables)
    if not tables["checks"].passed.all():
        raise ValueError("Network pilot contract checks failed")
    primary = tables["primary"]
    registered = primary[primary.registered]
    secondary = primary[(primary.model == design["primary"]["compare"]) & (primary.against != "bdt")]
    summary = dict(schema_version=2, phases=[10], stage=args.design, design=design,
                   evaluation_config=asdict(cfg), grid_report=str(grid_report), stage_c_report=str(stage_c_report),
                   training_coupling=training_coupling, couplings=couplings, device=device, device_name=device_name,
                   torch=torch.__version__, improvement={f"m{r.mass}": bool(r.improvement) for r in registered.itertuples()},
                   secondary={f"m{r.mass}": bool(r.improvement) for r in secondary.itertuples()},
                   lorentznet_report=args.lorentznet_report and str(Path(args.lorentznet_report).resolve()),
                   test_evaluated=False, status="checkpoint_ready",
                   note=f"Jet-level {design['primary']['compare']} pilot against the grid's BDTs on the same rows and "
                        "weights; validation only.")
    write_json(out / "provenance/summary.json", summary)
    from hepml.adapters.network_report import save_network_figures

    save_network_figures(out)
    finish_report(out, "06_jet_networks.ipynb", inputs, sorted(set(artifacts)))
    for row in pd.concat([registered, secondary]).itertuples():
        print(f"m{row.mass}: {row.model} - {row.against} weighted AUC {row.delta_auc_mean:+.4f} "
              f"[{row.interval_low:+.4f}, {row.interval_high:+.4f}]; improvement: {row.improvement}", flush=True)
