"""Phase 9 follow-up: sensitivity of a binned c-b mass shape fit on the grid points.

Same validation events, weights and preselection as the Phase 9 grid. The c-b mass
pairs the c-tagged jet with the b-tagged jet whose c-b mass is closest to the tested
mass. Stage A fits it without a BDT cut, with jets optionally calibrated by a
parton/reco pT response derived on training-split signal. Stage B fits it in
categories of the grid's primary-seed BDT score, whose boundaries background alone
sets. The test partition is never read.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from hepml_compact.contracts import part_glob_pattern

from hepml.adapters.binned_fit_report import save_binned_figures
from hepml.adapters.configuration import default_study_directory, load_analysis
from hepml.adapters.dataset_files import discover_samples
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
)
from hepml.adapters.truth_tag_datasets import TruthTagBenchmark
from hepml.application.binned_fit import (
    asimov_significance,
    boundaries_follow_rule,
    closest_cb_mass,
    log_bins,
    match_to_partons,
    merge_bins,
    pair_mass,
    response_curve,
    response_scale,
    score_boundaries,
)
from hepml.application.evaluation import evaluation_weights
from hepml.domain.artifacts import BenchmarkFiles

IDENTITY = ["event_id", "sample_key", "sample", "target", "sample_weight"]
ROLES = {"b1": ("ptb1", "etab1", "phib1", "massb1"), "b2": ("ptb2", "etab2", "phib2", "massb2"),
         "c1": ("ptc1", "etac1", "phic1", "massc1")}
TRUTH = ["truth_partons_pid", "truth_partons_status", "truth_partons_eta", "truth_partons_phi", "truth_partons_pt"]
STAGE = "binned_fit_A"
STAGE_B = "binned_fit_B"
STAGE_C = "binned_fit_C"
CLOSURE = "one_bin_above_the_grid_threshold_reproduces_its_Z_syst_5pct"
# Stage B's fits: its primary, then the descriptive single-category and score-only fits.
FITS = ("categories_x_mass", "single_category", "categories_only")
# Largest allowed difference between a designed and an achieved background fraction above a boundary.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study")
    parser.add_argument("--grid-report", required=True, help="Completed Phase 9 grid report")
    parser.add_argument("--stage", choices=["A", "B", "C"], default="A",
                        help="A: c-b mass without a BDT cut; B: BDT-score categories x c-b mass; "
                             "C: finer categories on a truth-tagged grid")
    parser.add_argument("--stage-b-report", help="Stage C: the completed stage-B report of the same grid")
    parser.add_argument("--compact-root", help="Compact exports, for stage A's jet calibration (parton truth)")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    run_study(args.outdir, _run if args.stage == "A" else _run_stage_b, args)


def valid_efficiencies(efficiencies):
    return (bool(efficiencies) and all(0 < e < 1 for e in efficiencies)
            and all(a > b for a, b in zip(efficiencies, efficiencies[1:])))


def validate_settings(settings):
    binning = settings["binning"]
    shapes = settings["shape_uncertainties"]
    if (settings["stage"] != "A" or settings["observable"] != "closest_cb_mass"
            or settings["jets"] not in {"calibrated", "reconstructed"}
            or settings["primary_shape_uncertainty"] not in shapes or any(s < 0 for s in shapes)
            or settings["normalization_uncertainty"] < 0 or binning["min_background_neff"] <= 0
            or settings["mc_statistics"] != "background_sum_w2_per_bin" or settings["closure_tolerance"] <= 0):
        raise ValueError("Unsupported binned-fit design")
    log_bins(binning["x_min"], binning["x_max"], binning["log_step"])
    if settings["jets"] == "calibrated":
        calibration = settings["calibration"]
        if (calibration["statistic"] != "median_parton_over_reco_pt"
                or calibration["interpolation"] != "linear_in_log_pt_between_bin_medians"
                or sorted(calibration["pt_edges"]) != calibration["pt_edges"] or calibration["match_delta_r"] <= 0):
            raise ValueError("Unsupported calibration design")


def validate_stage_b(stage_b, settings, grid, cfg):
    efficiencies = stage_b["category_background_efficiencies"]
    if (stage_b["stage"] != "B" or stage_b["observable"] != settings["observable"] or stage_b["jets"] != "reconstructed"
            or stage_b["templates"] != "validation" or stage_b["model_seed"] != grid["primary_seed"]
            or set(stage_b["descriptive"]) != {"infinite_mc", *FITS[1:]}
            or stage_b["closure"] != CLOSURE
            or cfg.primary_objective != "Z_syst_5pct" or settings["normalization_uncertainty"] != 0.05
            or not valid_efficiencies(efficiencies)):
        raise ValueError("Unsupported stage-B design")


def validate_stage_c(stage_c, settings, grid, cfg):
    sets = [stage_c["category_background_efficiencies"], *stage_c["descriptive_category_sets"].values()]
    if (stage_c["stage"] != "C" or stage_c["grid_tagging"] != "truth" or grid.get("tagging") != "truth"
            or stage_c["observable"] != settings["observable"] or stage_c["jets"] != "reconstructed"
            or stage_c["templates"] != "validation" or stage_c["model_seed"] != grid["primary_seed"]
            or set(stage_c["descriptive"]) != {"infinite_mc", *FITS[1:]}
            or stage_c["comparison"] != "single_bin_bdt_with_mc_statistics"
            or stage_c["closure"] != [CLOSURE, "stage_b_categories_reproduce_stage_b"]
            or "stage_b" not in stage_c["descriptive_category_sets"]
            or set(stage_c["descriptive_category_sets"]) & {"primary", *FITS}
            or cfg.primary_objective != "Z_syst_5pct" or settings["normalization_uncertainty"] != 0.05
            or not all(valid_efficiencies(e) for e in sets)):
        raise ValueError("Unsupported stage-C design, or a grid that is not truth-tagged")


def calibration_jets(paths, event_ids, max_delta_r):
    """Reco and matched parton pT of the tagged jets: b-tagged roles to b partons, c-tagged to c partons."""
    rows = []
    for path in paths:
        names = pq.read_schema(path).names
        if not set(TRUTH) <= set(names):
            raise ValueError(f"{path}: no parton truth; calibration needs a 0.2.0 or newer export")
        table = pq.read_table(path, columns=["event_id", "jets_pt", "jets_eta", "jets_phi", "role_b1", "role_b2",
                                              "role_c1", *TRUTH]).to_pydict()
        for i, event_id in enumerate(table["event_id"]):
            if event_id not in event_ids:
                continue
            partons = [np.asarray(table[name][i]) for name in TRUTH]
            pid, status, eta, phi, pt = partons
            for role, kind, flavour in (("role_b1", "b", 5), ("role_b2", "b", 5), ("role_c1", "c", 4)):
                jet = table[role][i]
                jet_pt, jet_eta, jet_phi = table["jets_pt"][i][jet], table["jets_eta"][i][jet], table["jets_phi"][i][jet]
                match = match_to_partons(jet_eta, jet_phi, pid, status, eta, phi, flavour=flavour,
                                         max_delta_r=max_delta_r)
                rows.append(dict(kind=kind, reco_pt=jet_pt, parton_pt=pt[match] if match >= 0 else np.nan))
    return pd.DataFrame(rows)


def role_arrays(frame):
    return {role: tuple(frame[column].to_numpy(dtype=float) for column in columns) for role, columns in ROLES.items()}


def merged_templates(x, frame, weights, mask, edges, min_neff):
    """Merged bins of x for the events in mask: S, B, MC variance, N_eff and B per background process."""
    target, samples = frame.target.to_numpy(), frame["sample"].to_numpy()
    signal, background = mask & (target == 1), mask & (target == 0)
    s_fine = _histogram(x, signal, weights, edges)
    b_fine = _histogram(x, background, weights, edges)
    b2_fine = _histogram(x, background, weights**2, edges)
    groups = merge_bins(b_fine, b2_fine, min_neff)
    processes = {sample: _histogram(x, background & (samples == sample), weights, edges)
                 for sample in sorted(frame.loc[target == 0, "sample"].unique())}
    rows = []
    for index, group in enumerate(groups):
        s, b, mc = s_fine[group].sum(), b_fine[group].sum(), b2_fine[group].sum()
        rows.append(dict(bin=index, x_low=edges[group[0]], x_high=edges[group[-1] + 1], S=s, B=b, mc_variance=mc,
                         background_neff=b**2 / mc, S_over_B=s / b,
                         **{f"B_{sample}": fine_values[group].sum() for sample, fine_values in processes.items()}))
    return rows


class DirectPoints:
    """Per-point prepared benchmarks of the direct-tag grid."""

    def __init__(self, grid, directories):
        self.root, self.directories = Path(grid["dataset"]), directories

    def _files(self, rho):
        return BenchmarkFiles(self.root / self.directories[rho])

    def full(self, mass, rho):
        return pd.read_parquet(self._files(rho).dataset(mass), columns=IDENTITY)

    def split(self, mass, rho, name, columns):
        return pd.read_parquet(self._files(rho).split(name, mass), columns=columns)

    def stored_lumi(self, mass, rho):
        return _stored_lumi(self._files(rho), mass)

    def paths(self, mass, rho, splits):
        files = self._files(rho)
        return [files.dataset(mass), files.meta(mass), *(files.split(name, mass) for name in splits)]


class TruthTagPoints:
    """Per-point truth-tagged rows: the point's signal and the shared background."""

    def __init__(self, grid, directories):
        self.bench, self.directories = TruthTagBenchmark(grid["dataset"]), directories

    def full(self, mass, rho):
        return self.bench.totals(self.directories[rho], mass)

    def split(self, mass, rho, name, columns):
        return self.bench.split(self.directories[rho], mass, name, columns)

    def stored_lumi(self, mass, rho):
        return float(self.bench.meta(self.directories[rho], mass)["lumi"])

    def paths(self, mass, rho, splits):
        return [self.bench.meta_path(self.directories[rho], mass),
                *(self.bench.signal_path(self.directories[rho], mass, name) for name in splits),
                *(self.bench.background_path(name) for name in splits)]


def point_data(grid, directories):
    return (TruthTagPoints if grid.get("tagging") == "truth" else DirectPoints)(grid, directories)


def truth_tag_settings(settings, registered, grid):
    """A truth-tagged grid uses the registered rerun's jets (reconstructed: no parton truth is exported)."""
    if grid.get("tagging") != "truth":
        return settings
    settings = {**settings, "jets": registered["truth_tag_rerun"]["binned_fit"]["jets"]}
    validate_settings(settings)
    return settings


def _load_design(args):
    study = load_study(args.study or default_study_directory())
    cfg = load_analysis(study.directory / "analysis.yaml").evaluation
    registered = yaml.safe_load((study.directory / "validation.yaml").read_text())
    validate_settings(registered["binned_fit_study"])
    grid_report = Path(args.grid_report).resolve()
    grid = completed_report(grid_report, [9])
    if grid.get("stage") is not None or "frozen_threshold_directories" not in grid:
        raise ValueError("Require the completed Phase 9 grid report itself")
    if comparable_config(grid["evaluation_config"]) != comparable_config(asdict(cfg)):
        raise ValueError("Evaluation settings (luminosity, k-factors) differ from the Phase 9 grid")
    return study, cfg, registered, grid_report, grid


def _grid_references(grid_table, grid, cfg, mass, rho):
    """The grid's primary-objective rows at one point: primary-seed BDT and cut reference."""
    reference = {}
    for method in ("bdt", "cuts"):
        reference[method] = grid_table[(grid_table.method == method) & (grid_table.mass == mass)
                                       & np.isclose(grid_table.rho_tc, rho)
                                       & (grid_table.objective == cfg.primary_objective)
                                       & ((grid_table.seed == grid["primary_seed"]) if method == "bdt" else True)
                                       ].iloc[0]
    return reference


def _run(args):
    study, cfg, registered, grid_report, grid = _load_design(args)
    settings = truth_tag_settings(registered["binned_fit_study"], registered, grid)
    if settings["jets"] == "calibrated" and not args.compact_root:
        raise ValueError("The stage-A jet calibration needs --compact-root")
    grid_settings = grid["settings"]
    masses, couplings = grid_settings["masses"], grid_settings["couplings"]
    directories = {float(key): value for key, value in grid_settings["coupling_directories"].items()}
    points, registry_path = point_data(grid, directories), Path(grid["registry"]) / "assignments.parquet"
    out = Path(args.outdir).resolve()
    refuse_existing(out)
    binning, norm = settings["binning"], settings["normalization_uncertainty"]
    shapes = [0.0, *settings["shape_uncertainties"]]
    edges = log_bins(binning["x_min"], binning["x_max"], binning["log_step"])
    jet_variants = ["calibrated", "reconstructed"] if settings["jets"] == "calibrated" else ["reconstructed"]

    paths = [grid_report / "provenance/summary.json", grid_report / "provenance/status.json",
             grid_report / "tables/grid.csv", registry_path]
    for mass in masses:
        for rho in couplings:
            paths += points.paths(mass, rho, ("train", "val"))
    calibration_paths = []
    if settings["jets"] == "calibrated":
        coupling = settings["calibration"]["coupling"]
        compact = Path(args.compact_root).resolve() / directories[coupling]
        signals, _ = discover_samples(compact)
        for mass in masses:
            meta = signals.get(str(mass))
            if meta is None or not np.isclose(float(meta["rho_tc"]), coupling):
                raise ValueError(f"No compact signal for m{mass}, rho_tc={coupling}")
            calibration_paths += sorted(compact.glob(part_glob_pattern(meta["_stem"])))
            paths.append(compact / meta["export_manifest"])
    inputs = hash_files(sorted(set(paths + calibration_paths)))
    preregistration = dict(study=str(study.directory), arguments=vars(args), settings=settings,
                           evaluation_config=asdict(cfg), grid_report=str(grid_report), test_evaluated=False)
    open_report(out, preregistration, inputs, preregister=True)
    checks = [dict(check="phase9_grid_report_complete_and_unchanged", passed=True),
              dict(check="evaluation_config_and_k_factors_match_the_grid", passed=True)]

    # 1. Jet calibration from training-split signal at the calibration coupling.
    curves, calibration_rows = {}, []
    if settings["jets"] == "calibrated":
        registry = pd.read_parquet(registry_path, columns=["event_id", "split"])
        train_ids = set(registry.loc[registry.split == "train", "event_id"])
        jets = calibration_jets(calibration_paths, train_ids, settings["calibration"]["match_delta_r"])
        pt_edges = settings["calibration"]["pt_edges"]
        for kind in ("b", "c"):
            chosen = jets[jets.kind == kind]
            nodes, values, counts = response_curve(chosen.reco_pt, chosen.parton_pt, pt_edges)
            curves[kind] = (nodes, values)
            upper = [*pt_edges[1:], np.inf]
            calibration_rows += [dict(kind=kind, pt_low=lo, pt_high=hi, node_pt=n, median_parton_over_reco=v,
                                      matched_jets=c)
                                 for lo, hi, n, v, c in zip(pt_edges, upper, nodes, values, counts)]
            match_rate = chosen.parton_pt.notna().mean()
            checks.append(dict(check=f"calibration: {kind}-tagged jets matched to {kind} partons "
                                     f"({match_rate:.0%} of {len(chosen)})", passed=bool(match_rate > 0.5)))
        print(f"Calibration derived from {len(jets)} training-split tagged jets", flush=True)

    # 2. Templates, merged bins and significances at every point.
    grid_table = pd.read_csv(grid_report / "tables/grid.csv")
    bins_rows, z_rows, peak_rows, comparison_rows = [], [], [], []
    closure = []
    templates = {"validation": ("val",), "train+validation": ("train", "val")}
    for mass in masses:
        for rho in couplings:
            full = points.full(mass, rho)
            columns = [*IDENTITY, *(c for cs in ROLES.values() for c in cs), "mcb1", "mcb2"]
            splits = {name: points.split(mass, rho, name, columns) for name in ("train", "val")}
            val_roles = role_arrays(splits["val"])
            closure.append(max(mass_residual(val_roles["b1"], val_roles["c1"], splits["val"].mcb1),
                               mass_residual(val_roles["b2"], val_roles["c1"], splits["val"].mcb2)))
            point_z = {}
            for template, names in templates.items():
                frame = pd.concat([splits[n] for n in names], ignore_index=True)
                weights, _ = evaluation_weights(full, frame, stored_lumi=points.stored_lumi(mass, rho),
                                                target_lumi=cfg.lumi_pb_inv, signal_k_factor=cfg.signal_k_factor,
                                                background_k_factor=cfg.background_k_factor)
                roles = role_arrays(frame)
                signal, background = frame.target.to_numpy() == 1, frame.target.to_numpy() == 0
                for jets_label in jet_variants:
                    scales = {role: (response_scale(roles[role][0], *curves["c" if role == "c1" else "b"])
                                     if jets_label == "calibrated" else 1.0) for role in ROLES}
                    values, _ = closest_cb_mass(roles["b1"], roles["b2"], roles["c1"], mass, scale_b1=scales["b1"],
                                                scale_b2=scales["b2"], scale_c1=scales["c1"])
                    x = values / mass
                    merged = merged_templates(x, frame, weights, np.ones(len(frame), dtype=bool), edges,
                                              binning["min_background_neff"])
                    bins_rows += [dict(mass=mass, rho_tc=rho, jets=jets_label, templates=template, **row)
                                  for row in merged]
                    s_bins, b_bins, mc_bins = (np.array([row[key] for row in merged]) for key in ("S", "B", "mc_variance"))
                    for shape in shapes:
                        for with_mc in (True, False):
                            z = asimov_significance(s_bins, b_bins, normalization=norm, shape=shape,
                                                    mc_variance=mc_bins if with_mc else None)
                            z_rows.append(dict(mass=mass, rho_tc=rho, jets=jets_label, templates=template,
                                               shape_uncertainty=shape, mc_statistics=with_mc, bins=len(merged), Z=z))
                            point_z[jets_label, template, shape, with_mc] = z
                    if template == "validation":
                        q16, q50, q84 = np.percentile(x[signal], [16, 50, 84])
                        peak_rows.append(dict(mass=mass, rho_tc=rho, jets=jets_label, peak_over_mass=q50,
                                              halfwidth_over_mass=(q84 - q16) / 2,
                                              signal_within_10pct=float(np.mean(np.abs(x[signal] - 1) <= 0.10))))
                        if jets_label == jet_variants[0]:
                            total_s, total_b = weights[signal].sum(), weights[background].sum()
                            no_selection = asimov_significance([total_s], [total_b], normalization=norm)
            primary = point_z[jet_variants[0], "validation", settings["primary_shape_uncertainty"], True]
            reference = _grid_references(grid_table, grid, cfg, mass, rho)
            comparison_rows.append(dict(
                mass=mass, rho_tc=rho, Z_binned=primary, Z_grid_bdt=reference["bdt"][cfg.primary_objective],
                Z_grid_cuts=reference["cuts"][cfg.primary_objective], Z_no_selection=no_selection,
                binned_over_bdt=primary / reference["bdt"][cfg.primary_objective],
                binned_over_cuts=primary / reference["cuts"][cfg.primary_objective],
                infinite_mc_over_primary=point_z[jet_variants[0], "validation", settings["primary_shape_uncertainty"],
                                                 False] / primary,
                train_plus_validation_over_primary=point_z[jet_variants[0], "train+validation",
                                                           settings["primary_shape_uncertainty"], True] / primary))
            # Closure: one bin with the grid's own operating point reproduces its Cowan Eq. 20 value.
            bdt = reference["bdt"]
            one_bin = asimov_significance([bdt.S], [bdt.B], normalization=norm)
            checks.append(dict(check=f"m{mass} rho{rho:g}: one bin reproduces the grid's {cfg.primary_objective}",
                               passed=bool(abs(one_bin / bdt[cfg.primary_objective] - 1) <= settings["closure_tolerance"])))
        print(f"m{mass}: {len(couplings)} couplings evaluated", flush=True)
    checks.append(dict(check=f"recomputed c-b masses reproduce stored mcb1/mcb2 (max |dm^2|/(E1+E2)^2 "
                             f"{max(closure):.1e})", passed=bool(max(closure) < 1e-5)))
    checks.append(dict(check="no_test_events_read", passed=not reads_test(inputs)))
    calibration_columns = ["kind", "pt_low", "pt_high", "node_pt", "median_parton_over_reco", "matched_jets"]
    tables = dict(checks=pd.DataFrame(checks), calibration=pd.DataFrame(calibration_rows, columns=calibration_columns),
                  signal_peak=pd.DataFrame(peak_rows), bins=pd.DataFrame(bins_rows),
                  significance=pd.DataFrame(z_rows), comparison=pd.DataFrame(comparison_rows))
    write_tables(out, tables)
    if not tables["checks"].passed.all():
        failed = tables["checks"].loc[~tables["checks"].passed, "check"]
        raise ValueError("Binned-fit contract checks failed: " + "; ".join(failed))
    comparison = tables["comparison"]
    summary = dict(schema_version=2, phases=[9], stage=STAGE, tagging=grid.get("tagging", "direct"),
                   matching_acceptance=grid.get("matching_acceptance"),
                   flavour_overlap=grid.get("flavour_overlap"), settings=settings, grid_settings=grid_settings,
                   evaluation_config=asdict(cfg), grid_report=str(grid_report), primary_seed=grid["primary_seed"],
                   primary=dict(jets=jet_variants[0], templates="validation",
                                shape_uncertainty=settings["primary_shape_uncertainty"], mc_statistics=True),
                   median_binned_over_bdt=float(comparison.binned_over_bdt.median()),
                   median_infinite_mc_over_primary=float(comparison.infinite_mc_over_primary.median()),
                   test_evaluated=False, status="checkpoint_ready",
                   note="Stage A: c-b mass shape fit on validation, without a BDT cut. Stage B (BDT categories "
                        "x c-b mass) and the background-MC decision follow this checkpoint.")
    write_json(out / "provenance/summary.json", summary)
    save_binned_figures(out)
    finish_report(out, "04_physics_results.ipynb", inputs)
    print(f"Binned fit: median Z_binned / Z_grid_bdt = {summary['median_binned_over_bdt']:.2f}; "
          f"median infinite-MC gain = {summary['median_infinite_mc_over_primary']:.2f}", flush=True)


def _run_stage_b(args):
    study, cfg, registered, grid_report, grid = _load_design(args)
    settings = registered["binned_fit_study"]
    stage_c = args.stage == "C"
    extra_inputs, stage_b_z = [], None
    if stage_c:
        stage_b = registered["binned_fit_stage_c"]
        validate_stage_c(stage_b, settings, grid, cfg)
        if not args.stage_b_report:
            raise ValueError("Stage C needs --stage-b-report")
        stage_b_report = Path(args.stage_b_report).resolve()
        previous = completed_report(stage_b_report, [9])
        if (previous.get("stage") != STAGE_B or Path(previous["grid_report"]).resolve() != grid_report
                or previous["stage_b"]["category_background_efficiencies"]
                != stage_b["descriptive_category_sets"]["stage_b"]):
            raise ValueError("--stage-b-report must be the completed stage B of this grid with stage B's categories")
        stage_b_z = pd.read_csv(stage_b_report / "tables/comparison.csv").set_index(["mass", "rho_tc"]).Z_binned
        extra_inputs = [stage_b_report / "provenance/summary.json", stage_b_report / "provenance/status.json",
                        stage_b_report / "tables/comparison.csv"]
        descriptive_sets = dict(stage_b["descriptive_category_sets"])
    else:
        stage_b = registered["binned_fit_stage_b"]
        validate_stage_b(stage_b, settings, grid, cfg)
        descriptive_sets = {}
    grid_settings = grid["settings"]
    masses, couplings = grid_settings["masses"], grid_settings["couplings"]
    directories = {float(key): value for key, value in grid_settings["coupling_directories"].items()}
    points = point_data(grid, directories)
    point_roots = {mass: Path(grid["frozen_threshold_directories"][f"m{mass}"]) for mass in masses}
    out = Path(args.outdir).resolve()
    refuse_existing(out)
    binning, norm, objective = settings["binning"], settings["normalization_uncertainty"], cfg.primary_objective
    shapes, primary_shape = [0.0, *settings["shape_uncertainties"]], settings["primary_shape_uncertainty"]
    edges = log_bins(binning["x_min"], binning["x_max"], binning["log_step"])
    efficiencies = stage_b["category_background_efficiencies"]
    bounds = [1.0, *efficiencies, 0.0]  # accepted background fraction at each category's upper and lower edge

    paths = [grid_report / "provenance/summary.json", grid_report / "provenance/status.json",
             grid_report / "tables/grid.csv", *extra_inputs]
    for mass in masses:
        for rho in couplings:
            point = point_roots[mass] / directories[rho]
            paths += [*points.paths(mass, rho, ("val",)), point / "preds_val.parquet",
                      point / "operating_points.json"]
    inputs = hash_files(sorted(set(paths)))
    preregistration = dict(study=str(study.directory), arguments=vars(args), settings=settings,
                           **{("stage_c" if stage_c else "stage_b"): stage_b},
                           evaluation_config=asdict(cfg), grid_report=str(grid_report), test_evaluated=False)
    open_report(out, preregistration, inputs, preregister=True)
    checks = [dict(check="phase9_grid_report_complete_and_unchanged", passed=True),
              dict(check="evaluation_config_and_k_factors_match_the_grid", passed=True),
              dict(check=f"categories use the grid's primary-seed ({grid['primary_seed']}) BDT scores", passed=True)]

    grid_table = pd.read_csv(grid_report / "tables/grid.csv")
    category_rows, bins_rows, z_rows, comparison_rows = [], [], [], []
    closure, yield_closure, fraction_offsets, boundary_rule = [], [], [], []
    columns = [*IDENTITY, *(c for cs in ROLES.values() for c in cs), "mcb1", "mcb2"]
    for mass in masses:
        for rho in couplings:
            point = point_roots[mass] / directories[rho]
            full = points.full(mass, rho)
            val = points.split(mass, rho, "val", columns)
            scores = pd.read_parquet(point / "preds_val.parquet", columns=["event_id", "sample_key", "bdt_score"])
            frame = val.merge(scores, on=["event_id", "sample_key"], how="left", validate="one_to_one")
            if len(scores) != len(val) or frame.bdt_score.isna().any():
                raise ValueError(f"m{mass} rho{rho:g}: BDT scores do not cover the validation events one to one")
            weights, _ = evaluation_weights(full, frame, stored_lumi=points.stored_lumi(mass, rho),
                                            target_lumi=cfg.lumi_pb_inv, signal_k_factor=cfg.signal_k_factor,
                                            background_k_factor=cfg.background_k_factor)
            score, target = frame.bdt_score.to_numpy(), frame.target.to_numpy()
            signal, background = target == 1, target == 0
            roles = role_arrays(frame)
            closure.append(max(mass_residual(roles["b1"], roles["c1"], frame.mcb1),
                               mass_residual(roles["b2"], roles["c1"], frame.mcb2)))
            x = closest_cb_mass(roles["b1"], roles["b2"], roles["c1"], mass)[0] / mass

            # Closure: one bin above the grid's frozen threshold reproduces its primary objective.
            reference = _grid_references(grid_table, grid, cfg, mass, rho)
            frozen = json.loads((point / "operating_points.json").read_text())["operating_points"][objective]
            if frozen["status"] != "valid":
                raise ValueError(f"m{mass} rho{rho:g}: the grid has no valid {objective} operating point")
            kept = score >= frozen["threshold"]
            s_kept, b_kept = weights[kept & signal].sum(), weights[kept & background].sum()
            yield_closure.append(max(abs(s_kept / frozen["metrics"]["S"] - 1), abs(b_kept / frozen["metrics"]["B"] - 1)))
            one_bin = asimov_significance([s_kept], [b_kept], normalization=norm)
            checks.append(dict(check=f"m{mass} rho{rho:g}: one bin above the frozen threshold reproduces the grid's "
                                     f"{objective}",
                               passed=bool(abs(one_bin / reference["bdt"][objective] - 1) <= settings["closure_tolerance"])))

            # Categories from background alone, then the c-b mass templates inside each.
            thresholds = score_boundaries(score[background], weights[background], efficiencies)
            category = np.searchsorted(thresholds, score, side="right")
            total_s, total_b = weights[signal].sum(), weights[background].sum()
            fraction_offsets += [abs(weights[background & (category >= k + 1)].sum() / total_b - efficiency)
                                 for k, efficiency in enumerate(efficiencies)]
            boundary_rule.append(boundaries_follow_rule(score[background], weights[background], thresholds,
                                                        efficiencies))
            fits = {name: ([], [], []) for name in FITS}
            templates = [(k, "categories_x_mass", category == k) for k in range(len(efficiencies) + 1)]
            templates.append((0, "single_category", np.ones(len(frame), dtype=bool)))
            for k, fit, inside in templates:
                merged = merged_templates(x, frame, weights, inside, edges, binning["min_background_neff"])
                bins_rows += [dict(mass=mass, rho_tc=rho, fit=fit, category=k, **row) for row in merged]
                for values, key in zip(fits[fit], ("S", "B", "mc_variance")):
                    values += [row[key] for row in merged]
                if fit != "categories_x_mass":
                    continue
                s, b = weights[inside & signal].sum(), weights[inside & background].sum()
                sum_w2 = (weights[inside & background] ** 2).sum()
                for values, value in zip(fits["categories_only"], (s, b, sum_w2)):
                    values.append(value)
                category_rows.append(dict(
                    mass=mass, rho_tc=rho, category=k, score_low=thresholds[k - 1] if k else 0.0,
                    score_high=thresholds[k] if k < len(thresholds) else 1.0,
                    background_fraction_design=bounds[k] - bounds[k + 1], background_fraction=b / total_b,
                    signal_fraction=s / total_s, S=s, B=b, S_over_B=s / b, background_mc=int((inside & background).sum()),
                    background_neff=b**2 / sum_w2, mass_bins=len(merged)))
            # Stage C's descriptive category sets: the same categories x c-b mass fit.
            for name, set_efficiencies in descriptive_sets.items():
                set_thresholds = score_boundaries(score[background], weights[background], set_efficiencies)
                set_category = np.searchsorted(set_thresholds, score, side="right")
                fraction_offsets += [abs(weights[background & (set_category >= k + 1)].sum() / total_b - efficiency)
                                     for k, efficiency in enumerate(set_efficiencies)]
                boundary_rule.append(boundaries_follow_rule(score[background], weights[background], set_thresholds,
                                                            set_efficiencies))
                fits[name] = ([], [], [])
                for k in range(len(set_efficiencies) + 1):
                    merged = merged_templates(x, frame, weights, set_category == k, edges, binning["min_background_neff"])
                    for values, key in zip(fits[name], ("S", "B", "mc_variance")):
                        values += [row[key] for row in merged]
            point_z = {}
            for fit, (s, b, mc) in fits.items():
                for shape in shapes:
                    for with_mc in (True, False):
                        z = asimov_significance(s, b, normalization=norm, shape=shape, mc_variance=mc if with_mc else None)
                        z_rows.append(dict(mass=mass, rho_tc=rho, fit=fit, shape_uncertainty=shape, mc_statistics=with_mc,
                                           bins=len(s), Z=z))
                        point_z[fit, shape, with_mc] = z
            primary = point_z["categories_x_mass", primary_shape, True]
            z_bdt, z_cuts = reference["bdt"][objective], reference["cuts"][objective]
            comparison_rows.append(dict(
                mass=mass, rho_tc=rho, Z_binned=primary, Z_single_category=point_z["single_category", primary_shape, True],
                Z_categories_only=point_z["categories_only", primary_shape, True],
                Z_infinite_mc=point_z["categories_x_mass", primary_shape, False], Z_grid_bdt=z_bdt, Z_grid_cuts=z_cuts,
                binned_over_bdt=primary / z_bdt, binned_over_cuts=primary / z_cuts,
                binned_over_single_category=primary / point_z["single_category", primary_shape, True],
                binned_over_categories_only=primary / point_z["categories_only", primary_shape, True],
                infinite_mc_over_primary=point_z["categories_x_mass", primary_shape, False] / primary))
            if stage_c:
                bdt = reference["bdt"]
                z_bdt_mc = asimov_significance([bdt.S], [bdt.B], normalization=norm,
                                               mc_variance=[bdt.B**2 / bdt.background_neff])
                comparison_rows[-1].update(
                    Z_grid_bdt_mc=z_bdt_mc, binned_over_bdt_mc=primary / z_bdt_mc,
                    **{f"Z_{name}": point_z[name, primary_shape, True] for name in descriptive_sets},
                    **{f"binned_over_{name}": primary / point_z[name, primary_shape, True] for name in descriptive_sets})
        print(f"m{mass}: {len(couplings)} couplings evaluated", flush=True)
    checks += [
        dict(check="BDT scores cover every validation event exactly once at every point", passed=True),
        dict(check=f"S and B above the frozen thresholds reproduce the saved operating points "
                   f"(max relative {max(yield_closure):.1e})", passed=bool(max(yield_closure) < 1e-6)),
        dict(check=f"category boundaries follow the rule: the lowest score keeping at most each designed "
                   f"background fraction (largest offset from the design {max(fraction_offsets):.1e}, set by single "
                   f"heavy events)", passed=all(boundary_rule)),
        dict(check=f"recomputed c-b masses reproduce stored mcb1/mcb2 (max |dm^2|/(E1+E2)^2 {max(closure):.1e})",
             passed=bool(max(closure) < 1e-5)),
        dict(check="no_test_events_read", passed=not reads_test(inputs))]
    if stage_c:
        reproduced = pd.DataFrame(comparison_rows).set_index(["mass", "rho_tc"]).Z_stage_b
        offset = float(np.max(np.abs(reproduced / stage_b_z.reindex(reproduced.index) - 1)))
        checks.append(dict(check=f"stage B's categories on the same rows reproduce stage B's primary Z (max relative "
                                 f"{offset:.1e})", passed=bool(offset < 1e-9)))
    tables = dict(checks=pd.DataFrame(checks), categories=pd.DataFrame(category_rows), bins=pd.DataFrame(bins_rows),
                  significance=pd.DataFrame(z_rows), comparison=pd.DataFrame(comparison_rows))
    write_tables(out, tables)
    if not tables["checks"].passed.all():
        failed = tables["checks"].loc[~tables["checks"].passed, "check"]
        raise ValueError("Binned-fit contract checks failed: " + "; ".join(failed))
    comparison = tables["comparison"]
    medians = ["binned_over_bdt", "binned_over_single_category", "binned_over_categories_only",
               "infinite_mc_over_primary"]
    if stage_c:
        medians += ["binned_over_bdt_mc", *(f"binned_over_{name}" for name in descriptive_sets)]
    summary = dict(schema_version=2, phases=[9], stage=STAGE_C if stage_c else STAGE_B,
                   tagging=grid.get("tagging", "direct"), matching_acceptance=grid.get("matching_acceptance"),
                   flavour_overlap=grid.get("flavour_overlap"),
                   settings=settings,
                   **{("stage_c" if stage_c else "stage_b"): stage_b},
                   **({"stage_b_report": str(stage_b_report)} if stage_c else {}),
                   grid_settings=grid_settings, evaluation_config=asdict(cfg), grid_report=str(grid_report),
                   primary_seed=grid["primary_seed"],
                   primary=dict(fit=FITS[0], jets=stage_b["jets"], templates=stage_b["templates"],
                                shape_uncertainty=primary_shape, mc_statistics=True),
                   **{f"median_{name}": float(comparison[name].median()) for name in medians},
                   test_evaluated=False, status="checkpoint_ready",
                   note=("Stage C: finer BDT-score categories x c-b mass on the truth-tagged grid's validation events; "
                         "compared with the single-bin BDT carrying the same MC-statistics term." if stage_c else
                         "Stage B: BDT-score categories x c-b mass on validation. The background-MC decision and a "
                         "measured per-bin shape uncertainty follow this checkpoint."))
    write_json(out / "provenance/summary.json", summary)
    save_binned_figures(out)
    finish_report(out, "04_physics_results.ipynb", inputs)
    if stage_c:
        print(f"Stage C: median Z_binned / single-bin BDT with MC statistics = "
              f"{summary['median_binned_over_bdt_mc']:.2f}; over stage B = {summary['median_binned_over_stage_b']:.2f}",
              flush=True)
    print(f"Stage {args.stage}: median Z_binned / Z_grid_bdt = {summary['median_binned_over_bdt']:.2f}; "
          f"median gain over one category = {summary['median_binned_over_single_category']:.2f}; "
          f"median infinite-MC gain = {summary['median_infinite_mc_over_primary']:.2f}", flush=True)


def mass_residual(first, second, stored):
    """Largest |m^2 - stored m^2| / (E1 + E2)^2 of a jet pair.

    The stored masses are float32 results; their rounding error scales with the pair's
    energy, not its mass, so a relative mass test would fail for nearly collinear jets.
    """
    energy = sum(np.sqrt((pt * np.cosh(eta)) ** 2 + mass**2) for pt, eta, _, mass in (first, second))
    stored = np.asarray(stored, dtype=float)
    return float(np.max(np.abs(pair_mass(first, second) ** 2 - stored**2) / energy**2))


def reads_test(paths):
    """Whether any input is a test split: direct (test_sig*.parquet, preds_test*) or truth-tag (*_test.parquet)."""
    names = [Path(p).name for p in paths]
    return any(n.startswith(("test_", "preds_test")) or n.endswith("_test.parquet") or n == "test.parquet"
               for n in names)


def _histogram(x, mask, weights, edges):
    return np.histogram(x[mask], bins=edges, weights=weights[mask])[0]


def _stored_lumi(files, mass):
    return float(json.loads(files.meta(mass).read_text())["lumi"])
