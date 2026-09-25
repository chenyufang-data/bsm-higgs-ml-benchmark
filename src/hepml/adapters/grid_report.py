"""Notebook/PDF views of the Phase 9 grid; reads saved artifacts only."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from hepml.adapters.evaluation_report import display_figure, read_evaluation_report
from hepml.domain.metrics import OBJECTIVES


def default_grid_report():
    return Path(os.environ.get("HEPML_GRID_REPORT", "outputs/cg_bbc/reports/phase9-grid-v1"))


def acceptance_note(summary):
    """Scope text for reports built on rows with the matching acceptance applied, else None."""
    acceptance = summary.get("matching_acceptance")
    if not acceptance:
        return None
    note = ("**Matching acceptance applied** (`matching_acceptance`): only events the MLM matching accepted at the "
            "central merging scale are used (non-zero central merging weight, identified per ROOT file), each sample "
            f"weighted by its Pythia merged cross section over its accepted count (`{acceptance['counts']}`).")
    overlap = summary.get("flavour_overlap")
    if overlap:
        rules = ", ".join(f"{key}: {rule}" for key, rule in overlap["remove"].items())
        note += (" **Heavy-flavour overlap removed** (`flavour_overlap`): events another sample also generates are "
                 f"dropped by their outgoing hard-process partons ({rules}).")
    return note


def _grid(summary, tables, *, method="bdt", seed=None):
    grid = tables["grid"]
    grid = grid[grid.method == method]
    if method == "bdt":
        grid = grid[grid.seed == (summary["primary_seed"] if seed is None else seed)]
    return grid


def grid_matrix(frame, value, masses, couplings):
    """Coupling-by-mass array of `value`; unsupported points are NaN."""
    table = frame.pivot_table(index="rho_tc", columns="mass", values=value, aggfunc="first")
    return table.reindex(index=couplings, columns=masses).to_numpy(dtype=float)


def heatmap(ax, values, masses, couplings, title, *, cmap="viridis", fmt="{:.2g}", center=None, log=False):
    """Coupling-by-mass map; significances span decades, so they use a log colour scale."""
    positive = values[np.isfinite(values) & (values > 0)]
    masked = np.ma.masked_invalid(np.where(values > 0, values, np.nan) if log else values)
    kwargs = {}
    if center is not None:
        spread = np.nanmax(np.abs(values - center)) if np.isfinite(values).any() else 1.0
        kwargs = dict(vmin=center - spread, vmax=center + spread)
    elif log and len(positive):
        kwargs = dict(norm=LogNorm(vmin=positive.min(), vmax=positive.max()))
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("#d9d9d9")
    image = ax.imshow(masked, origin="lower", aspect="auto", cmap=colormap, **kwargs)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "black"
            if np.isfinite(value):
                red, green, blue, _ = colormap(image.norm(value))
                color = "white" if 0.299 * red + 0.587 * green + 0.114 * blue < 0.5 else "black"
            ax.text(column, row, "n/a" if not np.isfinite(value) else fmt.format(value), ha="center", va="center",
                    fontsize=7, color=color)
    ax.set_xticks(range(len(masses)), [str(m) for m in masses])
    ax.set_yticks(range(len(couplings)), [f"{c:g}" for c in couplings])
    ax.set(xlabel="Charged-Higgs mass [GeV]", ylabel="rho_tc", title=title)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def sensitivity_grid_figure(directory):
    summary, tables = read_evaluation_report(directory)
    settings = summary["settings"]
    masses, couplings = settings["masses"], settings["couplings"]
    grid = _grid(summary, tables)
    fig, axes = plt.subplots(1, len(OBJECTIVES), figsize=(6 * len(OBJECTIVES), 4.2))
    for ax, objective in zip(axes, OBJECTIVES):
        rows = grid[(grid.objective == objective) & (grid.status == "valid")]
        heatmap(ax, grid_matrix(rows, objective, masses, couplings), masses, couplings,
                f"{objective} at its own validation optimum", log=True)
    fig.suptitle(f"Mass-specific BDTs trained at rho_tc={settings['training_coupling']:g} (seed "
                 f"{summary['primary_seed']}), {summary['evaluation_config']['lumi_pb_inv'] / 1000:g} fb^-1, validation; "
                 "grey: no MC-supported operating point")
    fig.tight_layout()
    return fig


def cut_comparison_figure(directory):
    summary, tables = read_evaluation_report(directory)
    settings, objective = summary["settings"], summary["evaluation_config"]["primary_objective"]
    masses, couplings = settings["masses"], settings["couplings"]
    values = {}
    for method in ("bdt", "cuts"):
        rows = _grid(summary, tables, method=method)
        rows = rows[(rows.objective == objective) & (rows.status == "valid")]
        values[method] = grid_matrix(rows, objective, masses, couplings)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = values["bdt"] / values["cuts"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    heatmap(axes[0], values["cuts"], masses, couplings, f"Cut reference: {objective}", log=True)
    heatmap(axes[1], ratio, masses, couplings, f"BDT / cut reference: {objective}", cmap="RdBu", center=1.0,
            fmt="{:.2f}")
    fig.suptitle("Each method at its own validation optimum on the same events; grey: either method unsupported")
    fig.tight_layout()
    return fig


def auc_figure(directory):
    summary, tables = read_evaluation_report(directory)
    settings, objective = summary["settings"], summary["evaluation_config"]["primary_objective"]
    masses, couplings = settings["masses"], settings["couplings"]
    auc = tables["auc"]
    primary = auc[auc.seed == summary["primary_seed"]]
    grid = tables["grid"]
    bdt = grid[(grid.method == "bdt") & (grid.objective == objective) & (grid.status == "valid")]
    spread = bdt.groupby(["mass", "rho_tc"])[objective].agg(lambda x: x.max() - x.min() if len(x) > 1 else np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    heatmap(axes[0], grid_matrix(primary, "auc_weighted", masses, couplings), masses, couplings,
            f"Weighted validation AUC (seed {summary['primary_seed']})", fmt="{:.3f}")
    heatmap(axes[1], grid_matrix(spread.rename("spread").reset_index(), "spread", masses, couplings), masses, couplings,
            f"{objective}: max - min over seeds {settings['seeds']}", cmap="Greys")
    fig.tight_layout()
    return fig


def sensitivity_vs_mass_figure(directory):
    summary, tables = read_evaluation_report(directory)
    settings, objective = summary["settings"], summary["evaluation_config"]["primary_objective"]
    grid = tables["grid"]
    valid = grid[(grid.objective == objective) & (grid.status == "valid")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for index, rho in enumerate(settings["couplings"]):
        color = f"C{index}"
        bdt = valid[(valid.method == "bdt") & np.isclose(valid.rho_tc, rho)]
        primary = bdt[bdt.seed == summary["primary_seed"]].sort_values("mass")
        band = bdt.groupby("mass")[objective].agg(["min", "max"]).reset_index()
        ax.plot(primary.mass, primary[objective], color=color, marker="o", label=f"BDT, rho_tc={rho:g}")
        ax.fill_between(band["mass"], band["min"], band["max"], color=color, alpha=0.15)
        cuts = valid[(valid.method == "cuts") & np.isclose(valid.rho_tc, rho)].sort_values("mass")
        ax.plot(cuts.mass, cuts[objective], color=color, linestyle="--", marker="s", markersize=3)
    ax.set_yscale("log")
    ax.set(xlabel="Charged-Higgs mass [GeV]", ylabel=f"{objective} at its validation optimum",
           title=f"Solid: BDT seed {summary['primary_seed']} (band: seed range); dashed: cut reference")
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


FIGURES = {"sensitivity_grid": sensitivity_grid_figure, "bdt_vs_cuts": cut_comparison_figure,
           "auc_and_seed_spread": auc_figure, "sensitivity_vs_mass": sensitivity_vs_mass_figure}


def save_grid_figures(directory):
    directory = Path(directory)
    for name, build in FIGURES.items():
        figure = build(directory)
        figure.savefig(directory / "figures" / f"{name}.pdf")
        plt.close(figure)


def display_grid_section(directory, section):
    from IPython.display import Markdown, display

    summary, tables = read_evaluation_report(directory)
    if summary.get("stage") in {"binned_fit_A", "binned_fit_B", "binned_fit_C"}:
        from hepml.adapters.binned_fit_report import display_binned_section

        return display_binned_section(directory, section)
    settings, cfg = summary["settings"], summary["evaluation_config"]
    objective = cfg["primary_objective"]

    def prose(text):
        display(Markdown(text))

    def table(frame):
        # Significances span many decades: significant figures, not fixed decimals.
        floats = frame.select_dtypes("float").columns
        display(frame.style.format({column: "{:.4g}" for column in floats}, na_rep="n/a"))

    if section == "scope":
        prose(
            f"**Phase 9:** expected sensitivity at every physics point, masses {settings['masses']} GeV and rho_tc "
            f"{settings['couplings']}. One mass-specific BDT per mass is trained on its rho_tc="
            f"{settings['training_coupling']:g} sample (the cut-based benchmark coupling) with the Phase 2 recipe. "
            f"**Test evaluated: {summary['test_evaluated']}.** All numbers use validation events."
        )
        if summary.get("tagging") == "truth":
            prose(
                "**Truth tagging** (separate from the direct-tag grid): every event with three jets in acceptance "
                "contributes one drawn passing tag outcome, weighted by its pass probability, which is also the fit "
                f"weight. Splits come from the closure's registry (`{summary['registry']}`); the closure is "
                f"`{summary['closure_report']}`, and the direct-tag grid it follows is `{summary['direct_grid_report']}`."
            )
        if acceptance_note(summary):
            prose(acceptance_note(summary))
        prose(
            "Every point is scored on its own genuine signal sample with its own merged cross section; no rho "
            "independence or rate rescaling is assumed. Each point's three operating points come from its own "
            f"validation events. Seed {summary['primary_seed']} is the primary model whose thresholds are frozen for "
            f"the single later test evaluation; seeds {settings['seeds']} show training variation. "
            f"Evaluation: {cfg['lumi_pb_inv'] / 1000:g} fb^-1, signal k={cfg['signal_k_factor']:g}, background "
            f"k={cfg['background_k_factor']:g}, each applied once."
        )
        prose(
            "Alternative (mass, rho_tc) hypotheses reuse the same background events and are mutually exclusive, so "
            "their sensitivities are never summed or combined. Values are expected local sensitivities."
        )
    elif section == "inputs":
        prose("Rates and MC support per point (cross sections as written in the production manifest):")
        table(tables["rates"])
        prose("Samples added to the common registry (deterministic hash-rank, existing assignments unchanged):")
        table(tables["registry_extension"])
    elif section == "checks":
        table(tables["checks"])
    elif section == "results":
        display_figure(sensitivity_grid_figure(directory))
        display_figure(sensitivity_vs_mass_figure(directory))
        display_figure(cut_comparison_figure(directory))
        display_figure(auc_figure(directory))
        grid = _grid(summary, tables)
        primary = grid[grid.objective == objective][["mass", "rho_tc", "status", "threshold", "S", "B", "S_over_B",
                                                     *OBJECTIVES, "background_neff", "signal_efficiency"]]
        prose(f"Primary objective {objective}, seed {summary['primary_seed']}:")
        table(primary.sort_values(["rho_tc", "mass"]))
    elif section == "thresholds":
        grid = tables["grid"]
        unsupported = grid[grid.status != "valid"][["method", "mass", "rho_tc", "seed", "objective", "status"]]
        prose(
            f"Operating points must keep background N_eff >= {cfg['min_background_neff']:g}, at least one MC event per "
            "background process and one signal event. Points without such a threshold are reported as unsupported, "
            "never extrapolated."
        )
        table(unsupported if len(unsupported) else pd.DataFrame(dict(status=["every point has a supported optimum"])))
        support = tables["process_support"]
        prose(f"Per-process MC support at the {objective} optimum, seed {summary['primary_seed']}:")
        table(support[(support.objective == objective) & (support.seed == summary["primary_seed"])])
    elif section == "checkpoint":
        prose(f"**Status: {summary['status']}.** {summary['note']}")
        prose(
            "Next: review this validation grid. Then evaluate the frozen primary-seed thresholds once on the test "
            "partition, unchanged; no threshold, model or seed may be re-chosen after that."
        )
    else:
        raise ValueError(f"Unknown grid report section: {section}")
