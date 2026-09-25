"""Notebook/PDF views of the binned shape-fit checks (stages A and B); reads saved artifacts only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hepml.adapters.evaluation_report import display_figure, read_evaluation_report
from hepml.adapters.grid_report import acceptance_note, grid_matrix, heatmap

PROCESS_COLORS = {"bbc": "C0", "bbj": "C1", "bjj": "C2", "ccj": "C3", "cjj": "C4"}
TITLES = {"binned_fit_A": "Stage A: c-b mass shape fit on validation events, no BDT cut",
          "binned_fit_B": "Stage B: BDT-score categories x c-b mass on validation events",
          "binned_fit_C": "Stage C: finer BDT-score categories x c-b mass, truth-tagged grid"}
# Stage B fits in the fixed order of the report; the primary first.
FIT_LABELS = {"categories_x_mass": "BDT categories x c-b mass (primary)",
              "single_category": "c-b mass only (one category)",
              "categories_only": "BDT categories only (no c-b mass)"}


def _showcase(summary):
    """Three masses at the grid's training coupling, low to high."""
    grid = summary["grid_settings"]
    masses = grid["masses"]
    return [masses[0], masses[len(masses) // 2], masses[-1]], grid["training_coupling"]


def _primary_rows(frame, summary, **extra):
    primary = summary["primary"]
    keep = (frame.jets == primary["jets"]) & (frame.templates == primary["templates"])
    for key, value in extra.items():
        keep &= frame[key] == value
    return frame[keep]


def calibration_figure(directory):
    _, tables = read_evaluation_report(directory)
    calibration = tables["calibration"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if len(calibration):
        for kind, color in (("b", "C0"), ("c", "C3")):
            rows = calibration[calibration.kind == kind]
            ax.plot(rows.node_pt, rows.median_parton_over_reco, color=color, marker="o", label=f"{kind}-tagged jets")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set(xlabel="Reconstructed jet pT [GeV] (bin median)", ylabel="median parton pT / reconstructed pT",
           title="Jet response from training-split signal; linear in log pT between points")
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def templates_figure(directory):
    summary, tables = read_evaluation_report(directory)
    masses, coupling = _showcase(summary)
    bins = _primary_rows(tables["bins"], summary)
    process_columns = [c for c in bins.columns if c.startswith("B_")]
    fig, axes = plt.subplots(2, len(masses), figsize=(6 * len(masses), 7), sharex=False,
                             gridspec_kw=dict(height_ratios=[3, 1]))
    for column, mass in enumerate(masses):
        rows = bins[(bins.mass == mass) & np.isclose(bins.rho_tc, coupling)].sort_values("bin")
        low = np.maximum(rows.x_low.to_numpy(), 0.1)  # underflow drawn from x = 0.1
        high = np.minimum(rows.x_high.to_numpy(), 4.0)  # overflow drawn up to x = 4
        edges = np.r_[low, high[-1]]
        top, bottom = axes[0, column], axes[1, column]
        for name in process_columns:
            label = name.split("_")[-1]
            top.stairs(rows[name].to_numpy(), edges, color=PROCESS_COLORS.get(label, "grey"), label=label)
        top.stairs(rows.B.to_numpy(), edges, color="black", linewidth=1.5, label="total background")
        top.stairs(rows.S.to_numpy(), edges, color="black", linestyle="--", linewidth=1.5, label="signal")
        top.set(yscale="log", xscale="log", ylabel="expected events per bin",
                title=f"m{mass}, rho_tc={coupling:g}: {summary['primary']['jets']} jets")
        top.axvline(1.0, color="grey", linestyle=":")
        top.legend(fontsize=7, ncol=2)
        bottom.stairs(rows.S_over_B.to_numpy(), edges, color="black")
        bottom.set(xscale="log", yscale="log", xlabel="c-b mass / tested mass (closest pairing)", ylabel="S/B")
        bottom.axvline(1.0, color="grey", linestyle=":")
        for ax in (top, bottom):
            ax.grid(alpha=0.2, which="both")
    fig.suptitle("Validation templates at 500 fb^-1 after merging to background N_eff >= "
                 f"{summary['settings']['binning']['min_background_neff']}")
    fig.tight_layout()
    return fig


def significance_figure(directory):
    summary, tables = read_evaluation_report(directory)
    grid = summary["grid_settings"]
    masses, couplings = grid["masses"], grid["couplings"]
    comparison = tables["comparison"]
    shape = summary["primary"]["shape_uncertainty"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2))
    heatmap(axes[0], grid_matrix(comparison, "Z_binned", masses, couplings), masses, couplings,
            f"Binned fit Z (shared 5% + {shape:.0%} per bin + MC stat)", log=True)
    heatmap(axes[1], grid_matrix(comparison, "binned_over_bdt", masses, couplings), masses, couplings,
            "Binned fit / single-bin BDT (grid)", log=True, fmt="{:.2f}")
    heatmap(axes[2], grid_matrix(comparison, "infinite_mc_over_primary", masses, couplings), masses, couplings,
            "Infinite-MC limit / primary", log=True, fmt="{:.2f}")
    fig.suptitle(TITLES[summary["stage"]])
    fig.tight_layout()
    return fig


def dependence_figure(directory):
    summary, tables = read_evaluation_report(directory)
    masses, coupling = _showcase(summary)
    z = tables["significance"]
    comparison = tables["comparison"]
    jets = summary["primary"]["jets"]
    fig, axes = plt.subplots(1, len(masses), figsize=(6 * len(masses), 4.2))
    for ax, mass in zip(axes, masses):
        rows = z[(z.mass == mass) & np.isclose(z.rho_tc, coupling) & (z.jets == jets)]
        for template, with_mc, style, label in (("validation", True, "-", "validation templates"),
                                                ("train+validation", True, "--", "train+validation templates (~9x MC)"),
                                                ("validation", False, ":", "infinite-MC limit")):
            line = rows[(rows.templates == template) & (rows.mc_statistics == with_mc)].sort_values("shape_uncertainty")
            ax.plot(100 * line.shape_uncertainty, line.Z, linestyle=style, marker="o", color="C0", label=label)
        reference = comparison[(comparison.mass == mass) & np.isclose(comparison.rho_tc, coupling)].iloc[0]
        ax.axhline(reference.Z_grid_bdt, color="C3", linestyle="-.", label="single-bin BDT (grid)")
        ax.axhline(reference.Z_grid_cuts, color="C2", linestyle="-.", label="single-bin cut reference (grid)")
        ax.axvline(100 * summary["primary"]["shape_uncertainty"], color="grey", linewidth=0.8)
        ax.set(yscale="log", xlabel="independent per-bin shape uncertainty [%]", ylabel="Z",
               title=f"m{mass}, rho_tc={coupling:g}")
        ax.grid(alpha=0.2, which="both")
        ax.legend(fontsize=7)
    fig.suptitle("Shared 5% normalization in every case; grey line: primary shape term")
    fig.tight_layout()
    return fig


def category_design(summary):
    """The category design a stage-B or stage-C report was run with."""
    return summary["stage_c"] if summary["stage"] == "binned_fit_C" else summary["stage_b"]


def category_labels(summary):
    """Each score category by the share of background it holds, lowest score first."""
    bounds = [1.0, *category_design(summary)["category_background_efficiencies"], 0.0]
    labels = []
    for k, (high, low) in enumerate(zip(bounds, bounds[1:])):
        if high == 1.0:
            labels.append(f"C{k}: lowest-score {high - low:.0%} of background")
        elif low == 0.0:
            labels.append(f"C{k}: highest-score {high:.0%} of background")
        else:
            labels.append(f"C{k}: background quantile {high:.0%} to {low:.0%}")
    return labels


def category_templates_figure(directory):
    summary, tables = read_evaluation_report(directory)
    masses, coupling = _showcase(summary)
    bins = tables["bins"][tables["bins"].fit == "categories_x_mass"]
    labels = category_labels(summary)
    # Ordered categories: one hue, light (low score) to dark (high score).
    colors = plt.cm.Blues(np.linspace(0.4, 0.95, len(labels)))
    fig, axes = plt.subplots(2, len(masses), figsize=(6 * len(masses), 7), gridspec_kw=dict(height_ratios=[3, 2]))
    for column, mass in enumerate(masses):
        top, bottom = axes[0, column], axes[1, column]
        at = bins[(bins.mass == mass) & np.isclose(bins.rho_tc, coupling)]
        for k, (label, color) in enumerate(zip(labels, colors)):
            rows = at[at.category == k].sort_values("bin")
            low = np.maximum(rows.x_low.to_numpy(), 0.1)
            edges = np.r_[low, min(rows.x_high.to_numpy()[-1], 4.0)]
            top.stairs(rows.B.to_numpy(), edges, color=color, linewidth=1.5, label=f"background, {label}")
            top.stairs(rows.S.to_numpy(), edges, color=color, linestyle="--", linewidth=1.5)
            bottom.stairs(rows.S_over_B.to_numpy(), edges, color=color, linewidth=1.5, label=label)
        top.set(yscale="log", xscale="log", ylabel="expected events per bin",
                title=f"m{mass}, rho_tc={coupling:g}: background solid, signal dashed")
        bottom.set(xscale="log", yscale="log", xlabel="c-b mass / tested mass (closest pairing)", ylabel="S/B")
        for ax in (top, bottom):
            ax.axvline(1.0, color="grey", linestyle=":")
            ax.grid(alpha=0.2, which="both")
        bottom.legend(fontsize=7)
    fig.suptitle("Validation templates per BDT-score category at 500 fb^-1, merged to background N_eff >= "
                 f"{summary['settings']['binning']['min_background_neff']} in each category")
    fig.tight_layout()
    return fig


def fit_comparison_figure(directory):
    summary, tables = read_evaluation_report(directory)
    coupling = summary["grid_settings"]["training_coupling"]
    rows = tables["comparison"][np.isclose(tables["comparison"].rho_tc, coupling)].sort_values("mass")
    lines = [("Z_binned", FIT_LABELS["categories_x_mass"], "C0", "-", "o"),
             ("Z_single_category", FIT_LABELS["single_category"], "C1", "--", "s"),
             ("Z_categories_only", FIT_LABELS["categories_only"], "C2", "--", "^"),
             ("Z_infinite_mc", "primary fit, infinite-MC limit", "C0", ":", "o"),
             ("Z_grid_bdt", "single-bin BDT (grid)", "C3", "-.", "D"),
             ("Z_grid_cuts", "single-bin cut reference (grid)", "C7", "-.", "v"),
             ("Z_grid_bdt_mc", "single-bin BDT with the MC-statistics term", "C3", ":", "D"),
             ("Z_stage_b", "stage B categories (50/20/5%)", "C4", "--", "x"),
             ("Z_four_categories", "four categories (20/5/1%)", "C5", "--", "+")]
    lines = [line for line in lines if line[0] in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for column, label, color, style, marker in lines:
        axes[0].plot(rows.mass, rows[column], color=color, linestyle=style, marker=marker, label=label)
        axes[1].plot(rows.mass, rows[column] / rows.Z_grid_bdt, color=color, linestyle=style, marker=marker, label=label)
    shape = summary["primary"]["shape_uncertainty"]
    axes[0].set(yscale="log", xlabel="tested mass [GeV]", ylabel="Z",
                title=f"rho_tc={coupling:g}: shared 5% + {shape:.0%} per bin + MC statistics")
    axes[1].set(yscale="log", xlabel="tested mass [GeV]", ylabel="Z / single-bin BDT",
                title="Relative to the grid's single-bin BDT")
    for ax in axes:
        ax.grid(alpha=0.2, which="both")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    return fig


def fit_dependence_figure(directory):
    summary, tables = read_evaluation_report(directory)
    masses, coupling = _showcase(summary)
    z, comparison = tables["significance"], tables["comparison"]
    lines = [("categories_x_mass", True, "C0", "-", FIT_LABELS["categories_x_mass"]),
             ("categories_x_mass", False, "C0", ":", "primary fit, infinite-MC limit"),
             ("single_category", True, "C1", "--", FIT_LABELS["single_category"]),
             ("categories_only", True, "C2", "--", FIT_LABELS["categories_only"])]
    fig, axes = plt.subplots(1, len(masses), figsize=(6 * len(masses), 4.2))
    for ax, mass in zip(axes, masses):
        rows = z[(z.mass == mass) & np.isclose(z.rho_tc, coupling)]
        for fit, with_mc, color, style, label in lines:
            line = rows[(rows.fit == fit) & (rows.mc_statistics == with_mc)].sort_values("shape_uncertainty")
            ax.plot(100 * line.shape_uncertainty, line.Z, color=color, linestyle=style, marker="o", label=label)
        reference = comparison[(comparison.mass == mass) & np.isclose(comparison.rho_tc, coupling)].iloc[0]
        ax.axhline(reference.Z_grid_bdt, color="C3", linestyle="-.", label="single-bin BDT (grid)")
        ax.axvline(100 * summary["primary"]["shape_uncertainty"], color="grey", linewidth=0.8)
        ax.set(yscale="log", xlabel="independent per-bin shape uncertainty [%]", ylabel="Z",
               title=f"m{mass}, rho_tc={coupling:g}")
        ax.grid(alpha=0.2, which="both")
        ax.legend(fontsize=7)
    fig.suptitle("Shared 5% normalization in every case; grey line: primary shape term")
    fig.tight_layout()
    return fig


FIGURES = {"binned_fit_A": {"calibration": calibration_figure, "cb_mass_templates": templates_figure,
                            "binned_significance": significance_figure, "uncertainty_dependence": dependence_figure},
           "binned_fit_B": {"category_templates": category_templates_figure, "binned_significance": significance_figure,
                            "fit_comparison": fit_comparison_figure, "uncertainty_dependence": fit_dependence_figure}}
FIGURES["binned_fit_C"] = FIGURES["binned_fit_B"]


def save_binned_figures(directory):
    directory = Path(directory)
    summary, _ = read_evaluation_report(directory)
    for name, build in FIGURES[summary["stage"]].items():
        figure = build(directory)
        figure.savefig(directory / "figures" / f"{name}.pdf")
        plt.close(figure)


def display_binned_section(directory, section):
    from IPython.display import Markdown, display

    summary, tables = read_evaluation_report(directory)
    if summary["stage"] in ("binned_fit_B", "binned_fit_C"):
        return _display_stage_b(directory, section, summary, tables)
    settings, primary = summary["settings"], summary["primary"]

    def prose(text):
        display(Markdown(text))

    def table(frame):
        floats = frame.select_dtypes("float").columns
        display(frame.style.format({column: "{:.4g}" for column in floats}, na_rep="n/a"))

    comparison = tables["comparison"]
    if section == "scope":
        prose(
            "**Binned shape fit, stage A:** how much sensitivity does a fit of the c-b mass distribution add over a "
            "single counting bin? Same 40 points, validation events, 500 fb^-1 weights and k-factors as the Phase 9 "
            "grid, without any BDT cut. **Test evaluated: False.**"
        )
        if acceptance_note(summary):
            prose(acceptance_note(summary))
        prose(
            "Observable: the c-b mass of the c-tagged jet with the b-tagged jet whose c-b mass is closest to the tested "
            f"mass, from **{primary['jets']}** jets. Bins are log-spaced in c-b mass / tested mass from "
            f"{settings['binning']['x_min']} to {settings['binning']['x_max']} (step {settings['binning']['log_step']}), "
            "with underflow and overflow kept, and merged upwards until each bin has background N_eff >= "
            f"{settings['binning']['min_background_neff']}; only background decides the binning."
        )
        prose(
            "Significance: the Gaussian limit of the binned profile likelihood, Z^2 = s^T C^-1 s, with a shared "
            f"{settings['normalization_uncertainty']:.0%} background normalization, an independent per-bin shape "
            f"uncertainty ({', '.join(f'{s:.0%}' for s in settings['shape_uncertainties'])}; primary "
            f"{primary['shape_uncertainty']:.0%}) and the per-bin MC-statistics variance of the background template. "
            "Without shape or MC terms, a background-only sideband would pin the normalization and remove the 5% "
            "almost entirely; the shape and MC terms stop the fit from over-trusting the simulation."
        )
    elif section == "inputs":
        if len(tables["calibration"]):
            display_figure(calibration_figure(directory))
            table(tables["calibration"])
        prose("Signal c-b mass (closest pairing, validation): peak = median / tested mass; half-width = 68% half-width:")
        table(tables["signal_peak"])
    elif section == "checks":
        table(tables["checks"])
    elif section == "results":
        display_figure(significance_figure(directory))
        display_figure(dependence_figure(directory))
        table(comparison.sort_values(["rho_tc", "mass"]))
    elif section == "thresholds":
        display_figure(templates_figure(directory))
        masses, coupling = _showcase(summary)
        bins = _primary_rows(tables["bins"], summary)
        prose(f"Merged bins at rho_tc={coupling:g} ({primary['jets']} jets, validation templates):")
        table(bins[bins.mass.isin(masses) & np.isclose(bins.rho_tc, coupling)]
              [["mass", "bin", "x_low", "x_high", "S", "B", "S_over_B", "background_neff", "mc_variance"]])
    elif section == "checkpoint":
        prose(
            f"Median over the 40 points: binned fit / single-bin BDT = **{summary['median_binned_over_bdt']:.2f}**; "
            f"infinite-MC limit / primary = **{summary['median_infinite_mc_over_primary']:.2f}** (the gain that "
            "unlimited background MC would allow with the same fit)."
        )
        prose(f"**Status: {summary['status']}.** {summary['note']}")
    else:
        raise ValueError(f"Unknown binned-fit report section: {section}")


def _display_stage_b(directory, section, summary, tables):
    from IPython.display import Markdown, display

    settings, primary, stage_b = summary["settings"], summary["primary"], category_design(summary)

    def prose(text):
        display(Markdown(text))

    def table(frame):
        floats = frame.select_dtypes("float").columns
        display(frame.style.format({column: "{:.4g}" for column in floats}, na_rep="n/a"))

    comparison = tables["comparison"]
    if section == "scope":
        if summary["stage"] == "binned_fit_C":
            prose(
                "**Stage C** repeats stage B on the truth-tagged grid with finer high-score categories: truth tagging "
                "moved the single-bin optimum to about 1% background efficiency, below stage B's top category. The "
                "headline comparison is the single-bin BDT carrying the same MC-statistics term; stage B's categories "
                f"on the same rows must reproduce stage B (`{summary['stage_b_report']}`)."
            )
        prose(
            "**Binned shape fit, stage B:** does splitting events by BDT score add sensitivity to the c-b mass fit? "
            "Same 40 points, validation events, 500 fb^-1 weights and k-factors as the Phase 9 grid. "
            "**Test evaluated: False.**"
        )
        if acceptance_note(summary):
            prose(acceptance_note(summary))
        prose(
            f"Categories: the grid's seed-{stage_b['model_seed']} BDT score, with boundaries where the weighted "
            "validation background above them is "
            f"{', '.join(f'{e:.0%}' for e in stage_b['category_background_efficiencies'])}; only background sets them. "
            "In each category, the closest-pairing c-b mass of reconstructed jets uses stage A's bins, merged within the "
            f"category to background N_eff >= {settings['binning']['min_background_neff']}."
        )
        prose(
            "Significance: stage A's model, Z^2 = s^T C^-1 s, with one "
            f"{settings['normalization_uncertainty']:.0%} normalization shared by every bin of every category, an "
            f"independent per-bin shape term ({primary['shape_uncertainty']:.0%} primary) and per-bin MC statistics. "
            "Descriptive fits: the same c-b mass fit in one category (stage A's observable, reconstructed jets), the "
            "categories without the c-b mass, and the infinite-MC limit."
        )
    elif section == "inputs":
        prose("Categories at every point: score range, share of background and signal, and background MC support:")
        table(tables["categories"])
    elif section == "checks":
        table(tables["checks"])
    elif section == "results":
        display_figure(fit_comparison_figure(directory))
        display_figure(significance_figure(directory))
        display_figure(fit_dependence_figure(directory))
        table(comparison.sort_values(["rho_tc", "mass"]))
    elif section == "thresholds":
        display_figure(category_templates_figure(directory))
        masses, coupling = _showcase(summary)
        bins = tables["bins"]
        prose(f"Merged bins per category at rho_tc={coupling:g}:")
        table(bins[(bins.fit == "categories_x_mass") & bins.mass.isin(masses) & np.isclose(bins.rho_tc, coupling)]
              [["mass", "category", "bin", "x_low", "x_high", "S", "B", "S_over_B", "background_neff"]])
    elif section == "checkpoint":
        if summary["stage"] == "binned_fit_C":
            prose(
                "Medians over the 40 points: primary fit / single-bin BDT with the MC-statistics term = "
                f"**{summary['median_binned_over_bdt_mc']:.2f}**; primary / stage B categories = "
                f"**{summary['median_binned_over_stage_b']:.2f}**; primary / four categories = "
                f"**{summary['median_binned_over_four_categories']:.2f}**."
            )
        prose(
            "Medians over the 40 points: primary fit / single-bin BDT = "
            f"**{summary['median_binned_over_bdt']:.2f}**; primary / c-b mass in one category = "
            f"**{summary['median_binned_over_single_category']:.2f}**; primary / categories alone = "
            f"**{summary['median_binned_over_categories_only']:.2f}**; infinite-MC limit / primary = "
            f"**{summary['median_infinite_mc_over_primary']:.2f}**."
        )
        prose(f"**Status: {summary['status']}.** {summary['note']}")
    else:
        raise ValueError(f"Unknown binned-fit report section: {section}")
