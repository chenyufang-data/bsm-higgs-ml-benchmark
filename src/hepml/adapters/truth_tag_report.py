"""Notebook/PDF views of the truth-tagging closure; reads saved artifacts only."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hepml.adapters.evaluation_report import display_figure, read_evaluation_report

STATE_STYLE = {"b_only": ("C0", "b tag only"), "c_only": ("C3", "c tag only"), "both": ("C2", "both tags")}


def default_truth_tag_report():
    return Path(os.environ.get("HEPML_TRUTH_TAG_REPORT", "outputs/cg_bbc/reports/truth-tag-closure-v1"))


def tag_rates_figure(directory):
    summary, tables = read_evaluation_report(directory)
    rates = tables["tag_rates"]
    tests = tables["tag_rate_tests"].set_index("flavour")
    flavours = list(dict.fromkeys(rates.flavour))
    fig, axes = plt.subplots(1, len(flavours), figsize=(6 * len(flavours), 4.4))
    for ax, flavour in zip(axes, flavours):
        rows = rates[(rates.flavour == flavour) & (rates.jets > 0)]
        for state, (color, label) in STATE_STYLE.items():
            part = rows[rows.state == state].sort_values("pt_low")
            high = np.where(np.isfinite(part.pt_high), part.pt_high, 1.6 * part.pt_low)
            x = np.sqrt(part.pt_low * high)
            ax.plot(x, part.expected / part.jets, color=color, label=f"{label}: Delphes formula")
            observed = part.observed / part.jets
            ax.errorbar(x, observed, yerr=np.sqrt(np.maximum(part.observed, 1)) / part.jets, fmt="o", ms=4,
                        color=color, label=f"{label}: stored tag bits")
        test = tests.loc[flavour]
        ax.set(xscale="log", yscale="log", xlabel="jet pT [GeV]", ylabel="fraction of jets in acceptance",
               title=f"{flavour} jets: chi2/dof = {test.chi2:.1f}/{int(test.dof)}, p = {test.p_value:.3g}")
        ax.grid(alpha=0.2, which="both")
        ax.legend(fontsize=7)
    fig.suptitle("Gate 2: tag rates of non-test background jets against the pre-registered efficiencies")
    fig.tight_layout()
    return fig


def yields_figure(directory):
    summary, tables = read_evaluation_report(directory)
    yields = tables["yields"].sort_values(["gated", "sample"], ascending=[False, True]).reset_index(drop=True)
    limit = summary["settings"]["closure"]["yield_max_abs_pull"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    x = np.arange(len(yields))
    colors = np.where(yields.gated, "C3", "0.55")
    axes[0].scatter(x, yields.pull, c=colors, s=18)
    axes[0].axhspan(-limit, limit, color="C2", alpha=0.08)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(ylabel="(direct - truth-tag) / sigma", title=f"Gate 3: yield pulls (backgrounds gated at |pull| <= {limit:g})")
    axes[1].bar(x, yields.gain, color=colors)
    axes[1].set(yscale="log", ylabel="effective MC: truth-tag N_eff / expected direct count",
                title="Effective-MC gain of truth tagging (inclusive selection)")
    for ax in axes:
        ax.set_xticks(x, [s.replace("bkg_", "").replace("sig_", "") for s in yields["sample"]], rotation=90, fontsize=6)
        ax.grid(alpha=0.2, axis="y", which="both")
    fig.suptitle("Red: backgrounds (gated); grey: signals (descriptive). Non-test events only.")
    fig.tight_layout()
    return fig


def shapes_figure(directory):
    summary, tables = read_evaluation_report(directory)
    primary = summary["settings"]["closure"]["shapes"]["primary_sample"]
    bins = tables["shape_bins"][tables["shape_bins"]["sample"] == primary]
    tests = tables["shapes"][tables["shapes"]["sample"] == primary].set_index("feature")
    features = list(dict.fromkeys(bins.feature))
    columns = 5
    fig, axes = plt.subplots(int(np.ceil(len(features) / columns)), columns, figsize=(18, 9), sharey=True)
    for ax, feature in zip(axes.ravel(), features):
        part = bins[bins.feature == feature].sort_values("bin")
        ax.bar(part["bin"], part.pull, color="C0")
        for level in (-3, 3):
            ax.axhline(level, color="grey", linewidth=0.8, linestyle=":")
        test = tests.loc[feature]
        ax.set_title(f"{feature}: p = {test.p_value:.3g}", fontsize=9)
        ax.set_xticks(part["bin"])
        ax.grid(alpha=0.2, axis="y")
    for ax in axes.ravel()[len(features):]:
        ax.set_visible(False)
    fig.supxlabel("decile bin of the direct-tag distribution")
    fig.supylabel("(direct - truth-tag) / sigma")
    fig.suptitle(f"Gate 4: feature shapes on {primary}, direct against truth-tag entries")
    fig.tight_layout()
    return fig


FIGURES = {"tag_rates": tag_rates_figure, "yields": yields_figure, "shapes": shapes_figure}


def save_truth_tag_figures(directory):
    directory = Path(directory)
    for name, build in FIGURES.items():
        figure = build(directory)
        figure.savefig(directory / "figures" / f"{name}.pdf")
        plt.close(figure)


def display_truth_tag_section(directory, section):
    from IPython.display import Markdown, display

    summary, tables = read_evaluation_report(directory)
    settings = summary["settings"]
    closure = settings["closure"]

    def prose(text):
        display(Markdown(text))

    def table(frame):
        floats = frame.select_dtypes("float").columns
        display(frame.style.format({column: "{:.4g}" for column in floats}, na_rep="n/a"))

    if section == "scope":
        prose(
            "**Truth-tagging closure:** does weighting every passing tag outcome by its Delphes probability reproduce "
            "direct tagging? Delphes tags each jet by independent b and c draws whose efficiencies depend only on the "
            "jet's flavour label and pT, so the closure is exact in expectation if the pre-registered efficiencies are "
            "the ones Delphes used. **Test evaluated: False**; test-split events are excluded and no model score is "
            "computed."
        )
        prose(
            "Gates, fixed in `validation.yaml: truth_tagging` before any truth-tag export existed: "
            "(1) the direct selection re-applied to the stored tag bits reproduces the direct export exactly; "
            f"(2) tag rates per flavour and pT bin match the formulas (p >= {closure['tag_rates']['min_p_value']} per "
            f"flavour class); (3) yields agree for every background (|pull| <= {closure['yield_max_abs_pull']:g}); "
            f"(4) the BDT features on {closure['shapes']['primary_sample']} agree in shape "
            f"(p >= {closure['shapes']['min_p_value']} per feature)."
        )
    elif section == "inputs":
        prose("Gate 1, per sample (non-test events of the direct export, and of the truth-tag export after the "
              "direct selection):")
        table(tables["consistency"])
        prose("Event registry extended to the truth-tag events; frozen assignments never move:")
        table(tables["registry_extension"])
    elif section == "checks":
        table(tables["checks"])
        table(tables["gates"])
    elif section == "results":
        display_figure(tag_rates_figure(directory))
        table(tables["tag_rate_tests"])
        display_figure(yields_figure(directory))
        table(tables["yields"])
        display_figure(shapes_figure(directory))
        shapes = tables["shapes"]
        table(shapes[shapes.gated])
        prose("Other backgrounds, descriptive (small direct samples), p-values per feature:")
        table(shapes[~shapes.gated].pivot(index="feature", columns="sample", values="p_value").reset_index())
    elif section == "checkpoint":
        gains = ", ".join(f"{name.replace('bkg_', '')} {value:.1f}x" for name, value in summary["gains"].items())
        prose(f"Measured effective-MC gain over the expected direct count (inclusive selection): {gains}.")
        prose(f"**Truth tagging adopted: {summary['adopted']}.** {summary['note']} "
              f"Extended registry: `{summary['registry']}`.")
    else:
        raise ValueError(f"Unknown truth-tagging report section: {section}")
