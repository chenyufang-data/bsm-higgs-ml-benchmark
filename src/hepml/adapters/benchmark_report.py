"""Figures and a markdown summary of the method benchmark; reads the benchmark report's tables only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed colour and label per method, in presentation order; the baseline is neutral grey.
METHODS = {"cut_based": ("#7f7f7f", "Cut-based"), "bdt": ("C0", "BDT (XGBoost)"),
           "lorentznet": ("C1", "LorentzNet"), "particle_transformer": ("C3", "Particle Transformer")}
MODELS = ["bdt", "lorentznet", "particle_transformer"]
INPUT_LABELS = {"pt": "pT", "eta": "η", "phi": "φ", "mass": "mass"}
JETS = {"b1": "b1", "b2": "b2", "c1": "c"}
# Readable names of the BDT's engineered features (all from the three tagged jets).
FEATURE_LABELS = {"ptb1": "b1 pT", "etab1": "b1 η", "ptb2": "b2 pT", "etab2": "b2 η", "ptc1": "c pT", "etac1": "c η",
                  "mcb1": "m(c, b1)", "mcb2": "m(c, b2)", "mbb": "m(b1, b2)", "mcbb": "m(c, b1, b2)",
                  "dr13": "ΔR(b1, c)", "dr23": "ΔR(b2, c)", "dr12": "ΔR(b1, b2)", "ratio_ptcb": "pT(c) / pT(b1)",
                  "ht": "Σ pT of the 3 jets"}


def _tables(directory):
    directory = Path(directory)
    summary = json.loads((directory / "provenance/summary.json").read_text())
    return summary, {path.stem: pd.read_csv(path) for path in (directory / "tables").glob("*.csv")}


def _input_label(name):
    for quantity, label in INPUT_LABELS.items():
        if name.startswith(quantity):
            return f"{JETS.get(name[len(quantity):], name[len(quantity):])} {label}"
    return name


def gain_figure(directory):
    summary, tables = _tables(directory)
    methods = tables["methods"]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.2
    for index, (method, (color, label)) in enumerate(METHODS.items()):
        rows = methods[methods.method == method].groupby("mass").z_asimov_ratio
        mean, low, high = rows.mean(), rows.min(), rows.max()
        x = np.arange(len(mean)) + (index - 1.5) * width
        ax.bar(x, mean.values, width * 0.9, color=color, label=label)
        if method != "cut_based":
            ax.errorbar(x, mean.values, yerr=[mean - low, high - mean], fmt="none", color="black", linewidth=1, capsize=2)
    ax.axhline(1, color="#7f7f7f", linewidth=0.8)
    ax.set_xticks(np.arange(len(summary["masses"])), [f"m = {m} GeV" for m in summary["masses"]])
    ax.set_ylabel("Expected significance / cut-based")
    ax.set_title("Expected significance at each method's validation optimum")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def roc_figure(directory):
    summary, tables = _tables(directory)
    roc, methods = tables["roc"], tables["methods"]
    fig, axes = plt.subplots(1, len(summary["masses"]), figsize=(6 * len(summary["masses"]), 4.2), squeeze=False)
    for ax, mass in zip(axes[0], summary["masses"]):
        for method in MODELS:
            color, label = METHODS[method]
            rows = roc[(roc.mass == mass) & (roc.method == method)].groupby("signal_efficiency").background_efficiency
            ax.plot(rows.mean().index, rows.mean().values, color=color, linewidth=2, label=label)
            ax.fill_between(rows.mean().index, rows.min().values, rows.max().values, color=color, alpha=0.2)
        cut = methods[(methods.mass == mass) & (methods.method == "cut_based")].iloc[0]
        ax.plot(cut.signal_efficiency, cut.background_efficiency, marker="*", markersize=12, color="#7f7f7f",
                linestyle="none", label="Cut-based working point")
        ax.set_yscale("log")
        ax.set(xlabel="Signal efficiency", ylabel="Background efficiency (lower is better)", title=f"m = {mass} GeV")
        ax.grid(alpha=0.2, which="both")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def permutation_figure(directory):
    summary, tables = _tables(directory)
    table = tables["permutation"]
    fig, axes = plt.subplots(1, len(summary["masses"]), figsize=(6 * len(summary["masses"]), 5), squeeze=False)
    for ax, mass in zip(axes[0], summary["masses"]):
        rows = table[table.mass == mass]
        order = rows.groupby("input").auc_drop.mean().sort_values().index
        y = np.arange(len(order))
        for index, method in enumerate(MODELS):
            color, label = METHODS[method]
            values = rows[rows.method == method].set_index("input").reindex(order)
            ax.barh(y + (index - 1) * 0.27, values.auc_drop, 0.25, xerr=values.auc_drop_spread, color=color, label=label,
                    error_kw=dict(linewidth=0.8))
        ax.set_yticks(y, [_input_label(name) for name in order])
        ax.set(xlabel="AUC drop when the input is shuffled", title=f"m = {mass} GeV")
        ax.grid(axis="x", alpha=0.2)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def shap_figure(directory):
    summary, tables = _tables(directory)
    table = tables["shap"]
    fig, axes = plt.subplots(1, len(summary["masses"]), figsize=(5 * len(summary["masses"]), 4.5), squeeze=False)
    for ax, mass in zip(axes[0], summary["masses"]):
        rows = table[table.mass == mass].sort_values("mean_abs_shap")
        ax.barh([FEATURE_LABELS.get(f, f) for f in rows.feature], rows.mean_abs_shap, color=METHODS["bdt"][0])
        ax.set(xlabel="Mean |SHAP value|", title=f"BDT features, m = {mass} GeV")
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def tradeoff_figure(directory):
    summary, tables = _tables(directory)
    costs, methods = tables["costs"], tables["methods"]
    fig, axes = plt.subplots(1, len(summary["masses"]), figsize=(5.5 * len(summary["masses"]), 4), squeeze=False)
    fit = costs.dropna(subset=["fit_seconds"]).set_index("method")
    for ax, mass in zip(axes[0], summary["masses"]):
        for method in MODELS:
            color, label = METHODS[method]
            auc = methods[(methods.mass == mass) & (methods.method == method)].auc
            size = fit.loc[method, "parameter_note"] if method == "bdt" else f"{int(fit.loc[method, 'parameters']):,} parameters"
            ax.errorbar(fit.loc[method, "fit_seconds"], auc.mean(), yerr=[[auc.mean() - auc.min()], [auc.max() - auc.mean()]],
                        marker="o", markersize=8, color=color, capsize=3, label=f"{label} ({size})")
        ax.set_xscale("log")
        ax.set(xlabel="Training time per fit [s]", ylabel="Weighted validation AUC", title=f"m = {mass} GeV")
        ax.grid(alpha=0.2, which="both")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


FIGURES = {"significance_gain": gain_figure, "roc": roc_figure, "permutation_importance": permutation_figure,
           "shap_bdt": shap_figure, "cost_tradeoff": tradeoff_figure}


def save_benchmark_figures(directory):
    directory = Path(directory)
    for name, build in FIGURES.items():
        figure = build(directory)
        figure.savefig(directory / "figures" / f"{name}.pdf")
        figure.savefig(directory / "figures" / f"{name}.png", dpi=150)
        plt.close(figure)


def write_benchmark_markdown(directory):
    """A compact results table for the public README; no absolute significance or yield."""
    directory = Path(directory)
    summary, tables = _tables(directory)
    methods, paired, costs = tables["methods"], tables["paired_auc"], tables["costs"]
    lines = ["| Method | " + " | ".join(f"m = {m} GeV: significance vs cut-based | AUC" for m in summary["masses"]) + " |",
             "| --- |" + " --- | --- |" * len(summary["masses"])]
    for method, (_, label) in METHODS.items():
        cells = []
        for mass in summary["masses"]:
            rows = methods[(methods.mass == mass) & (methods.method == method)]
            ratio = rows.z_asimov_ratio
            cells.append("1.00 (reference)" if method == "cut_based" else
                         f"{ratio.mean():.2f} ({ratio.min():.2f}-{ratio.max():.2f})")
            cells.append("-" if method == "cut_based" else f"{rows.auc.mean():.4f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "| Paired AUC difference (3 seeds, 95% bootstrap interval) | "
              + " | ".join(f"m = {m} GeV" for m in summary["masses"]) + " |",
              "| --- |" + " --- |" * len(summary["masses"])]
    against = {"bdt": "BDT", "lorentznet": "LorentzNet"}
    # The LorentzNet pilot's table predates the "against" column: all its rows are against the BDT.
    paired = paired.assign(against=paired["against"].fillna("bdt") if "against" in paired else "bdt")
    for (model, reference), rows in paired.groupby(["model", "against"], sort=False):
        if model == "deep_sets":
            continue
        cells = [f"{r.delta_auc_mean:+.4f} [{r.interval_low:+.4f}, {r.interval_high:+.4f}]"
                 for r in rows.sort_values("mass").itertuples()]
        lines.append(f"| {METHODS[model][1]} vs {against[reference]} | " + " | ".join(cells) + " |")
    fit = costs.dropna(subset=["fit_seconds"]).set_index("method")
    inference = costs.dropna(subset=["inference_ms_per_1k_events"]).groupby("method").agg(
        ms=("inference_ms_per_1k_events", "mean"), device=("device", "first"))
    lines += ["", "| Model | Size | Training time per fit | Inference per 1k events |", "| --- | --- | --- | --- |"]
    for method in MODELS:
        size = fit.loc[method, "parameter_note"] if method == "bdt" else f"{int(fit.loc[method, 'parameters']):,} parameters"
        lines.append(f"| {METHODS[method][1]} | {size} | {fit.loc[method, 'fit_seconds']:.0f} s ({fit.loc[method, 'fit_device']}) "
                     f"| {inference.loc[method, 'ms']:.1f} ms ({inference.loc[method, 'device']}) |")
    (directory / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
