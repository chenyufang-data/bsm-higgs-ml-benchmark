"""Notebook/PDF views of the Phase 10a jet-network pilots; reads saved artifacts only."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hepml.adapters.evaluation_report import display_figure, read_evaluation_report

# Fixed colours per model, the same in every pilot report.
STYLE = {"bdt": ("C0", "BDT"), "lorentznet": ("C1", "LorentzNet"), "deep_sets": ("C2", "Deep Sets (control)"),
         "particle_transformer": ("C3", "ParT")}


def _present(frame, exclude=()):
    return [model for model in STYLE if model in set(frame.model) and model not in exclude]


def _against(frame):
    return frame.against if "against" in frame else "bdt"


def default_network_report():
    return Path(os.environ.get("HEPML_NETWORK_REPORT", "outputs/cg_bbc/reports/phase10a-lorentznet-v1"))


def roc_figure(directory):
    summary, tables = read_evaluation_report(directory)
    roc = tables["roc"]
    masses = summary["design"]["masses"]
    fig, axes = plt.subplots(1, len(masses), figsize=(6 * len(masses), 4.5), squeeze=False)
    for ax, mass in zip(axes[0], masses):
        for model in _present(roc):
            color, label = STYLE[model]
            rows = roc[(roc.mass == mass) & (roc.model == model)].groupby("signal_efficiency").background_efficiency
            mean, low, high = rows.mean(), rows.min(), rows.max()
            ax.plot(mean.index, mean.values, color=color, label=label, linewidth=2)
            ax.fill_between(mean.index, low.values, high.values, color=color, alpha=0.2)
        ax.set_yscale("log")
        ax.set(xlabel="Signal efficiency", ylabel="Background efficiency (physical weights)",
               title=f"m{mass}, rho_tc={summary['training_coupling']:g}, validation")
        ax.grid(alpha=0.2, which="both")
        ax.legend()
    fig.suptitle("Seed mean; band: seed range")
    fig.tight_layout()
    return fig


def delta_auc_figure(directory):
    summary, tables = read_evaluation_report(directory)
    metrics = tables["metrics"]
    masses = summary["design"]["masses"]
    fig, axes = plt.subplots(1, len(masses), figsize=(6 * len(masses), 4.2), squeeze=False)
    for ax, mass in zip(axes[0], masses):
        rows = metrics[metrics.mass == mass]
        bdt = rows[rows.model == "bdt"].set_index(["rho_tc", "seed"]).auc_weighted
        for model in _present(metrics, exclude=("bdt",)):
            color, label = STYLE[model]
            delta = (rows[rows.model == model].set_index(["rho_tc", "seed"]).auc_weighted - bdt).groupby("rho_tc")
            ax.errorbar(delta.mean().index, delta.mean().values,
                        yerr=[delta.mean() - delta.min(), delta.max() - delta.mean()], color=color, marker="o",
                        capsize=3, label=label)
        ax.axhline(0, color="grey", linewidth=1)
        ax.set(xlabel="rho_tc", ylabel="Weighted AUC - BDT AUC (same seed)", title=f"m{mass}, validation")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Seed mean; bars: seed range")
    fig.tight_layout()
    return fig


def learning_curves_figure(directory):
    summary, tables = read_evaluation_report(directory)
    curves = tables["learning_curves"]
    masses = summary["design"]["masses"]
    fig, axes = plt.subplots(1, len(masses), figsize=(6 * len(masses), 4.2), squeeze=False)
    for ax, mass in zip(axes[0], masses):
        for model in _present(curves):
            color, label = STYLE[model]
            for index, (_, rows) in enumerate(curves[(curves.mass == mass) & (curves.model == model)].groupby("seed")):
                ax.plot(rows.epoch, rows.val_fit_weighted_auc, color=color, alpha=0.8,
                        label=label if index == 0 else None)
        ax.set(xlabel="Epoch", ylabel="Fit-weighted validation AUC", title=f"m{mass}: checkpoint metric per seed")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    return fig


def bootstrap_figure(directory):
    summary, tables = read_evaluation_report(directory)
    replicas, primary = tables["bootstrap"], tables["primary"]
    masses = summary["design"]["masses"]
    fig, axes = plt.subplots(1, len(masses), figsize=(6 * len(masses), 4.2), squeeze=False)
    replicas, primary = replicas.assign(against=_against(replicas)), primary.assign(against=_against(primary))
    for ax, mass in zip(axes[0], masses):
        for row in primary[primary.mass == mass].itertuples():
            color = STYLE[row.model][0] if row.against == "bdt" else STYLE[row.against][0]
            label = f"{STYLE[row.model][1]} - {STYLE[row.against][1]}"
            values = replicas[(replicas.mass == mass) & (replicas.model == row.model)
                              & (replicas.against == row.against)].delta_auc_mean
            ax.hist(values, bins=40, color=color, alpha=0.5, label=label, hatch=None if row.against == "bdt" else "//")
            ax.axvline(row.delta_auc_mean, color=color, linewidth=2)
        ax.axvline(0, color="grey", linewidth=1)
        ax.set(xlabel="Seed-mean weighted AUC difference", ylabel="Bootstrap replicas",
               title=f"m{mass}, rho_tc={summary['training_coupling']:g}")
        ax.legend()
    fig.suptitle("Paired bootstrap of validation events; line: nominal")
    fig.tight_layout()
    return fig


FIGURES = {"roc": roc_figure, "delta_auc": delta_auc_figure, "learning_curves": learning_curves_figure,
           "bootstrap": bootstrap_figure}


def save_network_figures(directory):
    directory = Path(directory)
    for name, build in FIGURES.items():
        figure = build(directory)
        figure.savefig(directory / "figures" / f"{name}.pdf")
        plt.close(figure)


def display_network_section(directory, section):
    from IPython.display import Markdown, display

    summary, tables = read_evaluation_report(directory)
    design = summary["design"]

    def prose(text):
        display(Markdown(text))

    def table(frame):
        floats = frame.select_dtypes("float").columns
        display(frame.style.format({column: "{:.4g}" for column in floats}, na_rep="n/a"))

    if section == "scope":
        if summary["stage"] == "particle_transformer_pilot":
            prose(
                "**Phase 10a, Particle Transformer:** does the upstream ParT (weaver-core "
                f"{design['implementation']['version']}, published configuration) over the three role jets as tokens "
                "separate signal from background better than the mass-specific BDT, and than LorentzNet, on the same "
                f"information? Its LorentzNet reference is `{summary['lorentznet_report']}`. "
                f"**Test evaluated: {summary['test_evaluated']}.** All numbers use validation events."
            )
        else:
            prose(
                "**Phase 10a pilot:** does a Lorentz-equivariant network over the three role jets separate signal from "
                "background better than the mass-specific BDT, given the same information? The BDT's 15 features are "
                "functions of the b1, b2 and c1 four-vectors; the networks see exactly those, plus two beam spurions. "
                f"**Test evaluated: {summary['test_evaluated']}.** All numbers use validation events."
            )
        prose(
            f"Rows, splits and fit weights (the truth-tag pass probability) come from `{summary['grid_report']}`, "
            f"with the matching acceptance and heavy-flavour overlap removal. Masses {design['masses']}, seeds "
            f"{design['seeds']}, trained at rho_tc={summary['training_coupling']:g}. Every network uses its published "
            "configuration, not tuned, with the same training protocol and checkpoint rule."
        )
        prose(
            "Primary, pre-registered: the physically weighted validation AUC at the training coupling, paired with "
            f"the grid's BDT of the same seed, {design['primary']['bootstrap_replicates']} paired bootstrap replicas. "
            "An improvement needs a positive seed-mean difference, its "
            f"{design['primary']['interval']:.0%} interval above zero and a positive difference for every seed. "
            "Background efficiencies, the grid's operating point and stage C are descriptive: background MC "
            "statistics limit them."
        )
    elif section == "inputs":
        prose(f"Fits on {summary['device_name']} with torch {summary['torch']}:")
        table(tables["fits"][["mass", "model", "seed", "parameters", "epochs_run", "best_epoch",
                              "best_val_fit_weighted_auc", "seconds", "peak_gpu_mb"]])
        display_figure(learning_curves_figure(directory))
    elif section == "checks":
        table(tables["checks"])
    elif section == "results":
        prose("Primary comparison (weighted AUC difference to the BDT of the same seed):")
        table(tables["primary"])
        display_figure(bootstrap_figure(directory))
        display_figure(roc_figure(directory))
        display_figure(delta_auc_figure(directory))
        metrics = tables["metrics"]
        at_training = metrics[np.isclose(metrics.rho_tc, summary["training_coupling"])]
        prose("Descriptive metrics at the training coupling (MC-limited beyond the AUC):")
        table(at_training.groupby(["mass", "model"]).mean(numeric_only=True).drop(columns=["seed", "rho_tc"])
              .reset_index())
    elif section == "checkpoint":
        verdict = ", ".join(f"{mass}: {'improvement' if value else 'no improvement'}"
                            for mass, value in summary["improvement"].items())
        prose(f"**Status: {summary['status']}.** Pre-registered decision against the BDT: {verdict}. {summary['note']}")
        if summary.get("secondary"):
            versus = ", ".join(f"{mass}: {'improvement' if value else 'no improvement'}"
                               for mass, value in summary["secondary"].items())
            prose(f"Secondary, same rule, against LorentzNet of the same seed: {versus}.")
        prose("Next: review this pilot before any Particle Transformer study, all-jet inputs or constituents.")
    else:
        raise ValueError(f"Unknown network report section: {section}")
