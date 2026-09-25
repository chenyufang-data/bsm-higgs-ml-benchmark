"""Notebook/PDF views of the Phase 5 paired comparison; reads saved artifacts only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hepml.adapters.evaluation_report import display_figure, read_evaluation_report
from hepml.domain.metrics import OBJECTIVES

LABELS = {"reference": "mass-specific f_m(x)", "candidate": "parameterized f(x, m)",
          "control": "loss-weight control", "cut_reference": "cut reference"}
COLORS = {"reference": "C0", "candidate": "C1", "control": "C2", "cut_reference": "C3"}
SEED_STYLES = ["-", "--", ":", "-."]
SEED_COLORS = ["C4", "C5", "C6", "C7"]


def _efficiency_columns(frame):
    columns = [c for c in frame.columns if c.startswith("signal_efficiency_at_")]
    return sorted(columns, key=lambda c: float(c.rsplit("_", 1)[1]))


def _seeded(summary, tables):
    """Tables with a seed column; the first single-seed report stored none."""
    seed = summary["settings"]["seeds"][0]
    return {name: frame if "seed" in frame or name not in {"paired_metrics", "bootstrap_replicas", "noninferiority",
                                                            "working_points"} else frame.assign(seed=seed)
            for name, frame in tables.items()}


def model_directories(summary):
    """Saved validation model directory of every compared model, by (mass, seed)."""
    result = {}
    for item in summary["compared"]:
        key = (item["mass"], item.get("seed", summary["settings"]["seeds"][0]))
        result[key] = {"reference": item["reference"], "candidate": item["candidate"]}
        if item.get("control"):
            result[key]["control"] = item["control"]
    return result


def paired_roc_figure(directory):
    summary, tables = read_evaluation_report(directory)
    settings = summary["settings"]
    seeds, masses = settings["seeds"], settings["masses"]
    points = [settings["background_efficiency"], *settings["secondary_background_efficiencies"]]
    cuts = tables["own_operating_points"]
    cuts = cuts[(cuts.model == "cut_reference") & (cuts.status == "valid")]
    directories = model_directories(summary)
    fig, axes = plt.subplots(2, len(masses), figsize=(6 * len(masses), 9), squeeze=False)
    for column, mass in enumerate(masses):
        for row, limits in enumerate([(0, 1), (0.1, 0.4)]):
            ax, visible = axes[row, column], []
            for index, seed in enumerate(seeds):
                for model, path in directories[mass, seed].items():
                    roc = pd.read_parquet(Path(path) / "roc.parquet").query("split == 'val'")
                    visible.append(roc)
                    ax.plot(roc.background_efficiency, roc.signal_efficiency, color=COLORS[model],
                            linestyle=SEED_STYLES[index % len(SEED_STYLES)], linewidth=1.2,
                            label=f"{LABELS[model]}, seed {seed}")
            selected = cuts[cuts.mass == mass]
            ax.scatter(selected.background_efficiency, selected.signal_efficiency, color=COLORS["cut_reference"],
                       marker="s", s=36, zorder=3, label=LABELS["cut_reference"] + " (validation optima)")
            for value in points:
                ax.axvline(value, color="gray", linestyle="-" if value == points[0] else ":", linewidth=1)
            ax.set_xlim(*limits)
            if row == 1:
                inside = pd.concat(visible).query(f"{limits[0]} <= background_efficiency <= {limits[1]}")
                if len(inside):
                    ax.set_ylim(inside.signal_efficiency.min() - 0.02, min(1.0, inside.signal_efficiency.max() + 0.02))
            ax.set(xlabel="Weighted background efficiency", ylabel="Weighted signal efficiency",
                   title=f"m{mass} validation ROC" + (" (working-point region)" if row else ""))
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Same validation events and weights for every model; solid vertical line: pre-registered working point")
    fig.tight_layout()
    return fig


def efficiency_ratio_figure(directory):
    summary, tables = read_evaluation_report(directory)
    tables = _seeded(summary, tables)
    settings = summary["settings"]
    margin = settings["noninferiority"]["max_relative_signal_efficiency_loss"]
    confidence = settings["noninferiority"]["one_sided_confidence"]
    estimates, replicas = tables["paired_metrics"], tables["bootstrap_replicas"]
    columns = _efficiency_columns(estimates)
    grid = np.array([float(c.rsplit("_", 1)[1]) for c in columns])
    masses = settings["masses"]
    fig, axes = plt.subplots(1, len(masses), figsize=(6 * len(masses), 4.5), squeeze=False)
    for ax, mass in zip(axes[0], masses):
        for index, seed in enumerate(settings["seeds"]):
            point = estimates[(estimates.mass == mass) & (estimates.seed == seed)].set_index("model")
            replica = replicas[(replicas.mass == mass) & (replicas.seed == seed)]
            for model, baseline in [("candidate", "reference"), ("control", "reference")]:
                if model not in point.index:
                    continue
                ratio = point.loc[model, columns].to_numpy(float) / point.loc[baseline, columns].to_numpy(float)
                wide = {name: replica[replica.model == name].sort_values("replicate")[columns].to_numpy(float)
                        for name in (model, baseline)}
                low, high = np.quantile(wide[model] / wide[baseline], [1 - confidence, confidence], axis=0)
                ax.plot(grid, ratio, color=COLORS[model], linestyle=SEED_STYLES[index % len(SEED_STYLES)],
                        marker="o", markersize=2.5, label=f"{LABELS[model]} / {LABELS[baseline]}, seed {seed}")
                ax.fill_between(grid, low, high, color=COLORS[model], alpha=0.1)
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.axhline(1 - margin, color="red", linestyle="--", linewidth=1, label=f"non-inferiority margin ({margin:.0%} loss)")
        ax.axvline(settings["background_efficiency"], color="gray", linewidth=1)
        for value in settings["secondary_background_efficiencies"]:
            ax.axvline(value, color="gray", linestyle=":", linewidth=1)
        ax.set(xlabel="Weighted background efficiency", ylabel="Signal-efficiency ratio",
               title=f"m{mass}: paired ratio, bands {1 - confidence:.0%}-{confidence:.0%} bootstrap quantiles")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def paired_difference_figure(directory):
    summary, tables = read_evaluation_report(directory)
    tables = _seeded(summary, tables)
    tests = tables["noninferiority"]
    gate = tests[(tests.purpose == "gate") & (tests.role == "primary")]
    replicas = tables["bootstrap_replicas"]
    masses, seeds = summary["settings"]["masses"], summary["settings"]["seeds"]
    fig, axes = plt.subplots(2, len(masses), figsize=(6 * len(masses), 7), squeeze=False)
    for column, mass in enumerate(masses):
        for row, metric in enumerate(sorted(gate.metric.unique(), key=lambda m: m == "auc_weighted")):
            ax = axes[row, column]
            for index, seed in enumerate(seeds):
                test = gate[(gate.mass == mass) & (gate.seed == seed) & (gate.metric == metric)].iloc[0]
                wide = replicas[(replicas.mass == mass) & (replicas.seed == seed)].pivot(
                    index="replicate", columns="model", values=metric)
                losses = 1 - wide.candidate / wide.reference if test.relative else wide.reference - wide.candidate
                color = SEED_COLORS[index % len(SEED_COLORS)]
                ax.hist(losses, bins=40, histtype="step", color=color, linewidth=1.3,
                        label=f"seed {seed}: {test.loss:+.4f}, upper {test.upper:+.4f} ({test.outcome})")
                ax.axvline(test.upper, color=color, linestyle=":")
            ax.axvline(test.margin, color="red", linestyle="--", label=f"margin {test.margin:g}")
            unit = "relative signal-efficiency loss" if test.relative else "weighted AUC loss"
            outcome = summary["gate"].get(str(mass), "")
            ax.set(xlabel=f"{unit} (positive: parameterized worse)", ylabel="Bootstrap replicas",
                   title=f"m{mass}: {metric} (mass outcome: {outcome})")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


FIGURES = {"paired_roc": paired_roc_figure, "efficiency_ratio": efficiency_ratio_figure,
           "paired_differences": paired_difference_figure}


def save_comparison_figures(directory):
    directory = Path(directory)
    for name, build in FIGURES.items():
        figure = build(directory)
        figure.savefig(directory / "figures" / f"{name}.pdf")
        plt.close(figure)


def display_comparison_section(directory, section):
    from IPython.display import Markdown, display

    summary, tables = read_evaluation_report(directory)
    tables = _seeded(summary, tables)
    settings, cfg = summary["settings"], summary["evaluation_config"]
    margins = settings["noninferiority"]
    seeds = settings["seeds"]

    def prose(text):
        display(Markdown(text))

    def table(frame):
        display(frame.style.format(precision=5, na_rep="unavailable"))

    if section == "scope":
        prose(
            "**Phase 5:** is one shared model f(x, m) acceptable at each trained mass, compared with a dedicated "
            f"model per mass? Masses {settings['masses']} GeV, rho_tc={settings['rho_tc']:g}, seeds {seeds}. "
            f"**Test evaluated: {summary['test_evaluated']}.** All numbers use validation events only."
        )
        prose(
            f"**Pre-registered gate** (recorded before any paired number): at each mass the parameterized model is "
            f"non-inferior if, with a one-sided {margins['one_sided_confidence']:.0%} paired-bootstrap bound, (a) its "
            f"weighted signal efficiency at a weighted background efficiency of {settings['background_efficiency']:g} "
            f"is at most {margins['max_relative_signal_efficiency_loss']:.0%} lower, and (b) its weighted AUC is at most "
            f"{margins['max_weighted_auc_loss']:g} lower than the mass-specific reference. Background efficiencies "
            f"{settings['secondary_background_efficiencies']} are secondary. {settings['bootstrap_replicates']} replicas, "
            f"seed {settings['bootstrap_seed']}."
        )
        if len(seeds) > 1:
            prose(
                "**Seed rule** (recorded before any added-seed number): each seed pairs the mass-specific and "
                "parameterized fits trained with that seed. A mass is non-inferior only if every seed is, inferior if "
                "any seed is, and inconclusive otherwise."
            )
        prose(
            f"Weights: {cfg['lumi_pb_inv'] / 1000:g} fb^-1, signal k={cfg['signal_k_factor']:g}, background "
            f"k={cfg['background_k_factor']:g}, each process restored to its full prepared yield, exactly as in Phases 2/3. "
            "At a fixed background efficiency the background yield is common to all models, so a relative loss in "
            "signal efficiency is the same relative loss in significance."
        )
        table(pd.DataFrame(summary["compared"]))
    elif section == "weights":
        prose(
            "The mass-specific reference is fit unweighted, so its class-loss ratio is N_signal(m)/N_background. The "
            "parameterized model anchors both classes to the reference-mass counts at every mass (Phase 3). The "
            "**loss-weight control** is a mass-specific fit with the parameterized model's ratio: signal weight "
            "N_signal(reference mass)/N_signal(m), background weight 1, per split and per seed. Its weights are "
            "checked against the parameterized model's saved weights divided by the hypothesis prior."
        )
        prose(
            "Contrasts: reference vs control isolates loss weighting; control vs parameterized isolates sharing one "
            "model across masses at matched loss weights. At the anchor mass the control equals the reference by "
            "construction and is not refit. The parameterized model also learns from the other mass's signal events "
            "(`unique_training_events`); that shared learning is intended and is disclosed rather than removed."
        )
        table(tables["loss_weights"])
    elif section == "checks":
        table(tables["checks"])
        if len(tables["resources"]):
            resources = tables["resources"].copy()
            resources["peak_process_RSS_MiB"] = resources.peak_sampled_rss_bytes / 1024**2
            table(resources[["stage", "trees", "seconds", "peak_process_RSS_MiB"]])
        prose(
            "Completed Phase 2/3 artifacts are re-hashed before use. Models are matched by `event_id`, never by row "
            "position, and must agree on process, class and stored weight. No test score is read or written."
        )
    elif section == "results":
        columns = ["mass", "seed", "model", "run_id", "auc_weighted", "auc_unweighted",
                   *[f"signal_efficiency_at_{x:g}" for x in sorted([settings["background_efficiency"],
                                                                     *settings["secondary_background_efficiencies"]])]]
        table(tables["paired_metrics"][columns].sort_values(["mass", "seed", "model"]))
        display_figure(paired_roc_figure(directory))
        display_figure(efficiency_ratio_figure(directory))
        prose(
            "Signal efficiency is interpolated along each model's validation ROC path. Ratio bands are paired bootstrap "
            "quantiles: every replica resamples validation events within each process once and applies the same "
            "multiplicities to every model, every seed and the common background at both masses. Bands describe finite "
            "validation MC; differences between seeds show training variation."
        )
    elif section == "thresholds":
        prose(
            "**MC support at the working points.** The threshold is the loosest one whose background efficiency does "
            f"not exceed the target; background N_eff there must stay well above the {cfg['min_background_neff']:g} screen "
            "used for the significance optima. S and B are 500 fb^-1 expected yields at the interpolated efficiencies."
        )
        table(tables["working_points"].sort_values(["mass", "seed", "model", "background_efficiency"]))
        prose(
            "**Each model at its own validation optima** (descriptive, not part of the gate). These optima sit at the "
            "background N_eff screen, which is why the gate uses a fixed background efficiency instead. The cut "
            "reference is the Phase 2 validation selection."
        )
        own = tables["own_operating_points"]
        columns = [c for c in ["mass", "seed", "model", "objective", "status", "threshold", "S", "B", "S_over_B",
                               *OBJECTIVES, "background_neff", "signal_efficiency", "background_efficiency"] if c in own]
        table(own[columns].sort_values([c for c in ["mass", "objective", "model", "seed"] if c in own]))
        prose("**Every trained mass-specific seed** (descriptive; includes any seed outside the comparison).")
        table(tables["reference_seed_spread"])
    elif section == "checkpoint":
        tests = tables["noninferiority"]
        prose(f"**Gate outcome: {summary['outcome']}** (per mass: {summary['gate']}).")
        table(tables["gate"])
        table(tests[tests.purpose == "gate"][["mass", "seed", "metric", "role", "baseline_value", "compared_value",
                                               "loss", "lower", "upper", "margin", "outcome"]])
        display_figure(paired_difference_figure(directory))
        descriptive = tests[tests.purpose != "gate"]
        if len(descriptive):
            prose("Control contrasts (descriptive, same bootstrap):")
            table(descriptive[["mass", "seed", "baseline", "compared", "purpose", "metric", "loss", "lower", "upper"]])
        outcome = summary["outcome"]
        if outcome == "non-inferior":
            prose(
                "The shared model meets both pre-registered criteria at every compared mass"
                + (" for every seed" if len(seeds) > 1 else "") + " on validation. Next: "
                + ("Phase 6 leave-one-mass-out interpolation." if len(seeds) > 1 else
                   "add the remaining seeds for both model families, then Phase 6 leave-one-mass-out interpolation.")
                + " The test partition stays sealed until those choices are frozen."
            )
        elif outcome == "inferior":
            prose(
                "At least one criterion shows a loss beyond its margin. Following the plan, diagnose capacity and "
                "weighting first (the control contrasts separate loss weighting from parameterization) and keep local "
                "models where the failure is reproducible. The metric, working point and margins stay fixed."
            )
        else:
            prose(
                "Neither non-inferiority nor inferiority is established at the pre-registered margins. The metric, "
                "working point and margins stay fixed; the control contrasts and seeds indicate whether the "
                "uncertainty comes from loss weighting, sharing, training variation or finite validation MC."
            )
        prose(f"**Status: {summary['status']}.** {summary['note']}")
    else:
        raise ValueError(f"Unknown comparison report section: {section}")
