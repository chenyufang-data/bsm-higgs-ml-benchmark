"""Thin notebook/PDF views for the conditional classifier and its leakage checks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from hepml.adapters.evaluation_report import (
    display_figure,
    display_report_figures,
    read_evaluation_report,
    report_curve_figure,
    stability_figure,
)
from hepml.domain.metrics import OBJECTIVES


def default_parameterized_report():
    return Path(os.environ.get("HEPML_PARAMETERIZED_REPORT", "outputs/cg_bbc/reports/phase3-rho01-v1"))


def conditional_diagnostics(directory, record):
    summary, tables = read_evaluation_report(directory)
    model = Path(summary["shared_model_directory"])
    history = json.loads((model / "learning_curves.json").read_text())
    metrics = json.loads((model / "metrics.json").read_text())
    best = metrics["xgb"]["best_iteration"] + 1
    roc = pd.read_parquet(Path(record["model_directory"]) / "roc.parquet")
    learning = tables["per_mass_learning"]
    learning = learning[learning.mass == record["mass"]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for name, group in roc.groupby("split"):
        axes[0, 0].plot(group.background_efficiency, group.signal_efficiency, label=name)
    axes[0, 0].plot([0, 1], [0, 1], color="gray", linestyle=":")
    axes[0, 0].set(xlabel="Background efficiency", ylabel="Signal efficiency", title="Physical-mixture ROC")
    for name, group in learning.groupby("split"):
        axes[0, 1].plot(group.trees, group.auc_weighted, label=name)
        axes[1, 0].plot(group.trees, group.fit_logloss, label=name)
    axes[0, 1].set(xlabel="Trees", ylabel="Weighted AUC", title="Per-mass physical-mixture AUC")
    axes[1, 0].set(xlabel="Trees", ylabel="Fit-weighted log loss", title="Per-mass validation/training loss")
    for key, name in [("validation_0", "train"), ("validation_1", "validation")]:
        values = history[key]["auc"]
        axes[1, 1].plot(range(1, len(values) + 1), values, label=name)
    axes[1, 1].set(xlabel="Trees", ylabel="Pooled fit-weighted AUC", title="Actual early-stopping objective")
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.axvline(best, color="gray", linestyle=":", label="Chosen shared checkpoint")
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(f"m{record['mass']}: one shared model, seed {record['seed']}")
    fig.tight_layout()
    return fig


def save_parameterized_figures(directory):
    from matplotlib.backends.backend_pdf import PdfPages

    directory = Path(directory)
    summary, _ = read_evaluation_report(directory)
    with (
        PdfPages(directory / "figures/learning_roc.pdf") as diagnostic,
        PdfPages(directory / "figures/threshold_curves.pdf") as curves,
        PdfPages(directory / "figures/threshold_stability.pdf") as stability,
    ):
        for record in summary["runs"]:
            figures = [
                (diagnostic, conditional_diagnostics(directory, record)),
                (curves, report_curve_figure(directory, record["run_id"])),
                (
                    stability,
                    stability_figure(
                        pd.read_csv(Path(record["model_directory"]) / "threshold_stability.csv"), title=record["run_id"]
                    ),
                ),
            ]
            for pdf, figure in figures:
                pdf.savefig(figure)
                plt.close(figure)


def display_parameterized_section(directory, section):
    from IPython.display import Markdown, display

    summary, tables = read_evaluation_report(directory)
    if summary["phases"] == [5]:
        from hepml.adapters.comparison_report import display_comparison_section

        return display_comparison_section(directory, section)
    settings, cfg = summary["settings"], summary["evaluation_config"]

    def prose(text):
        display(Markdown(text))

    def table(name, columns=None):
        frame = tables[name]
        display((frame if columns is None else frame[columns]).style.format(precision=6, na_rep="unavailable"))

    if section == "scope":
        prose(
            f"**Phase 3:** one classifier f(x, m) trained at {settings['masses']} GeV and fixed "
            f"rho_tc={settings['rho_tc']:g}, seed {settings['seed']}. **Test evaluated: {summary['test_evaluated']}.** "
            "This is a conditioning implementation checkpoint; paired superiority and interpolation are later studies."
        )
        prose(
            f"Evaluation: {cfg['lumi_pb_inv'] / 1000:g} fb^-1; signal k={cfg['signal_k_factor']:g}, "
            f"background k={cfg['background_k_factor']:g}, each applied once. Primary objective: {cfg['primary_objective']}."
        )
        display(settings)
        prose(
            "The 15 physical inputs remain fixed; `hypothesis_mass` is an additional continuous numerical input in GeV. "
            "Signal events receive their generated mass. Each background appears at both hypotheses, always inside its "
            "original split. At inference every event is scored at one explicitly requested mass. "
            "The same model file serves both hypotheses; evaluation records do not duplicate the model."
        )
    elif section == "weights":
        prose(
            f"**Fitting convention:** retain the unweighted within-class process mixture from Phase 2. "
            f"For each split, total signal and background fitting weights equal the m{settings['reference_mass']} "
            "reference counts. Both classes assign half of their loss weight to each mass. "
            "Each background copy therefore has weight 1/2; signal events at mass h have weight "
            "`N_signal(reference) / (2 * N_signal(h))`. These are temporary loss weights, never physical yields."
        )
        prose(
            "This necessary hypothesis balancing changes the pooled loss measure. The m200 class totals are anchored; "
            "the m400 standalone class ratio is not exactly identical. Phase 5 must disclose/control this distinction "
            "before attributing differences solely to parameterization. Physical background-mixture fitting was not adopted."
        )
        table("populations")
        table("hypothesis_priors")
        prose(
            "Stored physical-weight sums are shown per hypothesis for auditing; adding background replicas or "
            "alternative signals would double-count physical rates. Evaluation uses unique events and unchanged saved weights."
        )
    elif section == "checks":
        table("checks")
        table("mass_only_probe")
        prose(
            "The mass-only classifier uses the same fitting weights. Its weighted validation AUC must be 0.5: "
            "identical class-conditional mass priors make conditioning alone non-discriminating. "
            "This exact two-point prior check is not evidence of interpolation or physical rho independence."
        )
        resources = tables["resources"].copy()
        resources["peak_process_RSS_MiB"] = resources.peak_sampled_rss_bytes / 1024**2
        resources["RSS_increase_MiB"] = resources.peak_rss_delta_bytes / 1024**2
        display(
            resources[["stage", "trees", "seconds", "peak_process_RSS_MiB", "RSS_increase_MiB"]].style.format(
                precision=3
            )
        )
        prose(
            "Resource usage is sampled whole-process resident memory every 20 ms, including loaded data and native "
            "allocations. The bounded resource pilot is discarded; only the full-budget fit is evaluated for physics."
        )
    elif section == "results":
        table("seed_comparison")
        table(
            "operating_points",
            [
                "mass",
                "objective",
                "status",
                "threshold",
                "plateau_threshold",
                "S",
                "B",
                "S_over_B",
                *OBJECTIVES,
                "background_neff",
            ],
        )
        for record in summary["runs"]:
            display_figure(conditional_diagnostics(directory, record))
        prose(
            "The pooled, fit-weighted validation AUC selects one shared checkpoint. Per-mass curves diagnose "
            "whether that average conceals a weak point; they do not retune checkpoints separately. "
            "Training curves are diagnostics, not held-out sensitivity estimates."
        )
    elif section == "thresholds":
        prose(
            f"Exact validation optima use the same support rule as Phase 2: background N_eff >= {cfg['min_background_neff']:g}, "
            "at least one surviving MC event per background process, and one signal event. "
            "Statistics-only and 5%/10% total-background uncertainty retain separate objectives. "
            "The 2% plateau is recorded separately. Unsupported tails cannot supply a reported optimum."
        )
        for record in summary["runs"]:
            display_report_figures(directory, record["run_id"])
        support = tables["process_support"]
        display(support[support.objective == cfg["primary_objective"]].style.format(precision=5, na_rep="unavailable"))
        prose(
            "Shading and threshold histograms use conditional within-process validation bootstraps. They omit "
            "training-seed uncertainty and selection-optimism corrections. They are distinct from the 5%/10% "
            "background-systematic scenarios. Log-axis bands with nonpositive endpoints are omitted; full bounds are saved."
        )
    elif section == "checkpoint":
        effect = json.loads((Path(directory) / "provenance/conditioning_effect.json").read_text())
        prose(
            f"The model split on the mass feature **{effect['mass_split_count']}** times. Changing the requested mass "
            f"on the same background events changes the score by mean absolute **{effect['mean_absolute_score_shift']:.4f}**. "
            "This checks that the conditioning input is used, not that it improves sensitivity."
        )
        prose(f"**Status: {summary['status']}.** {summary['note']}")
        prose(
            "Next: Phase 5 must compare predictions on identical physical events, with paired MC resampling and "
            "the fitting-policy distinction recorded above. No best seed or model family is selected here. "
            "Only the two trained masses are accepted by the inference interface; unseen-mass interpolation and "
            "full-grid expansion remain gated. The test partition remains sealed."
        )
        prose(
            "PDFs are in `figures/`, CSV tables in `tables/`, and settings, hashes and the source archive in "
            "`provenance/`. The clean notebook calls source functions; it never retrains or changes a threshold."
        )
    else:
        raise ValueError(f"Unknown conditional report section: {section}")
