"""Saved-result views shared by PDF exports and the research report notebook."""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from hepml.domain.metrics import OBJECTIVES


def read_evaluation_report(directory):
    directory = Path(directory)
    summary = json.loads((directory / "provenance/summary.json").read_text())
    tables = {path.stem: pd.read_csv(path) for path in (directory / "tables").glob("*.csv")}
    return summary, tables


def checkpoint_text(summary, tables):
    """Summarize saved diagnostics without selecting a policy or a threshold."""
    objective = summary["evaluation_config"]["primary_objective"]
    lines = []
    for row in tables["policy_comparison"].itertuples():
        lines.append(f"- **{row.policy}:** mean weighted validation AUC {row.auc_mean:.4f}; "
                     f"mean {objective} {row.primary_Z_mean:.4g} across {row.valid_seeds} supported seeds.")
    selected = tables["operating_points"]
    selected = selected[(selected.objective == objective) & (selected.status == "valid")]
    if not selected.empty:
        limit = summary["evaluation_config"]["min_background_neff"]
        lines.append(f"- Primary-point background N_eff spans {selected.background_neff.min():.2f} "
                     f"to {selected.background_neff.max():.2f}, against a minimum of {limit:g}. "
                     "Optima near this boundary depend on the MC-support rule; review the bootstrap distributions.")
    lines.append("These are development comparisons on shared validation events. Seed spreads alone do not "
                 "establish a statistically significant policy improvement. Policy adoption remains a checkpoint decision.")
    return "\n\n".join(lines)


def threshold_figure(curve, bands, points, *, title):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    names = ["signal_efficiency", "background_efficiency", "S_over_B", *OBJECTIVES]
    for ax, name in zip(axes.flat, names):
        logarithmic = name in {"S_over_B", "Z_syst_5pct", "Z_syst_10pct"} and (curve[name] > 0).any()
        ax.plot(curve.threshold, curve[name], color="0.55", linewidth=1, linestyle=":",
                label="All thresholds (diagnostic)")
        accepted = curve.where(curve.eligible)
        ax.plot(accepted.threshold, accepted[name], color="C0", linewidth=2, label="MC-supported region")
        if f"{name}_lower" in bands:
            lower, upper = bands[f"{name}_lower"], bands[f"{name}_upper"]
            visible = (lower > 0) & (upper > 0) if logarithmic else lower.notna() & upper.notna()
            ax.fill_between(bands.threshold, lower, upper, where=visible, alpha=.2,
                            label="95% conditional MC band")
        for index, (objective, point) in enumerate(points.items()):
            if point["status"] == "valid":
                ax.axvline(point["threshold"], color=f"C{index+1}", linestyle="--", alpha=.75,
                           label=objective if name == "signal_efficiency" else None)
        ax.set(xlabel="BDT score threshold", ylabel=name, xlim=(0, 1))
        if logarithmic:
            ax.set_yscale("log", nonpositive="mask")
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def stability_figure(stability, *, title):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, objective in zip(axes, OBJECTIVES):
        values = stability.loc[(stability.objective == objective) & (stability.status == "valid"), "threshold"]
        ax.hist(values, bins=25)
        ax.set(title=objective, xlabel="Validation-bootstrap optimum", ylabel="Replicates")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def report_curve_figure(directory, run_id):
    directory = Path(directory)
    summary, _ = read_evaluation_report(directory)
    record = next(row for row in summary["runs"] if row["run_id"] == run_id)
    model = Path(record["model_directory"])
    curve = pd.read_parquet(model / "threshold_curves.parquet")
    bands = pd.read_parquet(model / "threshold_bands.parquet")
    evaluation = json.loads((model / "metrics.json").read_text())["evaluation"]
    return threshold_figure(curve, bands, evaluation["operating_points"],
                            title=f"{run_id}: validation at {evaluation['lumi_pb_inv']/1000:g} fb^-1")


def display_report_figures(directory, run_id):
    """Embed selected saved-result figures even in a noninteractive/Agg kernel."""
    summary, _ = read_evaluation_report(directory)
    record = next(row for row in summary["runs"] if row["run_id"] == run_id)
    model = Path(record["model_directory"])
    figures = [report_curve_figure(directory, run_id),
               stability_figure(pd.read_csv(model / "threshold_stability.csv"), title=run_id)]
    for figure in figures:
        display_figure(figure)


def display_figure(figure):
    from IPython.display import Image, display

    try:
        with BytesIO() as buffer:
            figure.savefig(buffer, format='png', dpi=140, bbox_inches='tight')
            display(Image(data=buffer.getvalue()))
    finally:
        plt.close(figure)


def diagnostic_figure(model_directory, *, title):
    directory = Path(model_directory)
    roc = pd.read_parquet(directory / 'roc.parquet')
    history = json.loads((directory / 'learning_curves.json').read_text())
    metrics = json.loads((directory / 'metrics.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, group in roc.groupby('split'):
        axes[0].plot(group.background_efficiency, group.signal_efficiency, label=name)
    axes[0].plot([0, 1], [0, 1], linestyle=':', color='gray')
    axes[0].set(xlabel='Background efficiency (physical mixture)', ylabel='Signal efficiency', title='Weighted ROC')
    for key, name in [('validation_0', 'train'), ('validation_1', 'validation')]:
        axes[1].plot(history[key]['auc'], label=name)
    axes[1].axvline(metrics['xgb']['best_iteration'], color='gray', linestyle=':', label='Best iteration')
    axes[1].set(xlabel='Boosting iteration (zero-based)', ylabel='Fit-objective AUC', title='Early-stopping history')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def save_reference_figures(directory):
    """Full PDF collections reuse the notebook's plotting implementations."""
    from matplotlib.backends.backend_pdf import PdfPages

    directory = Path(directory)
    summary, _ = read_evaluation_report(directory)
    with (PdfPages(directory / 'figures/threshold_curves.pdf') as curves,
          PdfPages(directory / 'figures/threshold_stability.pdf') as stability,
          PdfPages(directory / 'figures/roc_learning.pdf') as diagnostics):
        for record in summary['runs']:
            run, model = record['run_id'], Path(record['model_directory'])
            figures = [(curves, report_curve_figure(directory, run)),
                       (stability, stability_figure(pd.read_csv(model / 'threshold_stability.csv'), title=run)),
                       (diagnostics, diagnostic_figure(model, title=run))]
            for pdf, figure in figures:
                pdf.savefig(figure)
                plt.close(figure)


def display_baseline_section(directory, section):
    """Narrative views for the shared baseline notebook, using only saved artifacts."""
    from IPython.display import Markdown, display

    summary, tables = read_evaluation_report(directory)
    phase2 = summary['phases'] == [2]
    cfg, settings = summary['evaluation_config'], summary['settings']
    primary = cfg['primary_objective']
    first_seed = settings['seeds'][0]
    selected = [r for r in summary['runs'] if r['seed'] == first_seed]

    def prose(text):
        display(Markdown(text))

    def table(name, columns=None, first_only=False):
        frame = tables[name]
        if first_only:
            frame = frame[frame.seed == first_seed]
        if columns:
            frame = frame[columns]
        display(frame.style.format(precision=6, na_rep='unavailable'))

    if section == 'scope':
        prose('**Phase 2:** independent mass-specific BDT references and a bounded cut reference.' if phase2 else
              '**Phases 4/7/8:** fitting weights, physical yields and validation operating points.')
        prose(f"Evaluation at **{cfg['lumi_pb_inv']/1000:g} fb^-1**, with signal k={cfg.get('signal_k_factor',1):g} "
              f"and background k={cfg.get('background_k_factor',1):g}, applied once. "
              f"Primary objective: **{primary}**. Test evaluated: **{summary['test_evaluated']}**.")
        if phase2:
            prose(f"Masses: {settings['masses']} GeV at rho_tc={settings['rho_tc']:g}; fitting policy: "
                  f"**{settings['fit_policy']}**. {settings['policy_reason']} "
                  'These results refer to those physical points; they do not establish coupling independence.')
        else:
            prose('Unweighted fitting is the historical control. Balanced-mixture fitting preserves physical '
                  'process proportions within each class and balances class totals. No policy is adopted automatically.')
        display(settings)
        prose('Saved event assignments and preparation weights are preserved. Per-process efficiencies normalize '
              'each split to its full pre-BDT rate; luminosity and k-factors then give expected event counts. '
              'Alternative signal hypotheses are never added as physical rates.')
    elif section == 'checks':
        table('checks')
        if phase2:
            table('split_inventory')
        else:
            table('fit_weight_audit')
            table('hypothesis_fit_priors')
            table('constraint_audit')
    elif section == 'results':
        if not phase2:
            prose(checkpoint_text(summary, tables))
            table('policy_comparison')
        table('seed_comparison')
        prose('The operating-point table shows the first seed for each reference; full CSVs include every seed. '
              'S and B are full expected event counts. Each objective has its own validation optimum.')
        table('operating_points', ['mass' if phase2 else 'policy', 'objective', 'status', 'threshold',
              'plateau_threshold', 'S', 'B', 'S_over_B', *OBJECTIVES, 'background_neff'], first_only=True)
        if phase2:
            prose('**Cut reference:** pass either c-b mass window AND the activity threshold. The recorded grid '
                  'includes the no-additional-cut control; every recipe is evaluated once per mass on validation. '
                  'Each objective chooses its own supported recipe, with candidate order breaking exact ties. '
                  'This small control is separate from the externally supplied m300/rho04 cut-based result.')
            table('cut_operating_points', ['mass', 'objective', 'status', 'candidate_id', 'relative_half_width',
                  'minimum_activity_over_mass', 'S', 'B', *OBJECTIVES, 'background_neff'])
            for record in selected:
                display_figure(diagnostic_figure(record['model_directory'], title=record['run_id']))
            prose('ROC curves use physical mixture weights. Early stopping uses the fitting policy: for unweighted '
                  'fits its AUC is unweighted. Training performance is a diagnostic, not held-out sensitivity.')
    elif section == 'thresholds':
        prose(f"Threshold selection requires background N_eff >= {cfg['min_background_neff']:g}, "
              f"at least {cfg['min_process_mc']} MC event(s) per background process, and "
              f"{cfg['min_signal_mc']} signal event(s). No valid selection is explicitly recorded when unsupported. "
              'Raw optima and the separately recorded 2% plateau points are distinct.')
        prose("Statistical significance and [Cowan Eq. 20](https://www.pp.rhul.ac.uk/~cowan/stat/medsig/medsigNote.pdf) "
              'with 5%/10% total-background uncertainty share expected-count inputs. These scenarios are not a full '
              'correlated nuisance likelihood. Gray dotted curves include unsupported tails; blue curves are supported. '
              'S/B and systematic significance use log axes; bands with nonpositive endpoints are omitted there.')
        for record in selected:
            display_report_figures(directory, record['run_id'])
        prose('Shading is a pointwise 95% within-process bootstrap interval conditional on fitted scores and '
              'preselection rates. Histograms show reselected validation thresholds. Neither includes training '
              'uncertainty or corrects for validation optimization; seed variation is reported separately.')
        support = tables['process_support']
        chosen = support[support.run_id.isin([r['run_id'] for r in selected]) & (support.objective == primary)]
        available = [c for c in ['mass','run_id','sample','mc_count','N_eff','sum_weight','yield_lower','yield_upper'] if c in chosen]
        display(chosen[available].style.format(precision=4, na_rep='unavailable'))
        prose('Process intervals are exact binomial efficiency intervals at fixed preselection rates for uniform '
              'per-process MC weights. Expected-event intervals differ from raw MC counts. Zero survivors do not '
              'establish a zero rate. The full process table includes every objective and cut reference.')
    elif section == 'checkpoint':
        if phase2:
            for mass, group in tables['seed_comparison'].groupby('mass'):
                points = tables['operating_points']
                points = points[(points.mass == mass) & (points.objective == primary)]
                cut = tables['cut_operating_points']
                cut = cut[(cut.mass == mass) & (cut.objective == primary)]
                prose(f"**m{mass}:** mean weighted validation AUC {group.auc_weighted.mean():.4f} "
                      f"(seed range {group.auc_weighted.min():.4f}-{group.auc_weighted.max():.4f}); "
                      f"mean {primary} {points[primary].mean():.4g}; cut reference {cut[primary].iloc[0]:.4g}.")
            prose('Compare the MC-support boundary and seed/threshold spreads before interpreting improvements. '
                  'These development results do not establish final-test performance or paired statistical superiority. '
                  'The chosen unweighted policy is fixed for the subsequent parameterized comparison; no best seed is selected.')
        prose(f"**Status:** {summary['status']}. {summary['note']}")
        prose('The full PDF collections are in `figures/`, machine-readable tables in `tables/`, and settings, '
              'hashes and the source archive in `provenance/`. Models, predictions and complete curves remain '
              'in the referenced run directories. This notebook never retrains or changes operating points.')
    else:
        raise ValueError(f'Unknown report section: {section}')


def default_report_directory():
    """Notebook can be opened from the checkout or its saved report bundle."""
    return Path(os.environ.get("HEPML_EVALUATION_REPORT", "outputs/cg_bbc/reports/phase2-rho01-v1"))
