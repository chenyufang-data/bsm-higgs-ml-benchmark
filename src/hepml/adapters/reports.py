#!/usr/bin/env python3
"""Inference report serialization and Markdown serialization."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from hepml.adapters.configuration import output_paths
from hepml.domain.artifacts import split_meta_path
from hepml.domain.metrics import rho_scan as rho_scan
from hepml.log import get_logger

log = get_logger(__name__)


def _load_split_meta(mass: int) -> dict:
    """Split metadata written by prepare_ml; {} when unavailable."""
    p = split_meta_path(output_paths().splits, mass)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Could not read split meta %s (%s)", p, e)
        return {}


def safe_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def write_markdown_report(report: dict[str, Any], out_path: Path):
    """
    Write a concise markdown summary from the report dict.
    """
    mass = report.get("signal_mass", "NA")
    split = report.get("split", "NA")
    n_rows = report.get("n_rows", "NA")
    thr = report.get("threshold", "NA")

    score_stats = report.get("score_stats", {})
    yields = report.get("yields_weighted", {})
    by_sample = report.get("by_sample", {}).get("top_by_weight_sum", [])

    Z_full = yields.get("Z_Asimov_pass_full", None)
    Z_eval = yields.get("Z_Asimov_pass_eval", None)
    S_eff = yields.get("S_eff", None)
    B_eff = yields.get("B_eff", None)
    S_pass_full = yields.get("S_pass_full", None)
    B_pass_full = yields.get("B_pass_full", None)

    date_str = datetime.now().strftime("%Y-%m-%d")

    lines = []

    # Header
    lines.append(f"# Model Inference Summary - sig{mass} ({split})")
    lines.append(f"> Generated on {date_str} by `hepml summarize`")
    lines.append("")

    # Executive summary. Lead with the split-level number; the full-stat
    # figure is an extrapolation via the weighted split fractions.
    lines.append("## Executive Summary")
    version2 = bool(report.get("operating_points"))
    if version2:
        lines.append(f"- Validation-selected objective: **{report['objective']}**. "
                     f"Yields use per-process efficiencies at **{yields['evaluation_lumi_pb_inv']/1000:g} fb^-1**.")
        lines.append("- All three frozen operating points are reported below; this split does not choose thresholds.")
        factors = yields.get("k_factors")
        if factors:
            lines.append(f"- Cross-section k-factors: signal {factors['signal']:g}, "
                         f"background {factors['background']:g}, applied once to raw manifest rates.")
    elif Z_eval is not None:
        lines.append(
            f"- On the evaluated {split} split the model achieves **Z_Asimov = {Z_eval:.2f} sigma** "
            f"at the selected threshold."
        )
    if Z_full is not None and not version2:
        lines.append(
            f"- Extrapolated to the full dataset statistics (scaled by the split fractions), "
            f"**Z_Asimov = {Z_full:.2f} sigma**, "
            f"with **{S_eff * 100:.1f}% signal efficiency** and **{B_eff * 100:.2f}% background efficiency**."
        )
    if Z_eval is None and Z_full is None:
        lines.append("- Inference completed successfully. See metrics below for details.")
    lines.append("")

    # Dataset
    lines.append("## Dataset")
    lines.append(f"- **Signal mass:** {mass} GeV")
    lines.append(f"- **Split:** {split}")
    lines.append(f"- **Rows evaluated:** {n_rows}")
    lines.append("")

    # Model decision
    lines.append("## Decision Threshold")
    if isinstance(thr, float):
        lines.append(f"- **BDT threshold:** {thr:.3f}")
    else:
        lines.append(f"- **BDT threshold:** {thr}")
    lines.append("")

    # Core metrics
    lines.append("## Key Metrics")
    if Z_eval is not None:
        lines.append(f"- **Z_Asimov (evaluated split):** {Z_eval:.3f} sigma")
    if Z_full is not None:
        lines.append(f"- **Z_Asimov (full-stat extrapolation, pass):** {Z_full:.3f} sigma")
    if S_eff is not None:
        lines.append(f"- **Signal efficiency (weighted):** {S_eff * 100:.2f}%")
    if B_eff is not None:
        lines.append(f"- **Background efficiency (weighted):** {B_eff * 100:.2f}%")
    if S_pass_full is not None and B_pass_full is not None:
        lines.append(f"- **Expected S (pass):** {S_pass_full:,.1f} events")
        lines.append(f"- **Expected B (pass):** {B_pass_full:,.1f} events")
    lines.append("")

    if report.get("operating_points"):
        lines.append("## Frozen validation operating points")
        lines.append("")
        lines.append(f"Full expected yields at {yields['evaluation_lumi_pb_inv']/1000:g} fb^-1, "
                     "using per-process efficiencies. Thresholds are applied without rescanning this split.")
        lines.append("")
        lines.append("| Objective | Threshold | S | B | S/B | Z (stat) | Z (5%) | Z (10%) | Background N_eff |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for point in report["operating_points"]:
            ratio = "undefined" if point["S_over_B"] is None else f"{point['S_over_B']:.4g}"
            lines.append(f"| {point['objective']} | {point['threshold']:.6g} | {point['S']:.6g} | "
                         f"{point['B']:.6g} | {ratio} | {point['Z_Asimov']:.4g} | "
                         f"{point['Z_syst_5pct']:.4g} | {point['Z_syst_10pct']:.4g} | "
                         f"{point['background_neff']:.4g} |")
        lines.append("")

    # Score distribution
    lines.append("## Score Distribution")
    if score_stats:
        lines.append(
            f"- **Min / Mean / Max:** "
            f"{score_stats.get('min', float('nan')):.3f} / "
            f"{score_stats.get('mean', float('nan')):.3f} / "
            f"{score_stats.get('max', float('nan')):.3f}"
        )
        lines.append(
            f"- **P50 / P90 / P95 / P99:** "
            f"{score_stats.get('p50', float('nan')):.3f} / "
            f"{score_stats.get('p90', float('nan')):.3f} / "
            f"{score_stats.get('p95', float('nan')):.3f} / "
            f"{score_stats.get('p99', float('nan')):.3f}"
        )
    lines.append("")

    # Per-rho significance (signal coupling grid)
    rho = report.get("rho_scan")
    if rho and rho.get("rows"):
        lines.append("## Significance vs rho_tc")
        lines.append(f"> xs_ref = {rho['xs_ref_pb']:.6g} pb; S rescaled by xs(rho)/xs_ref, B unchanged.")
        lines.append("")
        lines.append("| rho_tc | xs [pb] | Z @ frozen thr | best thr | Z @ best thr |")
        lines.append("|--------|---------|----------------|----------|--------------|")
        for row in rho["rows"]:
            lines.append(
                f"| {row['rho_tc']:g} | {row['xs_pb']:.6g} | {row['Z_frozen_thr']:.3f} | "
                f"{row['best_thr']:.3f} | {row['Z_best_thr']:.3f} |"
            )
        lines.append("")

    # Top contributors
    if by_sample:
        lines.append("## Top Contributors (by weight)")
        lines.append("| Sample | Events | Pass rate (unweighted) | Weight sum | Weight pass |")
        lines.append("|--------|--------|------------------------|------------|-------------|")
        for row in by_sample:
            lines.append(
                f"| {row.get('sample', '')} | {row.get('n', '')} | "
                f"{row.get('pass_rate', 0) * 100:.1f}% | "
                f"{row.get('w_sum', 0):.1f} | {row.get('w_pass', 0):.1f} |"
            )
        lines.append("")

    # Notes
    lines.append("## Notes")
    lines.append("- `sample_weight` includes luminosity.")
    if version2:
        lines.append("- Full expected yields use each process's retained fraction times its full pre-BDT rate. "
                     "The finite MC support remains that of the evaluated split.")
    else:
        lines.append("- Full-stat metrics are extrapolated using recorded split-normalization metadata.")
    lines.append("- Per-sample pass rates are unweighted; the key metrics are weighted.")
    lines.append("- This summary is intended for quick inspection; see the JSON report for full details.")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote markdown report: %s", out_path)
