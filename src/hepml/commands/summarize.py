#!/usr/bin/env python3
"""Summarize predictions at a frozen model threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hepml.adapters.configuration import output_paths
from hepml.adapters.inference import load_threshold, read_frozen_record
from hepml.adapters.releases import resolve_model_dir
from hepml.adapters.reports import _load_split_meta, rho_scan, safe_float, write_markdown_report
from hepml.domain.config import EvaluationConfig
from hepml.domain.metrics import asimov_z, normalized_weights, operating_curve, yield_metrics
from hepml.domain.weights import cross_section_factors
from hepml.log import get_logger

log = get_logger(__name__)


# --------------------------------------
# Main
# --------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarize inference outputs into reports.")
    ap.add_argument(
        "--model-dir", default=None, help="final models directory (default: <output-root>/<study>/releases/)"
    )
    ap.add_argument("--mass", type=int, default=200, help="Signal mass tag, e.g. 200")
    ap.add_argument('--hypothesis-mass',type=float,help='Explicit mass for a direct conditional release directory')
    ap.add_argument(
        "--ver",
        default="latest",
        help=(
            "Frozen model version: integer (e.g. 1) or 'latest'. "
            "If no frozen model exists, fall back to working directory <output-root>/<study>/runs/sig{mass}."
        ),
    )
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which infer file to summarize")
    ap.add_argument("--preds-name", default=None, help="Override preds filename (default: preds_infer_{split}.parquet)")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="Preds format")
    ap.add_argument("--out-name", default=None, help="Override report filename (default: report_infer_{split}.json)")
    ap.add_argument(
        "--write-by-sample-csv", action="store_true", help="Write per-sample summary CSV next to JSON report"
    )
    args = ap.parse_args(argv)

    mass = int(args.mass if args.hypothesis_mass is None else args.hypothesis_mass)
    final_root = Path(args.model_dir) if args.model_dir is not None else output_paths().models_final
    direct_conditional = ((final_root / 'metrics.json').is_file()
                          and 'conditioning' in json.loads((final_root / 'metrics.json').read_text()))
    if direct_conditional != (args.hypothesis_mass is not None):
        raise ValueError('Conditional summaries require a direct model directory and --hypothesis-mass')
    model_dir = final_root if direct_conditional else resolve_model_dir(final_root, mass, args.ver)

    suffix = f'_m{mass}' if direct_conditional else ''
    preds_name = args.preds_name or f"preds_infer_{args.split}{suffix}.{args.format}"
    meta_name = Path(preds_name).with_suffix(".meta.json")
    out_name = args.out_name or f"report_infer_{args.split}{suffix}.json"

    preds_path = model_dir / preds_name
    thr_path = model_dir / "threshold.json"
    meta_path = model_dir / meta_name
    out_path = model_dir / out_name

    if not preds_path.exists():
        raise FileNotFoundError(f"Missing preds file: {preds_path}")
    if not thr_path.exists():
        raise FileNotFoundError(f"Missing threshold.json: {thr_path}")

    # load threshold
    thr_json = read_frozen_record(thr_path,args.hypothesis_mass)
    thr = float(thr_json["threshold"])

    # load preds
    if args.format == "csv":
        df = pd.read_csv(preds_path)
    else:
        df = pd.read_parquet(preds_path)

    meta: dict[str, Any] = {}
    meta_error = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            meta_error = str(error)
    evaluation = thr_json.get("evaluation")
    objective = meta.get("objective") or thr_json.get("objective")
    if evaluation is not None:
        if meta.get("evaluation") != evaluation:
            raise ValueError("Prediction metadata differs from the frozen evaluation protocol")
        thr = load_threshold(thr_path, objective, hypothesis_mass=args.hypothesis_mass)
        # The named objective can differ from the release's primary objective.
        df["bdt_pred"] = (df["bdt_score"] >= thr).astype(np.int8)

    required = ["bdt_score"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in {preds_path}")

    # if bdt_pred missing, derive from threshold
    if "bdt_pred" not in df.columns:
        df["bdt_pred"] = (df["bdt_score"] >= thr).astype(np.int8)

    # optional columns
    has_target = "target" in df.columns
    has_weight = "sample_weight" in df.columns
    has_sample = "sample" in df.columns

    # basic stats
    n_rows = int(len(df))
    score = df["bdt_score"].to_numpy(dtype=np.float64)
    pred = df["bdt_pred"].to_numpy(dtype=np.int8)

    report: dict[str, Any] = {
        "signal_mass": str(mass),
        "model_dir": str(model_dir),
        "split": args.split,
        "preds_file": preds_name,
        "n_rows": n_rows,
        "threshold": thr,
        "threshold_source": "threshold.json",
        "objective": objective,
        "score_stats": {
            "min": safe_float(np.min(score)) if n_rows else float("nan"),
            "mean": safe_float(np.mean(score)) if n_rows else float("nan"),
            "max": safe_float(np.max(score)) if n_rows else float("nan"),
            "p50": safe_float(np.quantile(score, 0.50)) if n_rows else float("nan"),
            "p90": safe_float(np.quantile(score, 0.90)) if n_rows else float("nan"),
            "p95": safe_float(np.quantile(score, 0.95)) if n_rows else float("nan"),
            "p99": safe_float(np.quantile(score, 0.99)) if n_rows else float("nan"),
        },
        "pass_rate_unweighted": safe_float(pred.mean()) if n_rows else 0.0,
        "columns_present": {
            "target": bool(has_target),
            "sample": bool(has_sample),
            "sample_weight": bool(has_weight),
            "bdt_pred": True,
        },
        "provenance": {},
    }

    # attach optional meta provenance
    if meta:
        report["provenance"]["preds_meta"] = meta
    if meta_error:
        report["provenance"]["preds_meta_error"] = meta_error

    # confusion matrix + weighted yields
    if has_target:
        y = df["target"].to_numpy(dtype=np.int8)
        tp = int(((y == 1) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())

        report["confusion_unweighted"] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
        report["rates_unweighted"] = {
            "tpr": safe_float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
            "fpr": safe_float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan"),
            "tnr": safe_float(tn / (fp + tn)) if (fp + tn) > 0 else float("nan"),
            "fnr": safe_float(fn / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        }

        if has_weight:
            wf = None
            if (
                isinstance(meta, dict)
                and meta.get("input_is_split")
                and isinstance(meta.get("weighted_frac_current_split"), dict)
            ):
                wf = meta["weighted_frac_current_split"]

            wxs = df["sample_weight"].to_numpy(dtype=np.float64)

            S_pass_eval = float(wxs[(y == 1) & (pred == 1)].sum())
            B_pass_eval = float(wxs[(y == 0) & (pred == 1)].sum())
            S_all_eval = float(wxs[(y == 1)].sum())
            B_all_eval = float(wxs[(y == 0)].sum())

            # optional scale-back
            S_pass_full, B_pass_full = S_pass_eval, B_pass_eval
            S_all_full, B_all_full = S_all_eval, B_all_eval

            normalization = meta.get("normalization") if isinstance(meta, dict) else None
            full_weights = wxs
            if normalization is not None:
                full_weights = normalized_weights(df, normalization) * float(meta.get("evaluation_weight_scale", 1.0))
                if evaluation is not None:
                    cfg = EvaluationConfig(**evaluation["config"])
                    full_weights *= cross_section_factors(df.target, signal_k_factor=cfg.signal_k_factor,
                                                         background_k_factor=cfg.background_k_factor)
                S_pass_full = float(full_weights[(y == 1) & (pred == 1)].sum())
                B_pass_full = float(full_weights[(y == 0) & (pred == 1)].sum())
                S_all_full = float(full_weights[y == 1].sum())
                B_all_full = float(full_weights[y == 0].sum())
            elif wf is not None:
                f_sig = float(wf.get("f_sig", 1.0))
                f_bkg = float(wf.get("f_bkg", 1.0))
                if f_sig > 0:
                    S_pass_full = S_pass_eval / f_sig
                    S_all_full = S_all_eval / f_sig
                if f_bkg > 0:
                    B_pass_full = B_pass_eval / f_bkg
                    B_all_full = B_all_eval / f_bkg

            report["yields_weighted"] = {
                "sample_weight_includes_lumi": True,
                "normalization": normalization,
                "evaluation_lumi_pb_inv": meta.get("evaluation_lumi_pb_inv") if isinstance(meta, dict) else None,
                "k_factors": ({"signal": cfg.signal_k_factor, "background": cfg.background_k_factor}
                              if evaluation is not None and normalization is not None else None),
                "weighted_frac_used": wf if normalization is None else None,
                # what this file contains (split-level)
                "S_pass_eval": S_pass_eval,
                "B_pass_eval": B_pass_eval,
                "S_all_eval": S_all_eval,
                "B_all_eval": B_all_eval,
                "Z_Asimov_pass_eval": asimov_z(S_pass_eval, B_pass_eval),
                "S_over_sqrtB_pass_eval": safe_float(S_pass_eval / np.sqrt(B_pass_eval)) if B_pass_eval > 0 else 0.0,
                # scaled-back (HEP "full-stat equivalent")
                "S_pass_full": S_pass_full,
                "B_pass_full": B_pass_full,
                "S_all_full": S_all_full,
                "B_all_full": B_all_full,
                "Z_Asimov_pass_full": asimov_z(S_pass_full, B_pass_full),
                "S_over_sqrtB_pass_full": safe_float(S_pass_full / np.sqrt(B_pass_full)) if B_pass_full > 0 else 0.0,
                "S_eff": safe_float(S_pass_full / S_all_full) if S_all_full > 0 else float("nan"),
                "B_eff": safe_float(B_pass_full / B_all_full) if B_all_full > 0 else float("nan"),
            }
            report["yields_weighted"]["full_metrics"] = yield_metrics(S_pass_full, B_pass_full)
            if evaluation is not None:
                if evaluation.get("threshold_source") != "validation":
                    raise ValueError("Frozen operating points must come from validation")
                if normalization is None:
                    raise ValueError("Version-2 reporting requires per-process split normalization")
                points = evaluation["operating_points"]
                thresholds = np.unique([p["threshold"] for p in points.values() if p["status"] == "valid"])
                cfg = EvaluationConfig(**evaluation["config"])
                curve, _ = operating_curve(df, full_weights, cfg, thresholds=thresholds)
                report["operating_points"] = [dict(curve.loc[curve.threshold == point["threshold"]].iloc[0].to_dict(),
                                                   objective=objective, threshold_source="validation")
                                               for objective, point in points.items() if point["status"] == "valid"]

            # ---- per-rho_tc significance (signal grid: xs rescaling only) ----
            split_meta = meta.get("split_meta", {}) if isinstance(meta, dict) else {}
            if not split_meta:
                split_meta = _load_split_meta(mass)
            sig_info = split_meta.get("signal") or {}
            xs_by_rho = sig_info.get("xs_by_rho")
            if xs_by_rho and evaluation is None:
                f_sig = float(wf.get("f_sig", 1.0)) if wf and normalization is None else 1.0
                f_bkg = float(wf.get("f_bkg", 1.0)) if wf and normalization is None else 1.0
                rows = rho_scan(
                    score,
                    y,
                    full_weights,
                    xs_by_rho=xs_by_rho,
                    xs_ref=float(sig_info["xs_pb"]),
                    frozen_thr=thr,
                    f_sig=f_sig,
                    f_bkg=f_bkg,
                )
                report["rho_scan"] = {
                    "xs_ref_pb": float(sig_info["xs_pb"]),
                    "note": (
                        "S scaled by xs(rho)/xs_ref with B unchanged (kinematics are "
                        "rho-independent); best_thr is an unconstrained argmax over the "
                        "stored scores at full-stat yields."
                    ),
                    "rows": rows,
                }

    # by-sample breakdown
    if has_sample:
        group_cols = ["sample"]
        agg: dict[str, Any] = {
            "n": ("bdt_score", "size"),
            "pass_rate": ("bdt_pred", "mean"),
            "score_mean": ("bdt_score", "mean"),
            "score_p95": ("bdt_score", lambda s: float(np.quantile(s, 0.95))),
        }
        if has_weight:
            agg["w_sum"] = ("sample_weight", "sum")
            agg["w_pass"] = ("sample_weight", lambda s: float(s[df.loc[s.index, "bdt_pred"] == 1].sum()))

        by = df.groupby(group_cols, sort=False).agg(**agg).reset_index()

        # clean types
        by["pass_rate"] = by["pass_rate"].astype(float)

        report["by_sample"] = {
            "n_samples": int(len(by)),
            "top_by_weight_sum": by.sort_values("w_sum", ascending=False).head(10).to_dict(orient="records")
            if has_weight and "w_sum" in by.columns
            else by.sort_values("n", ascending=False).head(10).to_dict(orient="records"),
        }

        if args.write_by_sample_csv:
            csv_path = model_dir / f"report_by_sample_{args.split}.csv"
            by.to_csv(csv_path, index=False)
            report["by_sample_csv"] = str(csv_path)

    # write report
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote report: %s", out_path)

    # quick console summary
    log.info("n_rows=%d, pass_rate=%.4f, thr=%.6f", n_rows, report["pass_rate_unweighted"], thr)
    if "yields_weighted" in report:
        yw = report["yields_weighted"]
        log.info(
            "Weighted pass: S=%.4e, B=%.4e, Z_Asimov=%.4g",
            yw["S_pass_eval"],
            yw["B_pass_eval"],
            yw["Z_Asimov_pass_eval"],
        )

    write_markdown_report(report, out_path.with_suffix(".md"))


if __name__ == "__main__":
    main()
