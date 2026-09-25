#!/usr/bin/env python3
"""
predict.py

Inference-only script for frozen XGBoost models.

Reads:
  - model.ubj (preferred) or model.joblib
  - features.txt (ordered feature list)
  - threshold.json (decision threshold + best_iteration + metadata)

Inputs:
  - Parquet or CSV file with the required feature columns

Outputs:
  - parquet/csv with bdt_score and bdt_pred (score >= threshold)

Example:
  hepml predict --mass 200 \
    --model-dir outputs/cg_bbc/releases \
    --input outputs/cg_bbc/datasets/splits/test_sig200.parquet \
    --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hepml.adapters.configuration import output_paths
from hepml.adapters.inference import (
    load_best_iteration,
    load_model,
    load_split_meta,
    load_threshold,
    read_features,
    read_frozen_record,
    read_input,
    write_output,
)
from hepml.adapters.releases import resolve_model_dir
from hepml.domain.artifacts import dataset_filename, split_filename
from hepml.domain.metrics import sample_normalization
from hepml.domain.validation import check_no_nan_inf, require_numeric, sanity_predictions
from hepml.log import get_logger

log = get_logger(__name__)


# --------------------------------------
# Main
# --------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Run inference with a frozen model.")
    ap.add_argument(
        "--model-dir", default=None, help="final models directory (default: <output-root>/<study>/releases/)"
    )
    ap.add_argument("--mass", type=int, default=200, help="Signal mass tag, e.g. 200")
    ap.add_argument('--hypothesis-mass',type=float,help='Explicit mass for a direct conditional model/release directory')
    ap.add_argument(
        "--ver",
        default="latest",
        help=(
            "Frozen model version: integer (e.g. 1) or 'latest'. "
            "If no frozen model exists, fall back to working dir ml_models/sig{mass}."
        ),
    )
    ap.add_argument("--input", required=True, help="Input parquet/csv with feature columns")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which infer file to generate")
    ap.add_argument("--out-name", default=None, help="Override output filename (default: preds_infer_{split}.parquet)")
    ap.add_argument(
        "--outdir-name", default=None, help="Optional output directory. If not set, outputs go under model_dir"
    )
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="Input/output format")
    ap.add_argument(
        "--keep-cols",
        nargs="*",
        default=["target", "sample", "sample_weight", "gen_weight"],
        help="Optional columns to keep if present",
    )
    ap.add_argument("--no-pred", action="store_true", help="Do not create hard label bdt_pred")
    ap.add_argument("--objective", choices=["Z_Asimov", "Z_syst_5pct", "Z_syst_10pct"],
                    help="Use one of the frozen validation operating points")
    args = ap.parse_args(argv)

    mass = int(args.mass if args.hypothesis_mass is None else args.hypothesis_mass)
    final_root = Path(args.model_dir) if args.model_dir is not None else output_paths().models_final
    direct_conditional = ((final_root / 'metrics.json').is_file()
                          and 'conditioning' in json.loads((final_root / 'metrics.json').read_text()))
    if direct_conditional and args.hypothesis_mass is None:
        raise ValueError('Conditional inference requires --hypothesis-mass explicitly')
    if args.hypothesis_mass is not None and not direct_conditional:
        raise ValueError('--hypothesis-mass requires a direct conditional model directory')
    model_dir = final_root if direct_conditional else resolve_model_dir(final_root, mass, args.ver)

    in_path = Path(args.input)

    if args.outdir_name is not None:
        out_dir = Path(args.outdir_name)
    else:
        out_dir = model_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f'_m{mass}' if direct_conditional else ''
    out_name = args.out_name or f"preds_infer_{args.split}{suffix}.{args.format}"
    out_path = out_dir / out_name

    features_path = model_dir / "features.txt"
    threshold_path = model_dir / "threshold.json"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Missing {threshold_path}")

    features = read_features(features_path)
    thr = load_threshold(threshold_path, args.objective, hypothesis_mass=args.hypothesis_mass)
    frozen = read_frozen_record(threshold_path,args.hypothesis_mass)

    model_kind, model_path, predict_scores = load_model(model_dir)
    log.info("Loaded model (%s): %s", model_kind, model_path)
    log.info("Loaded features: %s (%d features)", features_path, len(features))
    log.info("Loaded threshold: %s (thr=%.6f)", threshold_path, thr)

    df = read_input(in_path, args.format)
    if direct_conditional:
        from hepml.application.conditioning import condition_frame

        df = condition_frame(df,args.hypothesis_mass,frozen['conditioning'])
    # ---- schema sanity ----
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required features in input: {missing}")
    # ---- value sanity ----
    require_numeric(df, features, name="predict_input")
    check_no_nan_inf(df, features, name="predict_input")

    X = df[features].to_numpy(dtype=np.float32)
    scores = predict_scores(X)

    # Build output with a minimal contract:
    keep = [c for c in args.keep_cols if c in df.columns]
    out = df[keep].copy() if keep else pd.DataFrame(index=df.index)
    out["bdt_score"] = scores

    if not args.no_pred:
        out["bdt_pred"] = (out["bdt_score"] >= thr).astype(np.int8)
        for objective, point in frozen.get("evaluation", {}).get("operating_points", {}).items():
            if point["status"] == "valid":
                out[f"bdt_pred_{objective}"] = (out.bdt_score >= point["threshold"]).astype(np.int8)

    # ---- score sanity ----
    sanity_predictions(out, score_col="bdt_score", name="predict_output")

    # Helpful metadata
    pred_path = out_path
    if pred_path.suffix.lower() not in [".parquet", ".csv"]:
        pred_path = pred_path.with_suffix(f".{args.format}")
    write_output(out, pred_path, args.format)

    meta = {
        "model_dir": str(model_dir),
        "model_kind": model_kind,
        "model_path": str(model_path),
        "best_iteration": load_best_iteration(model_dir),
        "threshold": float(thr),
        "features_path": str(features_path),
        "threshold_path": str(threshold_path),
        "n_rows": int(len(out)),
        "keep_cols": keep,
        "features": features,
        "evaluation": frozen.get("evaluation"),
        "objective": args.objective or frozen.get("objective"),
        "hypothesis_mass": args.hypothesis_mass,
    }

    is_split_input = in_path.parent.name == "splits" and in_path.name == split_filename(args.split, mass)
    if not is_split_input:
        log.warning("Input is not a named prepared split: %s; reporting file-level yields", in_path)

    split_meta = load_split_meta(mass, in_path.parent) if is_split_input else {}
    weighted_frac_current = split_meta.get(f"weighted_frac_{args.split}")
    if is_split_input and {"sample", "target", "sample_weight"}.issubset(out.columns):
        full = pd.read_parquet(in_path.parent.parent / dataset_filename(mass),
                               columns=["sample", "target", "sample_weight"])
        meta["normalization"] = sample_normalization(full, df)
        prepared_lumi = float(split_meta["lumi"])
        metrics_path = model_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        evaluation_lumi = float(frozen["evaluation"]["lumi_pb_inv"] if frozen.get("evaluation") is not None
                                else metrics.get("lumi_pb_inv", prepared_lumi))
        meta["prepared_lumi_pb_inv"] = prepared_lumi
        meta["evaluation_lumi_pb_inv"] = evaluation_lumi
        meta["evaluation_weight_scale"] = evaluation_lumi / prepared_lumi
    meta["split_meta"] = split_meta

    meta["input_is_split"] = is_split_input
    meta["weighted_frac_current_split"] = weighted_frac_current

    meta_path = pred_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    log.info("Wrote: %s (rows=%d)", pred_path, len(out))
    log.info("Wrote: %s", meta_path)


if __name__ == "__main__":
    main()
