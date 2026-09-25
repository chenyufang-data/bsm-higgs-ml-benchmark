#!/usr/bin/env python3
"""Run feature ablation through the public training CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from hepml.adapters.configuration import default_study_directory, output_paths
from hepml.adapters.study_loader import load_study


def run_train(
    *,
    mass: str,
    outdir: Path,
    splits_dir: Path,
    features: list[str],
    drop_features: list[str],
    class_balance: bool,
    seed: int,
) -> Path:
    """
    Call hepml.commands.train in a subprocess and return the produced metrics.json path.
    """
    cmd = [
        sys.executable,
        "-m",
        "hepml.cli",
        "train",
        "--study",
        str(default_study_directory()),
        "--mass",
        mass,
        "--splits-dir",
        str(splits_dir),
        "--outdir",
        str(outdir),
        "--seed",
        str(seed),
        "--features",
        *features,
    ]
    if drop_features:
        cmd += ["--drop-features", *drop_features]
    if class_balance:
        cmd += ["--class-balance"]

    subprocess.run(cmd, check=True)

    metrics_path = outdir / f"sig{mass}" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics: {metrics_path}")
    return metrics_path


def load_key_metrics(metrics_path: Path) -> dict[str, float]:
    m = json.loads(metrics_path.read_text())

    out = {}
    # AUCs
    out["val_auc_w"] = float(m["val"]["auc_weighted"])
    out["val_auc_u"] = float(m["val"]["auc_unweighted"])
    out["test_auc_w"] = float(m["test"]["auc_weighted"])
    out["test_auc_u"] = float(m["test"]["auc_unweighted"])

    # best Z from scan (Asimov Z in train_bdt.py)
    out["val_Z"] = float(m["val"]["weighted_significance_scan"]["best_Z"])
    out["test_Z"] = float(m["test"]["weighted_significance_scan"]["best_Z"])

    # best threshold from scan
    out["val_thr"] = float(m["val"]["weighted_significance_scan"]["best_thr"])
    out["test_thr"] = float(m["test"]["weighted_significance_scan"]["best_thr"])
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    import csv

    if not rows:
        return
    # Union of all keys across rows
    keyset = set()
    for r in rows:
        keyset.update(r.keys())

    # Put common identifiers first if present, then the rest sorted
    preferred = [
        "mode",
        "mass",
        "features",
        "dropped",
        "val_auc_w",
        "test_auc_w",
        "val_Z",
        "test_Z",
        "d_val_auc_w",
        "d_test_auc_w",
        "d_val_Z",
        "d_test_Z",
    ]
    fieldnames = [k for k in preferred if k in keyset] + sorted(keyset - set(preferred))

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # Ensure missing keys become empty cells instead of KeyError
            w.writerow({k: r.get(k, "") for k in fieldnames})


def drop_one_scan(
    *,
    mass: str,
    splits_dir: Path,
    base_out: Path,
    features: list[str],
    class_balance: bool,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Train baseline + each feature dropped once.
    """
    rows = []

    # baseline
    base_dir = base_out / "baseline"
    mp = run_train(
        mass=mass,
        outdir=base_dir,
        splits_dir=splits_dir,
        features=features,
        drop_features=[],
        class_balance=class_balance,
        seed=seed,
    )
    base_metrics = load_key_metrics(mp)
    rows.append({"variant": "baseline", "dropped": "", **base_metrics})

    # drop-one
    for f in features:
        od = base_out / f"drop_{f}"
        mp = run_train(
            mass=mass,
            outdir=od,
            splits_dir=splits_dir,
            features=features,
            drop_features=[f],
            class_balance=class_balance,
            seed=seed,
        )
        met = load_key_metrics(mp)
        rows.append({"variant": "drop1", "dropped": f, **met})

    # add deltas vs baseline (val_Z as primary to optimize significance)
    for r in rows[1:]:
        r["d_val_Z"] = r["val_Z"] - rows[0]["val_Z"]
        r["d_test_Z"] = r["test_Z"] - rows[0]["test_Z"]
        r["d_val_auc_w"] = r["val_auc_w"] - rows[0]["val_auc_w"]

    # sort: most harmful drop first (big negative delta)
    rows[1:] = sorted(rows[1:], key=lambda x: x.get("d_val_Z", 0.0))
    return rows


def greedy_backward(
    *,
    mass: str,
    splits_dir: Path,
    base_out: Path,
    features: list[str],
    class_balance: bool,
    seed: int,
    min_features: int,
) -> list[dict[str, Any]]:
    """
    Iteratively remove the feature whose removal hurts val_Z the least (or improves it).
    """
    kept = features[:]
    rows = []

    step = 0
    while len(kept) > min_features:
        # baseline at this step
        step_dir = base_out / f"step{step:02d}"
        mp0 = run_train(
            mass=mass,
            outdir=step_dir / "baseline",
            splits_dir=splits_dir,
            features=kept,
            drop_features=[],
            class_balance=class_balance,
            seed=seed,
        )
        base = load_key_metrics(mp0)

        best_choice = None
        best_delta = None
        best_metrics = None

        # try dropping each one
        for f in kept:
            mp = run_train(
                mass=mass,
                outdir=step_dir / f"try_drop_{f}",
                splits_dir=splits_dir,
                features=kept,
                drop_features=[f],
                class_balance=class_balance,
                seed=seed,
            )
            met = load_key_metrics(mp)
            delta = met["val_Z"] - base["val_Z"]  # > 0 means improvement
            if (best_delta is None) or (delta > best_delta):
                best_delta = delta
                best_choice = f
                best_metrics = met

        if best_choice is None or best_metrics is None or best_delta is None:
            raise RuntimeError("No feature could be selected for removal.")

        rows.append(
            {
                "step": step,
                "n_features_before": len(kept),
                "removed": best_choice,
                "base_val_Z": base["val_Z"],
                "new_val_Z": best_metrics["val_Z"],
                "delta_val_Z": best_delta,
                "new_val_auc_w": best_metrics["val_auc_w"],
            }
        )

        kept.remove(best_choice)
        step += 1

    return rows


# --------------------------------------
# Main
# --------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Feature ablation via repeated train_bdt runs.")
    ap.add_argument("--mass", default="200", help="Signal mass tag, e.g. 200")
    ap.add_argument(
        "--splits-dir",
        default=None,
        help="Directory containing train/val/test parquet (default: <output-root>/<study>/datasets/splits/)",
    )
    ap.add_argument("--outdir", default=None, help="Ablation output directory (default: <output-root>/ablation/)")
    ap.add_argument(
        "--mode", choices=["drop1", "greedy"], default="drop1", help="Ablation mode choice, e.g. drop1 or greedy"
    )
    ap.add_argument("--features", nargs="+", default=list(load_study(default_study_directory()).plugin.FEATURES))
    ap.add_argument("--class-balance", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-features", type=int, default=8, help="For greedy mode")
    args = ap.parse_args(argv)

    out = output_paths()
    splits_dir = Path(args.splits_dir) if args.splits_dir is not None else out.splits
    ablation_root = Path(args.outdir) if args.outdir is not None else out.ablation
    base_out = ablation_root / f"sig{args.mass}"
    base_out.mkdir(parents=True, exist_ok=True)

    if args.mode == "drop1":
        rows = drop_one_scan(
            mass=args.mass,
            splits_dir=splits_dir,
            base_out=base_out / "drop1",
            features=args.features,
            class_balance=args.class_balance,
            seed=args.seed,
        )
        out_csv = base_out / "drop1_summary.csv"
        write_csv(rows, out_csv)
        print(f"Saved: {out_csv}")
        print("Tip: sort by d_val_Z (most negative = most important feature).")

    else:
        rows = greedy_backward(
            mass=args.mass,
            splits_dir=splits_dir,
            base_out=base_out / "greedy",
            features=args.features,
            class_balance=args.class_balance,
            seed=args.seed,
            min_features=args.min_features,
        )
        out_csv = base_out / "greedy_summary.csv"
        write_csv(rows, out_csv)
        print(f"Saved: {out_csv}")
        print("Tip: if delta_val_Z stays ~0 for several removals, you can simplify safely.")


if __name__ == "__main__":
    main()
