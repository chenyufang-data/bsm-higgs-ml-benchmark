#!/usr/bin/env python3
"""Retrain and plot the feature set selected by validation ablation."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from hepml.adapters.configuration import default_study_directory, output_paths


# --------------------------------------
# Main
# --------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Run optimal BDT training based on ablation results.")
    ap.add_argument(
        "--csvdir",
        default=None,
        help="Directory containing the ablation summary CSVs (default: <output-root>/ablation/)",
    )
    ap.add_argument("--mass", type=int, default=200, help="Signal mass tag, e.g. 200")
    ap.add_argument(
        "--mode", choices=["drop1", "greedy"], default="drop1", help="Ablation mode choice, e.g. drop1 or greedy"
    )
    ap.add_argument(
        "--outdir", default=None, help="Final BDT train output directory (default: <output-root>/<study>/runs/)"
    )
    ap.add_argument(
        "--seed", type=int, default=None, help="Seed forwarded to the final train_bdt run (use the ablation seed)"
    )
    args = ap.parse_args(argv)

    # Define Paths
    out = output_paths()
    csv_root = Path(args.csvdir) if args.csvdir is not None else out.ablation
    csv_path = csv_root / f"sig{args.mass}" / f"{args.mode}_summary.csv"
    output_dir = Path(args.outdir) if args.outdir is not None else out.models_work
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for File
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    # Parse CSV to find optimal feature set
    df = pd.read_csv(csv_path)

    if args.mode == "greedy":
        # Identify the step with the highest new_val_Z
        best_row = df.loc[df["new_val_Z"].idxmax()]
        best_step = best_row["step"]

        # ---- guard against non-improving step-0 ablation ----
        if best_step == 0:
            if best_row["new_val_Z"] <= best_row["base_val_Z"]:
                print(
                    f"[INFO] Step-0 ablation does not improve Z: "
                    f"new={best_row['new_val_Z']:.4f}, "
                    f"base={best_row['base_val_Z']:.4f}"
                )
                print("[INFO] Using baseline feature set (no drops).")
                best_step = -1

        # Collect all features dropped up to and including that best step
        features_to_drop = df[df["step"] <= best_step]["removed"].tolist() if best_step >= 0 else []

        print(f"--- Ablation Results for sig{args.mass} ---")
        print(f"Best val_Z ({best_row['new_val_Z']:.4f}) found at step {best_step}")
        print(f"Dropping {len(features_to_drop)} features: {features_to_drop}")
    else:
        # Filter first
        positive_dz = df[df["d_val_Z"] > 0]

        if not positive_dz.empty:
            # find the row with the maximum d_val_Z
            best_row = positive_dz.loc[positive_dz["d_val_Z"].idxmax()]

            # find 'dropped'
            features_to_drop = [best_row["dropped"]]

            print(f"--- Ablation Results for sig{args.mass} ---")
            print(f"Best val_Z ({best_row['val_Z']:.4f}) found by dropping {best_row['dropped']}")
        else:
            best_row = None
            features_to_drop = []
            print(f"--- No positive d_val_Z found for sig{args.mass} ---")

    # 4. Construct Command
    # We include --outdir and the list of features to drop
    cmd = [
        sys.executable,
        "-m",
        "hepml.cli",
        "train",
        "--study",
        str(default_study_directory()),
        "--mass",
        str(args.mass),
        "--outdir",
        str(output_dir),
    ]

    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    if features_to_drop:
        cmd.append("--drop-features")
        cmd.extend(features_to_drop)

    # 5. Execute
    print(f"\nRunning: {' '.join(map(str, cmd))}\n")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining script failed with exit code {e.returncode}")
        sys.exit(e.returncode)

    # 5. Plot
    bdtplot = [
        sys.executable,
        "-m",
        "hepml.commands.plots",
        "--modeldir",
        str(output_dir),
        "--mass",
        str(args.mass),
    ]
    print(f"\nRunning: {' '.join(map(str, bdtplot))}\n")
    try:
        subprocess.run(bdtplot, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nPlot failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
