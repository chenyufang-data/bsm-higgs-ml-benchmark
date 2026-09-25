#!/usr/bin/env python3
"""
freeze_final.py

Freeze the final optimized model by versioning the selected feature set
and decision threshold. This creates an immutable snapshot of the model
for reproducibility and deployment.

Workflow:
  - Read trained model artifacts from {workdir}/sig{MASS}/
    (default: <output-root>/<study>/runs/sig{MASS}/)
  - Assign a version tag (v1, v2, ...)
  - Write a versioned copy to {finaldir}/sig{MASS}_v{x}/
    (default: <output-root>/<study>/releases/sig{MASS}_v{x}/)

Outputs:
  {finaldir}/sig{MASS}_v{x}/
    - model.ubj
    - model.joblib
    - metrics.json
    - features.txt
    - threshold.json

Example:
  hepml freeze --mass 200
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from hepml.adapters.configuration import output_paths
from hepml.adapters.releases import next_version
from hepml.log import get_logger

log = get_logger(__name__)


def freeze_conditional(source: Path, base: Path, objective: str | None):
    """Version one shared model with separate validation operating points per hypothesis."""
    metrics = json.loads((source / 'metrics.json').read_text())
    conditioning = metrics['conditioning']
    hypotheses = {}
    for mass in conditioning['support']:
        key = str(int(mass))
        evaluation = metrics['evaluations'][key]
        if evaluation['schema_version'] != 2 or evaluation['threshold_source'] != 'validation':
            raise ValueError('Conditional releases require validation operating points')
        chosen = objective or evaluation['config']['primary_objective']
        point = evaluation['operating_points'][chosen]
        if point['status'] != 'valid':
            raise ValueError(f'm{mass}: no_valid_threshold; refusing a conditional fallback')
        hypotheses[key] = dict(schema_version=2, threshold=point['threshold'], objective=chosen,
            evaluation=evaluation, split_used='val', best_iteration=metrics['xgb']['best_iteration'])
    if not (source / 'model.ubj').is_file():
        raise FileNotFoundError('Missing shared model.ubj')
    destination = next_version(base,'conditional')
    destination.mkdir(parents=True,exist_ok=False)
    for filename in ['model.ubj','metrics.json']:
        shutil.copy2(source / filename,destination / filename)
    (destination / 'features.txt').write_text('\n'.join(metrics['features'])+'\n')
    record = dict(schema_version=2, conditioning=conditioning, hypotheses=hypotheses,
                  best_iteration=metrics['xgb']['best_iteration'], code_version=_git_commit())
    (destination / 'threshold.json').write_text(json.dumps(record,indent=2)+'\n')
    (base / 'LATEST_conditional.txt').write_text(destination.name+'\n')
    log.info('Created conditional release: %s',destination)


def _git_commit() -> str | None:
    """Current git commit hash, or None outside a repo / without git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


# --------------------------------------
# Main
# --------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Freeze a versioned model release.")
    ap.add_argument(
        "--workdir",
        default=None,
        help="directory holding the trained working model, sig{mass}/ inside (default: <output-root>/<study>/runs/)",
    )
    ap.add_argument(
        "--finaldir", default=None, help="directory for final model releases (default: <output-root>/<study>/releases/)"
    )
    ap.add_argument("--mass", type=int, default=200, help="Signal mass tag, e.g. 200")
    ap.add_argument('--conditional',action='store_true',help='workdir directly contains one shared conditional model')
    ap.add_argument(
        "--use-threshold-from", choices=["val", "test"], default="val", help="Which split provides the frozen threshold"
    )
    ap.add_argument("--objective", choices=["Z_Asimov", "Z_syst_5pct", "Z_syst_10pct"],
                    help="Version-2 primary operating point; all three are preserved")
    args = ap.parse_args(argv)
    if args.use_threshold_from != "val":
        raise ValueError("Production releases require validation-selected thresholds; test selection is forbidden")

    out = output_paths()
    base_dir = Path(args.finaldir) if args.finaldir is not None else out.models_final
    base_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir) if args.workdir is not None else out.models_work
    if args.conditional:
        freeze_conditional(workdir,base_dir,args.objective)
        return
    src_dir = workdir / f"sig{args.mass}"
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    metrics_path = src_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {src_dir}")

    metrics = json.loads(metrics_path.read_text())

    if "features" not in metrics:
        raise KeyError("metrics.json missing 'features'")
    evaluation = metrics.get("evaluation")
    if evaluation is not None:
        if evaluation.get("schema_version") != 2 or evaluation.get("threshold_source") != "validation":
            raise ValueError("Invalid operating-point schema or threshold source")
        objective = args.objective or evaluation["config"]["primary_objective"]
        point = evaluation["operating_points"][objective]
        if point["status"] != "valid":
            raise ValueError(f"{objective}: no_valid_threshold; refusing to freeze a fallback")
        scan = dict(best_thr=point["threshold"], best_Z=point["metrics"][objective],
                    best_S=point["metrics"]["S"]/evaluation["lumi_pb_inv"],
                    best_B=point["metrics"]["B"]/evaluation["lumi_pb_inv"],
                    method="exact_validation_argmax", constraints=evaluation["config"])
    else:
        if args.objective is not None:
            raise ValueError("Named systematic objectives require version-2 evaluation")
        if "weighted_significance_scan" not in metrics.get(args.use_threshold_from, {}):
            raise KeyError("Missing weighted_significance_scan in metrics")
        scan = metrics[args.use_threshold_from]["weighted_significance_scan"]
    required = ["best_thr", "best_Z", "best_S", "best_B"]
    for k in required:
        if k not in scan:
            raise KeyError(f"weighted_significance_scan missing '{k}'")

    # create versioned release dir
    tag = f"sig{args.mass}"
    final_dir = next_version(base_dir, tag)
    final_dir.mkdir(parents=True, exist_ok=False)

    # files to copy
    files_to_copy = [
        "model.ubj",
        "model.joblib",
    ]
    for fname in files_to_copy:
        src = src_dir / fname
        if src.exists():
            log.info("Copy file: %s", src.name)
            shutil.copy2(src, final_dir / fname)

    # directories to copy
    dirs_to_copy = [
        "plots",
    ]
    for dname in dirs_to_copy:
        src = src_dir / dname
        if src.exists() and src.is_dir():
            log.info("Copy dir:  %s/", dname)
            shutil.copytree(src, final_dir / dname)

    # ---- freeze features ----
    features = metrics["features"]
    (final_dir / "features.txt").write_text("\n".join(features) + "\n", encoding="utf-8")

    # ---- freeze threshold ----
    # best_iteration must travel with the model: the threshold was tuned on
    # early-stopped predict_proba scores, so inference has to slice the
    # booster to the same iteration or the scores (and threshold) are wrong.
    raw_best_it = metrics.get("xgb", {}).get("best_iteration", -1)
    best_iteration = int(raw_best_it) if raw_best_it is not None and int(raw_best_it) >= 0 else None

    threshold = {
        "threshold": float(scan["best_thr"]),
        "split_used": args.use_threshold_from,
        "best_iteration": best_iteration,
        "code_version": _git_commit(),
        "Z_Asimov": float(scan["best_Z"]),
        "S": float(scan["best_S"]),
        "B": float(scan["best_B"]),
        "method": scan.get("method", "argmax"),
        "constraints": scan.get("constraints", {}),
        "raw_argmax": scan.get(
            "raw_argmax",
            {
                "threshold": float(scan["best_thr"]),
                "Z": float(scan["best_Z"]),
            },
        ),
        "note": (
            "Threshold chosen from validation set using a robust plateau-based "
            "Asimov Z scan with minimum weighted-yield constraints."
            if args.use_threshold_from == "val"
            else "Threshold chosen from test set for reference only; validation-based threshold is preferred."
        ),
    }
    if evaluation is not None:
        threshold.update(schema_version=2, objective=objective, evaluation=evaluation,
                         significance=float(point["metrics"][objective]),
                         Z_Asimov=float(point["metrics"]["Z_Asimov"]),
                         note="Three objective-labeled validation optima preserved; plateau points remain separate.")

    (final_dir / "threshold.json").write_text(json.dumps(threshold, indent=2) + "\n", encoding="utf-8")

    # copy metrics.json for traceability
    (final_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    required = ["features.txt", "threshold.json"]
    missing = [f for f in required if not (final_dir / f).exists()]
    if missing:
        raise RuntimeError(f"Freeze incomplete; missing {missing} in {final_dir}")

    log.info("Created release: %s", final_dir)
    log.info("Frozen features  -> %s", final_dir / "features.txt")
    log.info("Frozen threshold -> %s", final_dir / "threshold.json")

    latest_txt = base_dir / f"LATEST_sig{args.mass}.txt"
    latest_txt.write_text(f"{final_dir.name}\n", encoding="utf-8")
    log.info("Wrote latest version pointer: %s", latest_txt)

    try:
        link_ptr = base_dir / f"sig{args.mass}"

        # Remove old pointer dir or symlink
        if os.path.lexists(link_ptr):
            if link_ptr.is_symlink() or link_ptr.is_file():
                link_ptr.unlink()
            elif link_ptr.is_dir():
                shutil.rmtree(link_ptr)

        # Create symlink
        rel_target = os.path.relpath(final_dir, start=link_ptr.parent)
        link_ptr.symlink_to(rel_target, target_is_directory=True)
        log.info("Updated symlink pointer: %s -> %s", link_ptr, rel_target)
    except OSError as e:
        log.info("Symlink not created (this is OK on Windows): %s", e)
        log.info("Use LATEST_sig%s.txt to resolve the latest version instead.", args.mass)


if __name__ == "__main__":
    main()
