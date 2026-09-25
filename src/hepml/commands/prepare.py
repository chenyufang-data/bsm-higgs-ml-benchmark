#!/usr/bin/env python3
"""Prepare weighted datasets and persist deterministic event splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hepml.adapters.configuration import load_analysis, output_paths
from hepml.adapters.dataset_files import _load_meta_v3, discover_samples, load_sample_frame
from hepml.adapters.study_loader import load_study
from hepml.application.datasets import assemble_dataset
from hepml.domain.artifacts import (
    assignments_filename,
    dataset_filename,
    split_filename,
    split_meta_filename,
)
from hepml.domain.dataset import _stratified_split
from hepml.domain.metrics import sample_normalization, split_fracs_weighted
from hepml.log import get_logger

log = get_logger(__name__)


def build_one_dataset(
    indir: Path,
    outdir: Path,
    sig_meta: dict,
    bkg_metas: dict[str, dict],
    features: list[str],
    lumi: float,
    study=None,
    background_dir: Path | None = None,
) -> pd.DataFrame:
    sig_mass = str(sig_meta["mass"])
    frames = [load_sample_frame(indir, sig_meta, f"sig_{sig_mass}", features, lumi, study)]
    for key, meta in bkg_metas.items():
        frames.append(load_sample_frame(background_dir or indir, meta, f"bkg_{key}", features, lumi, study))

    df, report = assemble_dataset(frames, sig_meta, features, lumi)

    # ---- SAVE PER-SAMPLE REPORT ----
    sanity_dir = outdir / "sanity_reports" / f"sig{sig_mass}"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    (sanity_dir / "sanity_prepared.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return df


# --------------------------------------
# Main
# --------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Merge compact samples, attach physics weights, write splits.")
    ap.add_argument(
        "--indir",
        default=None,
        help="Directory with compact parts + meta sidecars (default: <output-root>/<study>/compact/)",
    )
    ap.add_argument(
        "--outdir", default=None, help="Where to write combined ML datasets (default: <output-root>/<study>/datasets/)"
    )
    ap.add_argument(
        "--analysis-config",
        default=None,
        help="Analysis config YAML (default: studies/cg_bbc/analysis.yaml if present)",
    )
    ap.add_argument(
        "--study", default=None, help="Study directory: select its features and namespace default outputs by study name"
    )
    ap.add_argument("--background-dir", type=Path, help="Separate compact background directory (optional)")
    ap.add_argument("--mass", nargs="+", default=None, help="Signal mass tag(s) (default: every compact signal mass)")
    ap.add_argument(
        "--backgrounds",
        nargs="+",
        default=None,
        help="Background keys to include (default: every compact background)",
    )
    ap.add_argument("--features", nargs="+", default=None, help="Feature columns (default: selected study's features)")
    ap.add_argument("--extra-features", nargs="+", default=[], help="Extra features to append to base list")
    ap.add_argument("--write-splits", action="store_true", help="Also write train/val/test splits for each dataset")
    ap.add_argument("--test-size", type=float, default=None, help="Test fraction of the full dataset")
    ap.add_argument("--val-size", type=float, default=None, help="Validation fraction of the full dataset")
    ap.add_argument("--seed", type=int, default=None, help="Split shuffle seed")
    ap.add_argument("--lumi", type=float, default=None, help="Integrated luminosity (inverse unit of the manifest xs)")

    args = ap.parse_args(argv)

    cfg = load_analysis(
        args.analysis_config
        or (
            str(Path(args.study) / "analysis.yaml")
            if args.study and (Path(args.study) / "analysis.yaml").exists()
            else None
        )
    )
    study = load_study(args.study)
    out = output_paths(study=study.name)
    lumi = args.lumi if args.lumi is not None else cfg.lumi
    test_size = args.test_size if args.test_size is not None else cfg.splits.test_size
    val_size = args.val_size if args.val_size is not None else cfg.splits.val_size
    seed = args.seed if args.seed is not None else cfg.splits.seed

    indir = Path(args.indir) if args.indir is not None else out.root_parquet
    outdir = Path(args.outdir) if args.outdir is not None else out.datasets
    outdir.mkdir(parents=True, exist_ok=True)
    default_features = list(study.plugin.FEATURES)
    features = (args.features if args.features is not None else default_features) + args.extra_features

    signals, backgrounds = discover_samples(indir)
    if args.background_dir is not None:
        if backgrounds:
            raise ValueError("Use backgrounds in --indir OR --background-dir, not both")
        extra_signals, backgrounds = discover_samples(args.background_dir)
        if extra_signals:
            raise ValueError("--background-dir must contain backgrounds only")
    if study and any(m.get("study", study.name) != study.name for m in [*signals.values(), *backgrounds.values()]):
        raise ValueError("Selected study does not match input metadata")
    if not signals:
        raise SystemExit(f"No compact signal samples found in {indir} - run `hepml compact` first.")
    if not backgrounds:
        raise SystemExit(f"No compact background samples found in {indir} - run `hepml compact` first.")

    masses = [str(m) for m in args.mass] if args.mass is not None else sorted(signals)
    if args.backgrounds is not None:
        unknown = [b for b in args.backgrounds if b not in backgrounds]
        if unknown:
            raise SystemExit(f"Unknown background(s) {unknown}; compact: {sorted(backgrounds)}")
        backgrounds = {k: backgrounds[k] for k in args.backgrounds}
    log.info("Backgrounds: %s", sorted(backgrounds))

    for sig_mass in masses:
        if sig_mass not in signals:
            raise SystemExit(f"No compact signal for mass {sig_mass}; available: {sorted(signals)}")
        sig_meta = _load_meta_v3(indir / f"{signals[sig_mass]['_stem']}.meta.json")
        sig_meta["mass"] = sig_mass
        sig_meta["_stem"] = signals[sig_mass]["_stem"]

        df = build_one_dataset(indir, outdir, sig_meta, backgrounds, features, lumi, study, args.background_dir)

        out_all = outdir / dataset_filename(sig_mass)
        df.to_parquet(out_all, index=False)
        log.info("Saved: %s", out_all)

        if args.write_splits:
            train, val, test = _stratified_split(df, test_size, val_size, seed)
            (outdir / "splits").mkdir(parents=True, exist_ok=True)

            train_path = outdir / "splits" / split_filename("train", sig_mass)
            val_path = outdir / "splits" / split_filename("val", sig_mass)
            test_path = outdir / "splits" / split_filename("test", sig_mass)

            train.to_parquet(train_path, index=False)
            val.to_parquet(val_path, index=False)
            test.to_parquet(test_path, index=False)

            counts = {
                "n_all": int(len(df)),
                "n_train": int(len(train)),
                "n_val": int(len(val)),
                "n_test": int(len(test)),
                "frac_train": float(len(train) / len(df)),
                "frac_val": float(len(val) / len(df)),
                "frac_test": float(len(test) / len(df)),
            }

            split_meta = {
                "test_size_arg": test_size,
                "val_size_arg": val_size,
                "seed": seed,
                "lumi": float(lumi),
                "features": features,
                "object_schema": sig_meta["object_schema"],
                "extraction_fingerprint": sig_meta["extraction_fingerprint"],
                "signal": {
                    "mass": sig_mass,
                    "xs_pb": float(sig_meta["xs_pb"]),
                    "n_events_total": int(sig_meta["n_events_total"]),
                    "xs_by_rho": sig_meta.get("xs_by_rho"),
                },
                "backgrounds": {
                    k: {"xs_pb": float(m["xs_pb"]), "n_events_total": int(m["n_events_total"])}
                    for k, m in backgrounds.items()
                },
                "counts": counts,
                "normalization_by_split": {
                    name: sample_normalization(df, part)
                    for name, part in (("train", train), ("val", val), ("test", test))
                },
                "weighted_frac_train": split_fracs_weighted(df, train),
                "weighted_frac_val": split_fracs_weighted(df, val),
                "weighted_frac_test": split_fracs_weighted(df, test),
            }
            if study:
                split_meta["study"] = study.name
                split_meta["study_fingerprint"] = study.fingerprint
            if "event_id" in df:
                assignment_name = assignments_filename(sig_mass)
                assignments = []
                for split_name, part in (("train", train), ("val", val), ("test", test)):
                    assignment = part[["event_id"]].copy()
                    assignment["split"] = split_name
                    assignment["row_in_split"] = np.arange(len(part), dtype=np.int64)
                    assignments.append(assignment)
                pd.concat(assignments, ignore_index=True).to_parquet(outdir / "splits" / assignment_name, index=False)
                split_meta["assignments_file"] = assignment_name
            (outdir / "splits" / split_meta_filename(sig_mass)).write_text(
                json.dumps(split_meta, indent=2) + "\n", encoding="utf-8"
            )

            log.info("Saved splits: %s, %s, %s", train_path, val_path, test_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
