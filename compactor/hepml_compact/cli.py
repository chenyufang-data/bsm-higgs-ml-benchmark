"""CLI composition for server-side compact extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .config import load_compact_sample, load_profile
from .export import export_sample


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", action="version", version=f"hepml-compactor {__version__}")
    ap.add_argument(
        "--extraction",
        "--study",
        dest="extraction",
        required=True,
        help="Extraction YAML file or directory containing extraction.yaml",
    )
    ap.add_argument(
        "--input", nargs="+", help="ROOT files for ONE sample/hypothesis; otherwise read the sample manifest"
    )
    ap.add_argument("--config", help="Sample manifest (default: <study>/samples.yaml)")
    ap.add_argument("--data-root", help="Override the manifest data_root")
    ap.add_argument("--sample", required=True, help="Sample key; separate keys for different coupling hypotheses")
    ap.add_argument("--kind", choices=["signal", "background"])
    ap.add_argument("--mass", type=float)
    ap.add_argument("--rho-tc", type=float)
    ap.add_argument("--xs-pb", type=float, help="Pythia merged cross section; omitted = export only, no normalization")
    ap.add_argument("--n-generated", type=int, help="Pre-skim production count; default assumes unskimmed input files")
    ap.add_argument("--outdir", required=True, help="Dedicated compact export directory for this study")
    ap.add_argument(
        "--step-size", default="50 MB", help="Decoded branch-size estimate, or integer entries; not a RAM limit"
    )
    ap.add_argument("--max-chunks", type=int, help="Process at most this many NEW chunks; rerun without it to finish")
    args = ap.parse_args(argv)
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    study = load_profile(args.extraction)
    sample = {}
    if args.input:
        if args.config:
            ap.error("Use either --input or --config")
        inputs = [Path(p) for p in args.input]
    else:
        config_path = Path(args.config) if args.config else study.directory / "samples.yaml"
        try:
            inputs, sample = load_compact_sample(config_path, args.sample, args.data_root)
        except ValueError as exc:
            ap.error(str(exc))
    options = {}
    for key in ("kind", "mass", "rho_tc", "xs_pb", "n_generated"):
        value = getattr(args, key)
        options[key] = value if value is not None else sample.get(key)
    # Provenance of a manifest rate; a rate typed on the command line has none.
    options["xs_source"] = sample.get("xs_source") if args.xs_pb is None else None
    if options["kind"] is None:
        ap.error("--kind is required when --input is used")
    step_size = int(args.step_size) if args.step_size.isdigit() else args.step_size
    export_sample(
        inputs, Path(args.outdir), study, sample=args.sample, step_size=step_size, max_chunks=args.max_chunks, **options
    )


if __name__ == "__main__":
    main()
