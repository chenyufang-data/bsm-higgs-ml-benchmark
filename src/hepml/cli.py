"""hepml - one entry point for every pipeline stage.

Usage:
  hepml <command> [args...]
  python -m hepml.cli <command> [args...]

Each command dispatches to the corresponding module's main(argv);
`hepml <command> --help` shows that stage's own options.
"""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "paths": ("hepml.commands.paths", "show resolved output locations for the selected study"),
    "compact": ("hepml.commands.compact", "stream ROOT into resumable study-specific Parquet shards"),
    "prepare": ("hepml.commands.prepare", "merge samples, weights, train/val/test splits"),
    "train": ("hepml.commands.train", "train the XGBoost BDT"),
    "shape-audit": ("hepml.commands.shape_audit", "development-only coupling shapes and conditional mass audit"),
    "evaluation-study": ("hepml.commands.evaluation_study", "weight-policy pilot and 500 fb^-1 validation report"),
    "reference-study": ("hepml.commands.reference_study", "fixed-registry mass-specific BDT and cut references"),
    "parameterized-study": ("hepml.commands.parameterized_study", "conditional BDT with shared splits and hypothesis-prior checks"),
    "comparison-study": ("hepml.commands.comparison_study", "paired mass-specific vs parameterized comparison (Phase 5)"),
    "grid-study": ("hepml.commands.grid_study", "mass-specific BDTs over the full (mass, rho_tc) grid on validation (Phase 9)"),
    "binned-fit-study": ("hepml.commands.binned_fit_study", "binned c-b mass shape-fit sensitivity on the grid (Phase 9 follow-up)"),
    "truth-tag-closure": ("hepml.commands.truth_tag_closure", "pre-registered closure of truth tagging against direct tagging"),
    "truth-tag-grid-study": ("hepml.commands.truth_tag_grid_study", "Phase 9 grid on truth-tagged rows, separate from the direct-tag grid"),
    "jet-network-study": ("hepml.commands.jet_network_study", "Phase 10a jet-level LorentzNet pilot against the grid's BDTs"),
    "benchmark-report": ("hepml.commands.benchmark_report", "method benchmark (cuts, BDT, LorentzNet, ParT) for presentation"),
    "ablation": ("hepml.commands.ablation", "drop-1 / greedy feature ablation"),
    "optimal": ("hepml.commands.optimal", "retrain with the ablation-optimal feature set"),
    "freeze": ("hepml.commands.freeze", "freeze a versioned model release"),
    "plots": ("hepml.commands.plots", "diagnostic plots for a trained model"),
    "predict": ("hepml.commands.predict", "inference with a frozen model"),
    "summarize": ("hepml.commands.summarize", "summarize inference into reports"),
}


def _usage() -> str:
    lines = ["usage: hepml <command> [args...]", "", "commands:"]
    for name, (_, desc) in COMMANDS.items():
        lines.append(f"  {name:<10} {desc}")
    lines.append("")
    lines.append("run 'hepml <command> --help' for a command's options")
    lines.append("all commands accept --study <directory> (default: studies/cg_bbc)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0

    cmd = argv[0]
    if cmd not in COMMANDS:
        print(f"unknown command: {cmd!r}\n\n{_usage()}", file=sys.stderr)
        return 2

    # All stages share the selected study's paths and analysis settings. Only
    # extraction/prepare/train need the directory again as a stage argument.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--study")
    options, remaining = common.parse_known_args(argv[1:])
    from hepml.adapters.configuration import default_study_directory, study_context
    from hepml.adapters.study_loader import load_study

    directory = default_study_directory()
    if options.study:
        selected = load_study(options.study)
        directory = selected.directory
    module_name, _ = COMMANDS[cmd]
    with study_context(directory):
        module = importlib.import_module(module_name)
        result = module.main(argv[1:] if cmd in {"compact", "prepare", "train", "shape-audit", "evaluation-study", "reference-study", "parameterized-study", "comparison-study", "grid-study", "binned-fit-study", "truth-tag-closure", "truth-tag-grid-study", "jet-network-study", "benchmark-report"} else remaining)
    return int(result) if result is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
