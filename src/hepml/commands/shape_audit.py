"""Run the preregistered development-only coupling audit and conditional expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from hepml_compact.parquet_writer import sha256_file

from hepml.adapters.configuration import default_study_directory, load_analysis
from hepml.adapters.dataset_files import discover_samples, load_sample_frame
from hepml.adapters.research import check_splits
from hepml.adapters.shape_audit import joint_diagnostic, plot_shapes
from hepml.adapters.study_loader import load_study
from hepml.adapters.study_run import save_provenance, write_json
from hepml.application.shape_audit import (
    audit_decision,
    bootstrap_counts,
    distance_table,
    efficiency_tables,
    followup_reasons,
)
from hepml.domain.artifacts import BenchmarkFiles
from hepml.domain.splits import extend_registry


def load_anchor(directory, mass):
    files = BenchmarkFiles(Path(directory))
    check_splits(files.splits, mass)
    events = pd.read_parquet(files.dataset(mass), columns=["event_id", "sample_key"])
    assignments = pd.read_parquet(files.assignments(mass))
    result = events.merge(assignments[["event_id", "split"]], on="event_id", validate="one_to_one")
    result["assignment_method"] = "preserved_pilot_anchor"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=None)
    parser.add_argument("--config", help="Default: <study>/validation.yaml")
    parser.add_argument("--compact-root", required=True)
    parser.add_argument("--anchor-dataset", required=True, help="Pilot data/ directory; assignments are preserved")
    parser.add_argument("--anchor-mass", type=int, default=200)
    parser.add_argument("--registry-dir", required=True, help="New versioned assignment registry directory")
    parser.add_argument("--outdir", required=True, help="New audit report directory; never overwrite a prior run")
    parser.add_argument("--production-info", help="Optional local YAML with samples and production review flags")
    args = parser.parse_args(argv)
    study = load_study(args.study or default_study_directory())
    config_path = Path(args.config) if args.config else study.directory / "validation.yaml"
    config = yaml.safe_load(config_path.read_text())
    if (len(set(config["initial_masses"])) != len(config["initial_masses"])
            or set(config["initial_masses"]) & set(config["conditional_masses"])
            or sorted(set(config["couplings"])) != config["couplings"]
            or len(config["couplings"]) < 2 or config["bootstrap_replicates"] < 10
            or not 0 < config["stage_alpha"] < 1):
        raise ValueError("Invalid audit masses, couplings, bootstrap count or confidence level")
    features = list(study.plugin.FEATURES)
    if not set(config["reference_cut_features"]).issubset(features):
        raise ValueError("Reference cut features must be study observables")
    output, registry_dir = Path(args.outdir).resolve(), Path(args.registry_dir).resolve()
    if output.exists() or registry_dir.exists():
        raise FileExistsError("Use new audit and registry directories; completed or partial runs are immutable")
    registry = load_anchor(Path(args.anchor_dataset), args.anchor_mass)
    output.mkdir(parents=True)
    registry_dir.mkdir(parents=True)
    production = yaml.safe_load(Path(args.production_info).read_text()) if args.production_info else {}
    settings = dict(vars(args), validation=config)
    settings["study"] = str(study.directory.resolve())
    write_json(output / "preregistration.json", settings)
    write_json(output / "status.json", {"status": "running"})
    records, all_inventory, stages, table_sets, widths = [], [], [], [], []
    metadata_by_rho = {}
    root = Path(args.compact_root)
    for rho in config["couplings"]:
        directory = root / config["coupling_directories"][str(rho)]
        signals, _ = discover_samples(directory)
        metadata_by_rho[rho] = (directory, signals)
    source_ids = {}
    anchor_files = list(Path(args.anchor_dataset).glob("*.parquet")) + list((Path(args.anchor_dataset) / "splits").glob("*"))
    anchor_hashes = {str(path.resolve()): sha256_file(path) for path in anchor_files if path.is_file()}

    def run_stage(masses, stage):
        nonlocal registry
        development, inventory, counts = {}, {}, {}
        for mass in masses:
            for rho in config["couplings"]:
                directory, signals = metadata_by_rho[rho]
                meta = signals[str(mass)]
                if not np.isclose(float(meta["rho_tc"]), rho):
                    raise ValueError("Directory rho does not match export metadata")
                export_path = directory / meta["export_manifest"]
                export = json.loads(export_path.read_text())
                for source in export["sources"]:
                    identity = source["root_uuid"]
                    if identity in source_ids and source_ids[identity] != meta["key"]:
                        raise ValueError("Different signal samples share a ROOT UUID; check production lineage")
                    source_ids[identity] = meta["key"]
                frame = load_sample_frame(directory, meta, meta["key"], features,
                                          load_analysis(study.directory / "analysis.yaml").lumi, study)
                if len(frame) != meta["n_selected"] or set(frame.sample_key) != {meta["key"]}:
                    raise ValueError("Sample identity/count differs from compact metadata")
                registry = extend_registry(registry, frame[["event_id", "sample_key"]], seed=config["seed"])
                selected = registry.loc[(registry.sample_key == meta["key"]) & (registry.split == "train"), "event_id"]
                dev = frame.loc[frame.event_id.isin(selected), ["event_id", *features]].copy()
                dev = dev.sort_values("event_id").reset_index(drop=True)
                dev["rho_tc"] = rho
                development[mass, rho] = dev
                row = dict(mass=mass, rho=rho, sample=meta["key"], n_selected=len(frame), n_development=len(dev),
                           n_root=sum(source["entries"] for source in export["sources"]), xs_pb=meta["xs_pb"])
                inventory[mass, rho] = row
                all_inventory.append(row)
                rng = np.random.default_rng(np.random.SeedSequence([config["seed"], int(mass), round(rho * 10000)]))
                counts[mass, rho] = bootstrap_counts(len(dev), config["bootstrap_replicates"], rng)
                records.append(dict(directory=str(directory.resolve()), meta=meta, export=export,
                                    export_sha256=sha256_file(export_path)))
                info = production.get("samples", {}).get(meta["key"], {})
                width = info.get("width_GeV")
                widths.append(dict(sample=meta["key"], mass=mass, rho=rho, width_GeV=width,
                                   width_over_mass=float(width) / mass if width is not None else None,
                                   branching_ratio=info.get("branching_ratio"),
                                   status="provided" if width is not None else "unknown"))
        # Freeze the registry and input/source records before evaluating shapes.
        registry.to_parquet(registry_dir / "assignments.parquet", index=False)
        save_provenance(output, settings, records)
        write_json(output / "anchor_hashes.json", anchor_hashes)
        print(f"{stage}: {len(development)} samples; all feature comparisons use train only", flush=True)
        distances = distance_table(development, features, config, counts)
        efficiencies, efficiency_comparisons = efficiency_tables(development, inventory, config)
        print(f"{stage}: marginal shapes complete; checking joint distributions", flush=True)
        joint = []
        for mass in masses:
            joint.append(joint_diagnostic(development[mass, config["couplings"][0]],
                                           development[mass, config["couplings"][-1]], features, config,
                                           mass=mass, directory=output, family_size=len(masses)))
            print(f"{stage}: m{mass} cross-validated coupling diagnostic complete", flush=True)
        joint = pd.DataFrame(joint)
        reasons = followup_reasons(distances, efficiency_comparisons, joint, config)
        for table, name in ((distances, "shape_distances"), (efficiencies, "acceptance"),
                            (efficiency_comparisons, "efficiency_comparisons"), (joint, "joint_diagnostics")):
            table.to_csv(output / f"{name}_{stage}.csv", index=False)
        stages.append(dict(stage=stage, masses=masses, followup_reasons=reasons))
        table_sets.append((distances, efficiency_comparisons, joint))
        plot_shapes(development, counts, features, config, output, stage=stage)
        return reasons

    try:
        reasons = run_stage(config["initial_masses"], "initial")
        if reasons and config["conditional_masses"]:
            print("Follow-up triggered: " + "; ".join(reasons), flush=True)
            run_stage(config["conditional_masses"], "conditional")
        combined = [pd.concat([tables[i] for tables in table_sets], ignore_index=True) for i in range(3)]
        production_complete = (production.get("comparability_reviewed") is True
                               and production.get("independent_events_confirmed") is True
                               and all(row["status"] == "provided" for row in widths))
        decision = audit_decision(*combined, production_complete=production_complete, config=config)
        pd.DataFrame(all_inventory).to_csv(output / "inventory.csv", index=False)
        pd.DataFrame(widths).to_csv(output / "widths.csv", index=False)
        registry.groupby(["sample_key", "split"]).size().rename("events").to_csv(output / "split_counts.csv")
        for name, digest in anchor_hashes.items():
            if sha256_file(Path(name)) != digest:
                raise ValueError("Pilot artifacts changed during the audit")
        provenance = json.loads((output / "provenance.json").read_text())
        for name, record in provenance["sources"].items():
            if sha256_file(Path(name)) != record["sha256"]:
                raise ValueError("Source changed during the audit")
        production_record = {"information": production, "cards": {}}
        if args.production_info:
            production_record["input_sha256"] = sha256_file(Path(args.production_info))
            for sample, info in production.get("samples", {}).items():
                for card in info.get("cards", []):
                    path = (Path(args.production_info).resolve().parent / card).resolve()
                    production_record["cards"][f"{sample}:{path.name}"] = {"path": str(path), "sha256": sha256_file(path)}
        write_json(output / "production.json", production_record)
        metadata = dict(status="complete", method="anchored_hash_rank_v1", seed=config["seed"],
                        anchor_hashes=anchor_hashes, assignments_sha256=sha256_file(registry_dir / "assignments.parquet"),
                        samples=sorted(registry.sample_key.unique()), audit=str(output))
        write_json(registry_dir / "registry.json", metadata)
        result = dict(decision=decision, stages=stages, production_complete=production_complete,
                      registry=str(registry_dir), rows=len(registry), test_features_used=False)
        write_json(output / "decision.json", result)
        report = ["# Phase 1 coupling audit", "", f"Decision: **{decision}**.", "",
                  f"Initial masses: {config['initial_masses']}; couplings: {config['couplings']}.",
                  f"Conditional expansion performed: {len(stages) > 1}.", "",
                  "All shape/correlation/classifier checks used the training partition only.",
                  "ROOT acceptance uses aggregate compact counts, not test kinematics or scores.",
                  "Pilot signal/background assignments and physical weights were preserved.", ""]
        for stage in stages:
            report += [f"## {stage['stage'].title()} masses {stage['masses']}", "",
                       *[f"- {reason}" for reason in stage["followup_reasons"]], ""]
        report += ["## Interpretation and limitations", "",
                   "A shape-relevant result is evidence of rho-associated differences requiring follow-up;",
                   "it does not establish causation or prove that rho conditioning improves classification.",
                   "An inconclusive result does not justify rate-only rescaling or a full-grid training run.",
                   "Production widths, matching/detector settings and independent event lineage must be reviewed.",
                   f"Production review complete: {production_complete}. Missing widths remain unknown.", "",
                   "KS intervals use conservative simultaneous DKW bounds (valid with ties).",
                   "W1/IQR intervals use an approximate simultaneous event bootstrap per stage, with a fixed",
                   "rho01 development IQR. Confidence budgets apply separately to each diagnostic family.",
                   "Histogram/ratio bands are pointwise 95%; tails are folded into edge bins and recorded.",
                   "Efficiency intervals are simultaneous Clopper-Pearson intervals; zero denominators are undefined.",
                   "Two-sample OOF AUC bands condition on fitted scores and omit training uncertainty;",
                   "permutation controls refit every fold. They are supporting diagnostics, not proof of equivalence.",
                   "All uncertainty calculations assume independent generated events; ROOT UUIDs alone cannot prove this.", "",
                   "Review shape_overlays_initial.pdf, shape_ratios_initial.pdf, shape_distances_initial.csv,",
                   "efficiency_comparisons_initial.csv, joint_diagnostics_initial.csv and widths.csv.",
                   "Conditional artifacts use the same filenames with the conditional suffix."]
        (output / "rho_decision.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        write_json(output / "status.json", dict(status="complete", **result))
        print(f"Decision: {decision}. Report: {output / 'rho_decision.md'}", flush=True)
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
