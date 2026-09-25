"""File-based inspection and provenance for the optional research notebooks."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from hepml_compact.config import load_profile, recipes_compatible
from hepml_compact.parquet_writer import sha256_file

from hepml.adapters.configuration import load_analysis
from hepml.adapters.dataset_files import discover_samples
from hepml.adapters.study_loader import load_study
from hepml.adapters.study_run import save_provenance, write_json
from hepml.domain.artifacts import BenchmarkFiles, dataset_filename, model_dirname
from hepml.domain.metrics import asimov_z, normalized_weights, safe_auc, sample_normalization, threshold_yields
from hepml.domain.weights import compute_sample_weight


def read_settings(path: Path) -> dict:
    """Resolve machine-local paths relative to the settings JSON, not kernel cwd."""
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("study", "signal_dir", "background_dir", "workspace"):
        config[key] = str((path.resolve().parent / config[key]).resolve())
    study = load_study(config["study"])
    config["analysis"] = asdict(load_analysis(Path(config["study"]) / "analysis.yaml"))
    config["features"] = list(study.plugin.FEATURES)
    config["study_fingerprint"] = study.fingerprint
    config.setdefault("preview_rows_per_sample", 5000)
    config.setdefault("training_args", [])
    config.setdefault("weight_mode", "none")
    if not isinstance(config["mass"], int) or config["mass"] <= 0:
        raise ValueError("mass must be a positive integer for the current benchmark CLI")
    allowed = {
        "--seed", "--max-depth", "--n-estimators", "--learning-rate", "--subsample", "--colsample-bytree",
        "--reg-lambda", "--reg-alpha", "--min-child-weight", "--gamma", "--early-stopping-rounds",
        "--tree-method", "--signif-steps", "--min-nS", "--min-nB", "--min-B", "--eps-plateau",
    }
    arguments = config["training_args"]
    if (
        not isinstance(arguments, list) or len(arguments) % 2
        or not all(isinstance(x, str) for x in arguments)
        or any(x not in allowed for x in arguments[::2])
    ):
        raise ValueError("training_args must be option/value pairs for model or scan settings, not data/path overrides")
    if config["weight_mode"] not in {"none", "balanced-mixture"}:
        raise ValueError("Unknown weight_mode")
    # IDs become path components, never paths themselves.
    for key in ("benchmark_id", "run_id"):
        value = config[key]
        if not isinstance(value, str) or not value or value in {".", ".."} or any(c in value for c in "/\\:"):
            raise ValueError(f"{key} must be a single nonempty directory name")
    return config


def inspect_inputs(config: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Validate complete exports and report full-population counts and yields."""
    study = load_study(config["study"])
    profile = load_profile(study.directory)
    lumi = float(config["analysis"]["lumi"])
    if not math.isfinite(lumi) or lumi <= 0:
        raise ValueError("Luminosity must be finite and positive, in pb^-1")
    rows, records = [], []
    seen_samples, seen_sources = set(), set()
    directories = list(dict.fromkeys([config["signal_dir"], config["background_dir"]]))
    for directory in map(Path, directories):
        if not directory.is_dir():
            raise FileNotFoundError(f"Compact directory not found: {directory}")
        # Fail on unreadable JSON; discover_samples otherwise logs and skips it.
        for path in directory.glob("*.meta.json"):
            json.loads(path.read_text(encoding="utf-8"))
        signals, backgrounds = discover_samples(directory)
        samples = [*signals.values(), *backgrounds.values()]
        if not samples:
            raise ValueError(f"No complete samples in {directory}")
        for meta in samples:
            if meta["study"] != study.name or not recipes_compatible(
                meta["object_schema"], meta["extraction_fingerprint"], profile
            ):
                raise ValueError(f"{meta['key']}: study extraction recipe selects events differently from the export")
            if meta["key"] in seen_samples:
                raise ValueError(f"Duplicate sample: {meta['key']}")
            seen_samples.add(meta["key"])
            export_path = directory / meta["export_manifest"]
            export = json.loads(export_path.read_text(encoding="utf-8"))
            for source in export["sources"]:
                identity = source["source_id"]
                if identity in seen_sources:
                    raise ValueError(f"ROOT source reused across samples: {identity}")
                seen_sources.add(identity)
            n_root = sum(source["entries"] for source in export["sources"])
            selected = sum(chunk["rows"] for chunk in export["chunks"])
            denominator, xs = float(meta["n_events_total"]), float(meta["xs_pb"])
            if not math.isfinite(denominator) or denominator <= 0 or not math.isfinite(xs) or xs <= 0:
                raise ValueError(f"{meta['key']}: invalid normalization")
            weight = compute_sample_weight(xs, denominator, lumi=lumi)
            rows.append(dict(
                sample=meta["key"], kind=meta["kind"], mass=meta["mass"], rho_tc=meta["rho_tc"],
                root_files=len(export["sources"]), n_root=n_root, n_selected=selected,
                normalization_count=denominator, xs_pb=xs, event_weight=weight,
                selected_yield=weight * selected, selection_fraction=selected / n_root if n_root else 0.0,
            ))
            records.append(dict(directory=str(directory), meta=meta, export=export,
                                export_sha256=sha256_file(export_path)))
    return pd.DataFrame(rows), records


def feature_preview(config: dict, records: list[dict]) -> pd.DataFrame:
    """Read a bounded prefix per selected sample; never interpret it as full yields."""
    limit = config["preview_rows_per_sample"]
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("preview_rows_per_sample must be a positive integer")
    study = load_study(config["study"])
    features = list(study.plugin.FEATURES)
    parts = []
    for record in records:
        meta = record["meta"]
        if meta["kind"] == "signal" and float(meta["mass"]) != float(config["mass"]):
            continue
        count = 0
        for chunk in record["export"]["chunks"]:
            if not chunk["file"] or count >= limit:
                continue
            parquet = pq.ParquetFile(Path(record["directory"]) / chunk["file"])
            for batch in parquet.iter_batches(batch_size=min(limit, 4096)):
                raw = batch.slice(0, min(batch.num_rows, limit - count)).to_pandas()
                derived = study.plugin.derive_features(raw.copy())
                if len(raw) != len(derived) or not raw.index.equals(derived.index):
                    raise ValueError("Feature adapter changed event membership or order")
                pd.testing.assert_frame_equal(raw, derived[raw.columns], check_exact=True)
                if not np.isfinite(derived[features].to_numpy(dtype=float)).all():
                    raise ValueError(f"Nonfinite features in {meta['key']}")
                part = derived[["event_id", "label", *features]].copy()
                part["sample"] = meta["key"]
                part["sample_weight"] = compute_sample_weight(
                    meta["xs_pb"], meta["n_events_total"], lumi=config["analysis"]["lumi"]
                )
                parts.append(part)
                count += len(raw)
                if count >= limit:
                    break
    if not parts:
        raise ValueError("No selected events for the requested benchmark")
    frame = pd.concat(parts, ignore_index=True)
    if frame["event_id"].duplicated().any():
        raise ValueError("Duplicate event IDs in preview")
    if set(frame["label"]) != {0, 1}:
        raise ValueError("The selected mass needs both signal and background events")
    return frame


def stage_status(directory: Path) -> dict:
    path = directory / "status.json"
    return json.loads(path.read_text()) if path.exists() else {"status": "not started"}


def verify_stage(directory: Path) -> dict:
    status = stage_status(directory)
    if status["status"] != "complete":
        raise ValueError(f"{directory}: stage is {status['status']}; inspect stage.log")
    for name, digest in status["artifacts"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Stage artifact changed: {directory / name}")
    return status


def benchmark_directory(config: dict) -> Path:
    return Path(config["workspace"]) / "datasets" / config["benchmark_id"]


def prepare_benchmark(config: dict, records: list[dict], *, wait=False) -> dict:
    directory = benchmark_directory(config)
    argv = ["prepare", "--study", config["study"], "--indir", config["signal_dir"],
            "--outdir", str(directory / "data"), "--mass", str(config["mass"]), "--write-splits"]
    if config["background_dir"] != config["signal_dir"]:
        argv += ["--background-dir", config["background_dir"]]
    return launch_stage(directory, argv, config, records, wait=wait)


def train_benchmark(config: dict, records: list[dict], *, wait=False) -> dict:
    benchmark = benchmark_directory(config)
    status = verify_stage(benchmark)
    provenance = json.loads((benchmark / "provenance.json").read_text())
    previous = provenance["settings"]
    if any(config[key] != previous[key] for key in ("features", "mass", "study_fingerprint")):
        raise ValueError("Benchmark hypothesis or feature definitions changed; prepare a new benchmark")
    old_inputs = {r["metadata"]["key"]: r["export_sha256"] for r in provenance["inputs"]}
    new_inputs = {r["meta"]["key"]: r["export_sha256"] for r in records}
    if old_inputs != new_inputs:
        raise ValueError("Compact inputs differ from the saved benchmark; select the original inputs or a new benchmark")
    if config["analysis"]["lumi"] != previous["analysis"]["lumi"]:
        raise ValueError("Luminosity differs from the saved benchmark")
    check_splits(benchmark / "data/splits", config["mass"])
    settings = dict(config, benchmark_artifacts=status["artifacts"])
    directory = Path(config["workspace"]) / "runs" / config["run_id"]
    argv = ["train", "--study", config["study"], "--mass", str(config["mass"]),
            "--splits-dir", str(benchmark / "data/splits"), "--outdir", str(directory / "models"),
            "--weight-mode", config["weight_mode"], *config["training_args"]]
    return launch_stage(directory, argv, settings, records, wait=wait)


def evaluation_weights_for_run(directory: Path, scores: pd.DataFrame) -> tuple[np.ndarray, dict, float]:
    """Re-evaluate saved scores without changing the model, threshold or artifacts."""
    settings = json.loads((directory / "provenance.json").read_text())["settings"]
    benchmark = benchmark_directory(settings)
    status = verify_stage(benchmark)
    if status["artifacts"] != settings["benchmark_artifacts"]:
        raise ValueError("Benchmark artifacts differ from the saved run")
    full = pd.read_parquet(benchmark / "data" / dataset_filename(settings["mass"]),
                           columns=["sample", "target", "sample_weight"])
    normalization = sample_normalization(full, scores)
    metrics = json.loads((directory / "models" / model_dirname(settings["mass"]) / "metrics.json").read_text())
    prepared_lumi = float(settings["analysis"]["lumi"])
    lumi = float(metrics.get("lumi_pb_inv", prepared_lumi))
    return normalized_weights(scores, normalization) * lumi / prepared_lumi, normalization, lumi


def compare_runs(directories: list[Path], *, split="val") -> pd.DataFrame:
    """Compare saved scores on one fixed benchmark at validation-selected thresholds."""
    if split not in {"train", "val", "test"}:
        raise ValueError("Comparison split must be train, val or test")
    rows, reference = [], None
    for directory in directories:
        status = verify_stage(directory)
        settings = json.loads((directory / "provenance.json").read_text())["settings"]
        identity = settings["benchmark_artifacts"]
        if reference is not None and identity != reference:
            raise ValueError("Runs use different benchmark artifacts; compare identical saved splits")
        reference = identity
        model = directory / "models" / model_dirname(settings["mass"])
        metrics = json.loads((model / "metrics.json").read_text())
        threshold = metrics["val"]["weighted_significance_scan"]["best_thr"]
        scores = pd.read_parquet(model / f"preds_{split}.parquet")
        eval_weights, normalization, lumi = evaluation_weights_for_run(directory, scores)
        y, score, weight = (scores[c].to_numpy() for c in ("target", "bdt_score", "sample_weight"))
        selected = score >= threshold
        signal = float(weight[selected & (y == 1)].sum())
        background = float(weight[selected & (y == 0)].sum())
        full = threshold_yields(y, score, eval_weights, threshold=threshold, lumi=lumi)
        rows.append(dict(
            run=directory.name, model="XGBoost", split=split,
            weight_mode=metrics["xgb"]["weight_mode"], seed=metrics["xgb"]["params"]["random_state"],
            threshold_from="validation", threshold=threshold,
            validation_threshold_valid=metrics["val"]["weighted_significance_scan"]["best_Z"] > 0,
            normalization=normalization["method"], lumi_pb_inv=lumi,
            threshold_normalization=metrics["val"].get("normalization", {}).get("method", "legacy_per_class"),
            auc_weighted=safe_auc(y, score, eval_weights), auc_unweighted=safe_auc(y, score),
            signal_yield_full=full["signal_yield"], background_yield_full=full["background_yield"],
            signal_xs_pb=full["signal_xs_pb"], background_xs_pb=full["background_xs_pb"],
            significance_full=full["significance"],
            signal_yield_in_split=signal, background_yield_in_split=background,
            significance_in_split=asimov_z(signal, background), elapsed_seconds=status["elapsed_seconds"],
        ))
    if not rows:
        raise ValueError("Select at least one completed run")
    return pd.DataFrame(rows)


def launch_stage(directory: Path, argv: list[str], config: dict, records: list[dict], *, wait=False) -> dict:
    """Run the existing CLI in a detached worker, with logs and a completion record."""
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    save_provenance(directory, config, records)
    write_json(directory / "command.json", {"argv": argv})
    write_json(directory / "status.json", {"status": "queued"})
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    with (directory / "stage.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "hepml.commands.research", str(directory)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **options,
        )
    if wait:
        code = process.wait()
        if code:
            raise RuntimeError(f"Stage failed ({code}); see {directory / 'stage.log'}")
    return {"pid": process.pid, "directory": str(directory), **stage_status(directory)}


def check_splits(directory: Path, mass: int) -> pd.DataFrame:
    """Verify saved assignment order and that every event is in exactly one split."""
    files = BenchmarkFiles(Path(directory).parent)
    assignment = pd.read_parquet(files.assignments(mass))
    if assignment["event_id"].duplicated().any() or set(assignment["split"]) != {"train", "val", "test"}:
        raise ValueError("Invalid or overlapping saved assignments")
    rows = []
    for name in ("train", "val", "test"):
        frame = pd.read_parquet(files.split(name, mass))
        expected = assignment.loc[assignment["split"] == name].sort_values("row_in_split")
        if frame["event_id"].tolist() != expected["event_id"].tolist():
            raise ValueError(f"{name}: saved event assignment differs from split")
        for target, group in frame.groupby("target"):
            rows.append(dict(split=name, target=target, events=len(group), weighted_yield=group.sample_weight.sum()))
    full = pd.read_parquet(files.dataset(mass), columns=["event_id"])
    if full.event_id.duplicated().any() or set(full.event_id) != set(assignment.event_id):
        raise ValueError("Saved splits do not partition the dataset")
    return pd.DataFrame(rows)
