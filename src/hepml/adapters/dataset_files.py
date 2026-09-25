"""Read and verify sample metadata and prepared Parquet datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from hepml_compact.config import load_profile, recipes_compatible
from hepml_compact.contracts import READABLE_META_SCHEMAS, meta_name, part_glob_pattern, sample_stem

from hepml.adapters.study_loader import load_study
from hepml.domain.artifacts import BenchmarkFiles
from hepml.domain.dataset import _clean_df
from hepml.domain.weights import compute_sample_weight
from hepml.log import get_logger

log = get_logger(__name__)


def _read_parquet(path: Path, columns=None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path, columns=columns)


def benchmark_columns(path: Path, profile) -> list[str]:
    """Stored columns a prepared benchmark keeps.

    Every scalar event value, plus the non-truth object lists of the collections that
    the event selection and role indices use. Constituents, particle flow and
    generator truth stay in the compact export for studies that read it directly.
    """
    selected = {group["collection"] for group in profile.config["selection"].values()}
    truth = set(profile.truth_columns)
    columns = []
    for field in pq.read_schema(path):
        if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
            if field.name in truth or not any(field.name.startswith(f"{name}_") for name in selected):
                continue
        columns.append(field.name)
    return columns


def _load_meta_v3(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text())
    if meta.get("schema") not in READABLE_META_SCHEMAS:
        raise SystemExit(
            f"{meta_path} is not a current object-export meta (produced by an older pipeline). "
            "Re-run `hepml compact` to flatten this sample with merged cross-sections."
        )
    for key in ("n_events_total", "xs_pb", "label"):
        if meta.get(key) is None:
            raise SystemExit(f"{meta_path}: missing '{key}' - re-run `hepml compact`.")
    return meta


def discover_samples(indir: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Find compact samples via their meta sidecars.

    Returns ({mass: signal_meta}, {key: background_meta}); each meta dict
    gains a '_stem' entry (the file stem its parts are globbed by).
    """
    signals: dict[str, dict] = {}
    backgrounds: dict[str, dict] = {}
    study_names: set[str] = set()
    # Compact exports only publish normal sidecars once complete. Do not
    # silently omit a partially exported background from the benchmark.
    for export_path in indir.glob("*.export.json"):
        export = json.loads(export_path.read_text(encoding="utf-8"))
        if export.get("status") != "complete":
            raise ValueError(f"Incomplete compact export: {export_path}; resume `hepml compact` first")
        sidecar = indir / meta_name(sample_stem(export["kind"], export["sample"]))
        if not sidecar.is_file():
            raise ValueError(f"Missing compact sidecar: {sidecar}; finish the export or transfer")
    for meta_path in sorted(indir.glob("*.meta.json")):
        stem = meta_path.name[: -len(".meta.json")]
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Skipping unreadable meta %s (%s)", meta_path, e)
            continue
        if meta.get("schema") not in READABLE_META_SCHEMAS or "export_manifest" not in meta:
            raise ValueError(f"{meta_path}: expected a current compact export; run `hepml compact`")
        from hepml_compact.parquet_writer import validate_export

        if meta.get("normalization_status") != "ready" or meta.get("xs_pb") is None:
            raise ValueError(f"{meta_path}: missing merged cross section; rerun compact with --xs-pb")
        export_name = meta["export_manifest"]
        if Path(export_name).name != export_name or "/" in export_name or "\\" in export_name:
            raise ValueError(f"{meta_path}: invalid export manifest path")
        export = json.loads((indir / export_name).read_text(encoding="utf-8"))
        validate_export(indir, export)
        expected_meta = {
            "key": export["sample"],
            "kind": export["kind"],
            "study": export["study_name"],
            "label": int(export["kind"] == "signal"),
            "object_schema": export["object_schema"],
            "columns": export["columns"],
            "mass": export["sample_metadata"].get("mass"),
            "rho_tc": export["sample_metadata"].get("rho_tc"),
            "extraction_fingerprint": export["study"],
            "xs_pb": export["sample_metadata"]["xs_pb"],
            "n_events_total": export["sample_metadata"].get(
                "n_generated", sum(s["entries"] for s in export["sources"])
            ),
        }
        if any(meta.get(key) != value for key, value in expected_meta.items()):
            raise ValueError(f"{meta_path}: sidecar does not match its export manifest")
        meta["_stem"] = stem
        if meta.get("study"):
            study_names.add(meta["study"])
            if len(study_names) > 1:
                raise ValueError("Input directory mixes different studies; use one study per benchmark")
        if meta["kind"] == "signal":
            if meta.get("mass") is None:
                raise ValueError(f"{meta_path}: Mass-based preparation requires signal mass metadata")
            # CLI floats and manifest string masses should resolve alike.
            mass_value = float(meta["mass"])
            mass_key = str(int(mass_value)) if mass_value.is_integer() else str(mass_value)
            if mass_key in signals:
                raise ValueError(
                    f"Multiple signal hypotheses at mass {mass_key}; prepare each in a separate input directory"
                )
            signals[mass_key] = meta
        elif meta["kind"] == "background":
            backgrounds[meta.get("key", stem.removeprefix("background_"))] = meta
    return signals, backgrounds


def load_split(split_path: Path) -> pd.DataFrame:
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    df = pd.read_parquet(split_path)
    if "target" not in df.columns:
        raise KeyError(f"'target' column not found in {split_path}")
    return df


def load_sample_frame(
    indir: Path, meta: dict, sample_name: str, features: list[str], lumi: float, study=None
) -> pd.DataFrame:
    """Load one compact sample's parts and attach its uniform weight."""
    stem = meta["_stem"]
    files = sorted(indir.glob(part_glob_pattern(stem)))
    if not files:
        raise FileNotFoundError(f"No parquet parts for stem '{stem}' in {indir}")
    study = study or load_study()
    profile = load_profile(study.directory)
    # Extra or missing stored fields are fine; a different event selection is not.
    if not recipes_compatible(meta["object_schema"], meta["extraction_fingerprint"], profile):
        raise ValueError("Compact data used a different event selection; use its original study recipe")
    parts, stored = [], benchmark_columns(files[0], profile)
    for path in files:
        raw = _read_parquet(path, stored)
        derived = study.plugin.derive_features(raw.copy())
        if len(derived) != len(raw) or not derived.index.equals(raw.index):
            raise ValueError("Feature adapter changed event membership or order")
        pd.testing.assert_frame_equal(raw, derived[raw.columns], check_exact=True)
        columns = list(dict.fromkeys([*raw.columns, *study.plugin.RETAINED_COLUMNS]))
        parts.append(_clean_df(derived, features, columns))
    df = pd.concat(parts, ignore_index=True)
    if "event_id" in df and df["event_id"].duplicated().any():
        raise ValueError(f"Duplicate event IDs in {sample_name}")
    df["sample"] = sample_name
    df["xs_pb"] = float(meta["xs_pb"])
    df["n_events_total"] = float(meta["n_events_total"])
    df["sample_weight"] = compute_sample_weight(float(meta["xs_pb"]), float(meta["n_events_total"]), lumi=lumi)
    return df


def prepare_registered_reference(study, directory, meta, anchor, registry, destination, *, anchor_mass, lumi):
    """Write a new benchmark with the common registry and unchanged stored weights."""
    from hepml.application.datasets import registered_splits
    from hepml.domain.metrics import sample_normalization

    mass, features = int(meta['mass']), list(study.plugin.FEATURES)
    anchor, target = BenchmarkFiles(Path(anchor)), BenchmarkFiles(Path(destination))
    original = pd.read_parquet(anchor.dataset(anchor_mass))
    signal = load_sample_frame(directory, meta, f'sig{mass}', features, lumi, study).rename(
        columns={'label': 'target', 'weight': 'gen_weight', 'xs': 'evt_xs'})
    if len(signal) != meta['n_selected']:
        raise ValueError('Signal membership changed during feature preparation')
    # One schema for the whole benchmark: the anchor's columns, whatever else the export stores.
    missing = [column for column in original.columns if column not in signal]
    if missing:
        raise ValueError(f'Compact signal lacks anchor dataset columns: {missing}')
    signal = signal[list(original.columns)]
    background = original[original.target == 0].copy()
    # The anchor's own signal sample reuses its saved rows; any other sample, including
    # another coupling at the anchor mass, joins the anchor's background.
    anchor_sample = set(original.loc[original.target == 1, 'sample_key']) == {meta['key']}
    if anchor_sample:
        saved_signal = original[original.target == 1]
        columns = ['event_id', *features, 'sample_weight']
        pd.testing.assert_frame_equal(signal[columns].sort_values('event_id').reset_index(drop=True),
                                      saved_signal[columns].sort_values('event_id').reset_index(drop=True))
        full = original
    else:
        full = pd.concat([signal, background], ignore_index=True)
    splits = registered_splits(full, registry)
    if anchor_sample:
        # Preserve pilot row order as well as membership, so the seed-42 fit is comparable.
        for name in splits:
            saved = pd.read_parquet(anchor.split(name, mass))
            if set(saved.event_id) != set(splits[name].event_id):
                raise ValueError('Pilot split disagrees with common registry')
            splits[name] = saved
    target.splits.mkdir(parents=True, exist_ok=True)
    full.to_parquet(target.dataset(mass), index=False)
    assignments = []
    for name, part in splits.items():
        part.to_parquet(target.split(name, mass), index=False)
        assignments.append(part[['event_id', 'sample_key']].assign(split=name, row_in_split=range(len(part))))
    pd.concat(assignments, ignore_index=True).to_parquet(target.assignments(mass), index=False)
    metadata = dict(lumi=lumi, features=features, signal=meta, normalization_by_split={
        name: sample_normalization(full, part) for name, part in splits.items()},
        assignment_method='existing_common_registry', k_factors_applied_to_stored_weights=False)
    target.meta(mass).write_text(json.dumps(metadata, indent=2)+'\n')
    return full, splits
