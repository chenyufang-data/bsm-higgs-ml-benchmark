"""Resumable extraction policy, independent of ROOT and Parquet implementations."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .config import ExtractionProfile
from .contracts import EXPORT_SCHEMA, META_SCHEMA, meta_name, part_name, sample_stem
from .parquet_writer import ParquetExportStore, combine_accumulators
from .provenance import extraction_provenance
from .root_reader import RootEventSource

log = logging.getLogger(__name__)


def export_sample(
    inputs: list[Path],
    outdir: Path,
    study: ExtractionProfile,
    *,
    sample: str,
    kind: str,
    mass: float | None = None,
    rho_tc: float | None = None,
    xs_pb: float | None = None,
    n_generated: int | None = None,
    xs_source: dict | None = None,
    source_reader=None,
    store=None,
    provenance: dict | None = None,
    step_size: str | int = "50 MB",
    max_chunks: int | None = None,
) -> dict:
    """Export or resume one sample. max_chunks bounds NEW chunks for a pilot."""
    source_reader = source_reader or RootEventSource()
    store = store or ParquetExportStore()
    provenance = provenance or extraction_provenance()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", sample) or kind not in {"signal", "background"}:
        raise ValueError("Expected a safe sample name and kind signal/background")
    if not inputs:
        raise ValueError("At least one ROOT input is required")
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    if xs_pb is not None and (not math.isfinite(xs_pb) or xs_pb <= 0):
        raise ValueError("Merged xs_pb must be finite and positive")
    if xs_source is not None and not isinstance(xs_source, dict):
        raise ValueError("xs_source must be a mapping recorded by the cross-section helper")
    for name, value in (("mass", mass), ("rho_tc", rho_tc)):
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if kind == "signal" and mass is not None and mass <= 0:
        raise ValueError("Signal mass must be positive")
    sources = [source_reader.inspect(Path(p), study, step_size) for p in inputs]
    if len({s["source_id"] for s in sources}) != len(sources):
        raise ValueError("Duplicate ROOT source identity; do not export the same events twice")
    total_entries = sum(s["entries"] for s in sources)
    if n_generated is not None and (n_generated < total_entries or n_generated <= 0):
        raise ValueError("n_generated must be positive and at least the number of input entries")
    # Extraction is independent of physics normalization. Missing normalization
    # can be supplied later by rerunning with xs_pb, without reading event data.
    signature_payload = {
        "export_schema": EXPORT_SCHEMA,
        "study": study.fingerprint,
        "extractor": provenance["extractor"],
        "libraries": provenance["libraries"],
        "sample": sample,
        "kind": kind,
        "sources": [{k: v for k, v in s.items() if k != "path"} for s in sources],
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    outdir = store.initialize(Path(outdir))
    stem = sample_stem(kind, sample)
    manifest_path = outdir / f"{stem}.export.json"
    started = time.monotonic()
    with store.lock(outdir / f"{stem}.lock"):
        if store.exists(manifest_path):
            manifest = store.read_json(manifest_path)
            if manifest.get("signature") != signature:
                raise ValueError(
                    "Inputs, study, extraction code, library versions or chunk settings changed; "
                    "use a new output directory"
                )
            store.validate(outdir, manifest, require_complete=False)
        else:
            if store.has_parts(outdir, stem) or store.exists(outdir / meta_name(stem)):
                raise ValueError("Output already contains unmanaged sample files; choose a new output directory")
            manifest = {
                **signature_payload,
                "signature": signature,
                "study_name": study.name,
                "study_config": study.config,
                "compactor": {key: provenance.get(key) for key in ("version", "python", "bundle")},
                "sources": sources,
                "status": "incomplete",
                "chunks": [],
                "identity_scheme": "sha256(ROOT_UUID:tree)[:32]:source_entry; not a full source-content hash",
                "object_schema": study.config,
                "columns": study.output_dtypes,
                "truth_columns": study.truth_columns,
                "cutflow": {},
                "extraction_seconds": 0.0,
            }
            store.write_json(manifest_path, manifest)
        # Metadata edits are intentional and recorded; never erase previously
        # supplied normalization/hypothesis information when flags are omitted.
        metadata = manifest.get("sample_metadata", {})
        supplied = {"mass": mass, "rho_tc": rho_tc, "xs_pb": xs_pb, "n_generated": n_generated, "xs_source": xs_source}
        for key, value in supplied.items():
            if value is not None:
                if metadata.get(key) is not None and metadata[key] != value:
                    raise ValueError(f"Existing {key} differs; use a new output directory for a different hypothesis")
                metadata[key] = value
        manifest["sample_metadata"] = metadata
        store.write_json(manifest_path, manifest)
        previous = {(c["source_id"], c["entry_start"]): c for c in manifest["chunks"]}
        new_chunks = 0
        paused = False
        label = int(kind == "signal")
        identity = {name: dtype for name, dtype in study.output_dtypes.items()
                    if name in {"source_entry", "source_id", "event_id", "sample_key"}}
        for source in sources:
            with source_reader.open(source, study.config.get("tree", "Delphes")) as reader:
                for start in range(0, source["entries"], source["chunk_entries"]):
                    stop = min(start + source["chunk_entries"], source["entries"])
                    prior = previous.get((source["source_id"], start))
                    if prior is not None:
                        if prior["entry_stop"] != stop:
                            raise ValueError("Resume manifest has inconsistent entry ranges")
                        continue
                    if max_chunks is not None and new_chunks >= max_chunks:
                        paused = True
                        break
                    arrays = reader.read(source["branches"], start, stop)
                    selected = study.process_chunk(arrays, label)
                    table = selected.table
                    offsets = np.asarray(selected.offsets)
                    if (
                        offsets.ndim != 1
                        or table.num_rows != len(offsets)
                        or offsets.dtype.kind not in "iu"
                        or np.any(offsets < 0)
                        or np.any(offsets >= stop - start)
                        or np.any(np.diff(offsets.astype(np.int64)) <= 0)
                    ):
                        raise ValueError("Study must return unique, ordered input offsets aligned to its rows")
                    if table.num_rows and not pc.all(pc.equal(table["label"], label)).as_py():
                        raise ValueError("Study changed sample labels")
                    entries = offsets.astype(np.int64) + start
                    added = {
                        "source_entry": pa.array(entries, type=pa.int64()),
                        "source_id": pa.array([source["source_id"]] * len(entries), type=pa.string()),
                        "event_id": pa.array([f"{source['source_id']}:{entry}" for entry in entries], type=pa.string()),
                        "sample_key": pa.array([sample] * len(entries), type=pa.string()),
                    }
                    for name in identity:
                        table = table.append_column(name, added[name])
                    chunk = {
                        "source_id": source["source_id"],
                        "entry_start": start,
                        "entry_stop": stop,
                        "input_rows": stop - start,
                        "rows": table.num_rows,
                        "file": None,
                    }
                    if selected.accumulators:
                        chunk["accumulators"] = selected.accumulators
                    if table.num_rows:
                        shard = outdir / part_name(stem, len(manifest["chunks"]))
                        chunk.update(store.write_shard(shard, table, study.output_dtypes))
                    for key, count in selected.cutflow.items():
                        if not isinstance(count, (int, np.integer)) or count < 0:
                            raise ValueError("Study cutflow counts must be nonnegative integers")
                        manifest["cutflow"][key] = manifest["cutflow"].get(key, 0) + int(count)
                    manifest["chunks"].append(chunk)
                    store.write_json(manifest_path, manifest)
                    new_chunks += 1
                    log.info("%s: entries %d-%d / %d, selected %d", sample, start, stop, source["entries"], table.num_rows)
                if paused:
                    break
        # Detect modification of an input during extraction before publishing.
        for source in sources:
            source_reader.assert_unchanged(source)
        manifest["processed_entries"] = sum(c["input_rows"] for c in manifest["chunks"])
        manifest["selected_events"] = sum(c["rows"] for c in manifest["chunks"])
        manifest["status"] = "complete" if manifest["processed_entries"] == total_entries else "incomplete"
        manifest["extraction_seconds"] += time.monotonic() - started
        if manifest["status"] == "complete":
            manifest["accumulators"] = combine_accumulators(manifest["chunks"])
            store.validate(outdir, manifest)
        store.write_json(manifest_path, manifest)
        if manifest["status"] == "complete":
            known_xs = metadata.get("xs_pb")
            meta = {
                "schema": META_SCHEMA,
                "key": sample,
                "kind": kind,
                "mass": metadata.get("mass"),
                "label": label,
                "study": study.name,
                "extraction_fingerprint": study.fingerprint,
                "object_schema": study.config,
                "columns": study.output_dtypes,
                "truth_columns": study.truth_columns,
                "export_manifest": manifest_path.name,
                "export_status": "complete",
                "compactor_version": provenance.get("version"),
                "normalization_status": "ready" if known_xs is not None else "missing_cross_section",
                "normalization_policy": "merged_xs_uniform_v2",
                "xs_pb": known_xs,
                "xs_source": metadata.get("xs_source"),
                "n_events_total": metadata.get("n_generated", total_entries),
                "denominator_source": "explicit_production_count"
                if "n_generated" in metadata
                else "input_entries_assumed_unskimmed",
                "n_selected": manifest["selected_events"],
                "rho_tc": metadata.get("rho_tc"),
                "xs_by_rho": {str(metadata["rho_tc"]): [known_xs, None]}
                if metadata.get("rho_tc") is not None and known_xs is not None
                else None,
                "accumulators": manifest["accumulators"],
                "runs": sources,
                "cuts": study.config.get("selection", {}),
                "cutflow": manifest["cutflow"],
            }
            store.write_json(outdir / meta_name(stem), meta)
        log.info(
            "Export %s: %d/%d events selected; manifest %s",
            manifest["status"],
            manifest["selected_events"],
            manifest["processed_entries"],
            manifest_path,
        )
        return manifest
