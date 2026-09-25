"""Filesystem and Parquet implementation of the compact export store."""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import READABLE_EXPORT_SCHEMAS, part_glob_pattern, part_name, sample_stem


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def _export_lock(path: Path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Export is locked: {path}. If its writer crashed, remove the lock after checking it stopped."
        ) from exc
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


def combine_accumulators(chunks: list[dict]) -> dict:
    """Sum per-chunk accumulator records in chunk order (deterministic across resumes)."""
    result = {}
    for chunk in chunks:
        for name, item in (chunk.get("accumulators") or {}).items():
            entry = result.setdefault(name, {"kind": item["kind"], "events": 0, "sum": None})
            if entry["kind"] != item["kind"]:
                raise ValueError(f"{name}: inconsistent accumulator kinds between chunks")
            entry["events"] += item["events"]
            if item["sum"] is None:
                continue
            if entry["sum"] is None:
                entry["sum"] = list(item["sum"])
            elif len(entry["sum"]) != len(item["sum"]):
                raise ValueError(f"{name}: the number of weights changes between chunks")
            else:
                entry["sum"] = [a + b for a, b in zip(entry["sum"], item["sum"])]
    return result


def validate_export(directory: Path, manifest: dict, *, require_complete: bool = True) -> None:
    """Verify a transferred export without needing access to the source ROOT files."""
    if manifest.get("export_schema") not in READABLE_EXPORT_SCHEMAS:
        raise ValueError("Unsupported compact export schema")
    if require_complete and manifest.get("status") != "complete":
        raise ValueError("Compact export is incomplete; resume extraction before preparing a dataset")
    expected = iter(
        (s["source_id"], start, min(start + s["chunk_entries"], s["entries"]))
        for s in manifest["sources"]
        for start in range(0, s["entries"], s["chunk_entries"])
    )
    expected_files = set()
    stem = sample_stem(manifest["kind"], manifest["sample"])
    for index, chunk in enumerate(manifest["chunks"]):
        identity = (chunk["source_id"], chunk["entry_start"], chunk["entry_stop"])
        if identity != next(expected, None) or chunk["input_rows"] != identity[2] - identity[1]:
            raise ValueError("Export has missing, duplicated, or out-of-order entry ranges")
        if not 0 <= chunk["rows"] <= chunk["input_rows"]:
            raise ValueError("Invalid selected event count")
        filename = chunk["file"]
        if filename is None:
            if chunk["rows"] != 0:
                raise ValueError("Nonempty chunk has no Parquet file")
            continue
        if filename != part_name(stem, index):
            raise ValueError("Invalid shard path in export manifest")
        expected_files.add(filename)
        path = directory / filename
        if not path.is_file() or path.stat().st_size != chunk["bytes"] or sha256_file(path) != chunk["sha256"]:
            raise ValueError(f"Missing or corrupted shard: {path}; restore it before resuming")
        if pq.read_metadata(path).num_rows != chunk["rows"]:
            raise ValueError(f"Row-count mismatch: {path}")
    if require_complete:
        if next(expected, None) is not None:
            raise ValueError("Export marked complete has missing entry ranges")
        if {p.name for p in directory.glob(part_glob_pattern(stem))} != expected_files:
            raise ValueError("Unexpected Parquet shards in export directory")
        recorded = manifest.get("accumulators") or {}
        combined = combine_accumulators(manifest["chunks"])
        if set(recorded) != set(combined) or any(
            recorded[k]["events"] != combined[k]["events"]
            or (recorded[k]["sum"] is None) != (combined[k]["sum"] is None)
            or (combined[k]["sum"] is not None and not all(
                math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(recorded[k]["sum"], combined[k]["sum"])))
            for k in combined
        ):
            raise ValueError("Export accumulators do not match its chunks")


class ParquetExportStore:
    lock = staticmethod(_export_lock)
    write_json = staticmethod(_atomic_json)
    validate = staticmethod(validate_export)

    def initialize(self, directory: Path) -> Path:
        directory = directory.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def exists(self, path: Path) -> bool:
        return path.exists()

    def read_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def has_parts(self, directory: Path, stem: str) -> bool:
        return any(directory.glob(part_glob_pattern(stem)))

    def write_shard(self, path: Path, table: pa.Table, dtypes: dict) -> dict:
        from .extraction import arrow_type

        schema = pa.schema([(name, arrow_type(kind)) for name, kind in dtypes.items()])
        if table.column_names != list(dtypes):
            raise ValueError("Shard columns do not match the recipe's output schema")
        table = table.cast(schema)
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="zstd", compression_level=3)
        os.replace(temporary, path)
        return dict(file=path.name, bytes=path.stat().st_size, sha256=sha256_file(path))
