"""Bounded-memory Delphes ROOT reader; no selection rules."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import uproot

from .config import ExtractionProfile


def source_id_for(uuid: str, tree_name: str) -> str:
    """ROOT UUID is stable when a file is copied. Paths and sample labels
    deliberately do not enter event IDs, so copies can be detected."""
    return hashlib.sha256(f"{uuid}:{tree_name}".encode()).hexdigest()[:32]


def _source_info(path: Path, study: ExtractionProfile, step_size: str | int) -> dict:
    path = path.resolve(strict=True)
    stat = path.stat()
    tree_name = study.config.get("tree", "Delphes")
    with uproot.open(path) as root:
        tree = root[tree_name]
        available = set(tree.keys())
        missing = set(study.required_branches) - available
        if missing:
            raise ValueError(f"{path}: missing required branches {sorted(missing)}")
        branches = list(study.required_branches) + [b for b in study.optional_branches if b in available]
        uuid = str(root.file.uuid)
        source_id = source_id_for(uuid, tree_name)
        chunk_entries = (
            step_size if isinstance(step_size, int) else tree.num_entries_for(step_size, expressions=branches)
        )
        if chunk_entries <= 0:
            raise ValueError("Step size must be positive")
        return {
            "path": str(path),
            "source_id": source_id,
            "root_uuid": uuid,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "entries": int(tree.num_entries),
            "chunk_entries": int(chunk_entries),
            "branches": branches,
            "missing_optional": sorted(set(study.optional_branches) - available),
        }


class _TreeReader:
    def __init__(self, tree):
        self.tree = tree

    def read(self, branches, start, stop):
        return self.tree.arrays(branches, entry_start=start, entry_stop=stop, library="ak")


class RootEventSource:
    inspect = staticmethod(_source_info)

    @contextmanager
    def open(self, source: dict, tree_name: str):
        with ThreadPoolExecutor(max_workers=1) as decompression:
            with uproot.open(source["path"], num_workers=1, decompression_executor=decompression) as root:
                yield _TreeReader(root[tree_name])

    def assert_unchanged(self, source: dict) -> None:
        stat = Path(source["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (source["size"], source["mtime_ns"]):
            raise ValueError("Source file changed during extraction; output was not published")
