"""Frozen-model release versioning: sig{mass}_v{N} directories.

Shared by freeze_final (allocating the next version) and
predict/summarize_inference (resolving which version to load).
"""

from __future__ import annotations

import re
from pathlib import Path

from hepml.log import get_logger

log = get_logger(__name__)


def next_version(base: Path, tag: str) -> Path:
    """Next free release directory: {tag}_v1, {tag}_v2, ..."""
    pattern = re.compile(rf"{re.escape(tag)}_v(\d+)")
    versions = []
    if base.exists():
        for p in base.iterdir():
            if p.is_dir():
                m = pattern.fullmatch(p.name)
                if m:
                    versions.append(int(m.group(1)))
    v = 1 if not versions else max(versions) + 1
    return base / f"{tag}_v{v}"


def latest_final_version(final_root: Path, mass: int) -> int | None:
    """Highest existing version number for sig{mass}, or None."""
    if not final_root.exists():
        return None
    pat = re.compile(rf"^sig{mass}_v(\d+)$")
    best = None
    for p in final_root.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        v = int(m.group(1))
        best = v if best is None else max(best, v)
    return best


def resolve_model_dir(
    base_dir: Path,
    mass: int,
    ver,
    *,
    working_root: Path | None = None,
) -> Path:
    """Resolve --ver ('latest' or an integer) to a concrete model directory.

    'latest' falls back to the working dir working_root/sig{mass} (default:
    <output-root>/<study>/runs/sig{mass}) when no frozen release exists, so
    the tools stay usable before the first freeze.
    """
    if working_root is None:
        from hepml.adapters.configuration import output_paths

        working_root = output_paths().models_work
    tag_base = f"sig{mass}"
    v = str(ver).strip().lower()

    if v == "latest":
        version = latest_final_version(base_dir, mass)
        if version is None:
            model_dir = working_root / tag_base
            log.info("No frozen model found. Using working dir: %s", model_dir)
        else:
            model_dir = base_dir / f"{tag_base}_v{version}"
            log.info("Using latest frozen model: %s", model_dir)
        return model_dir

    try:
        iv = int(v)
    except ValueError as e:
        raise SystemExit(f"--ver must be an integer or 'latest' (got: {ver!r})") from e

    model_dir = base_dir / f"{tag_base}_v{iv}"
    log.info("Using specified frozen model: %s", model_dir)
    return model_dir
