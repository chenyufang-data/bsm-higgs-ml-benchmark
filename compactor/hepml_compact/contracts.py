"""File-naming and schema contracts between compact export, preparation and model stages."""

from __future__ import annotations

# Export manifest (.export.json) and sidecar metadata (.meta.json) versions.
# New exports use the latest; readers accept every listed version.
EXPORT_SCHEMA = 3
READABLE_EXPORT_SCHEMAS = (2, 3)
META_SCHEMA = 4
READABLE_META_SCHEMAS = (3, 4)


# Compact shards and metadata
def sample_stem(kind: str, key: str) -> str:
    """kind is 'signal' or 'background'; key is the path-config key (e.g. 'sig_200', 'bbj')."""
    return f"{kind}_{key}"


def part_name(stem: str, index: int) -> str:
    return f"{stem}.part{index:05d}.parquet"


def part_glob_pattern(stem: str) -> str:
    return f"{stem}.part*.parquet"


def meta_name(stem: str) -> str:
    return f"{stem}.meta.json"
