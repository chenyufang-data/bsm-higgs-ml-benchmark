"""File-naming contracts for prepared benchmarks and working models.

Compact-export file names (shards and sidecars) belong to the standalone compactor:
see hepml_compact.contracts. Signal hypotheses are still keyed by mass here, so
moving to (mass, coupling) hypothesis IDs is a change to this module only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def dataset_filename(mass) -> str:
    return f"dataset_sig{mass}_vs_bkg.parquet"


def split_filename(split: str, mass) -> str:
    return f"{split}_sig{mass}.parquet"


def split_meta_filename(mass) -> str:
    return f"split_sig{mass}.meta.json"


def assignments_filename(mass) -> str:
    return f"assignments_sig{mass}.parquet"


def model_dirname(mass) -> str:
    """Per-hypothesis model directory inside a run's models/ folder."""
    return f"sig{mass}"


def split_meta_path(splits_dir: Path, mass) -> Path:
    return Path(splits_dir) / split_meta_filename(mass)


@dataclass(frozen=True)
class BenchmarkFiles:
    """Layout of one prepared benchmark: <root>/dataset_*.parquet and <root>/splits/*."""

    root: Path

    @property
    def splits(self) -> Path:
        return Path(self.root) / "splits"

    def dataset(self, mass) -> Path:
        return Path(self.root) / dataset_filename(mass)

    def split(self, name: str, mass) -> Path:
        return self.splits / split_filename(name, mass)

    def assignments(self, mass) -> Path:
        return self.splits / assignments_filename(mass)

    def meta(self, mass) -> Path:
        return self.splits / split_meta_filename(mass)
