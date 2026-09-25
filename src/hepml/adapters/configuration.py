"""Study settings, compact sample manifests and machine-local output paths."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from hepml.domain.config import (
    AnalysisConfig,
    EvaluationConfig,
    ScanConfig,
    SplitConfig,
    TrainingConfig,
)
from hepml.log import get_logger

log = get_logger(__name__)

_study_directory: ContextVar[Path] = ContextVar("study_directory", default=Path("studies/cg_bbc"))


@contextmanager
def study_context(directory: Path):
    """Scope defaults to one CLI invocation without changing the environment."""
    token = _study_directory.set(directory)
    try:
        yield
    finally:
        _study_directory.reset(token)


def default_study_directory() -> Path:
    return _study_directory.get()


# ------------------------------------------------------------
# .env loading (machine-local overrides; never committed)
# ------------------------------------------------------------
def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Seed os.environ from a .env file; existing variables are NOT overridden.

    Supports comments, blank lines, and optionally quoted values. Returns
    the variables that were actually applied. Missing file is fine.
    """
    p = Path(path)
    applied: dict[str, str] = {}
    if not p.exists():
        return applied
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    if applied:
        log.info("Loaded %d setting(s) from %s: %s", len(applied), p, ", ".join(sorted(applied)))
    return applied


# ------------------------------------------------------------
# Output directory layout (single root, relocatable via .env)
# ------------------------------------------------------------
@dataclass(frozen=True)
class OutputPaths:
    """One output root; each study owns compact/, datasets/, runs/, releases/."""

    base: Path
    root_parquet: Path
    datasets: Path
    splits: Path
    models_work: Path
    models_final: Path
    ablation: Path


def output_paths(base: str | Path | None = None, *, study: str | None = None) -> OutputPaths:
    """Resolve the output layout; HEPML_OUTPUT_ROOT (or .env) moves the base.

    Point HEPML_OUTPUT_ROOT outside synced folders (OneDrive/Dropbox) to
    keep training churn off the cloud.
    """
    load_dotenv()
    if base is None:
        base = os.environ.get("HEPML_OUTPUT_ROOT", "outputs")
    base = Path(base)
    if study is None:
        config = default_study_directory() / "extraction.yaml"
        study = yaml.safe_load(config.read_text(encoding="utf-8"))["name"] if config.exists() else "cg_bbc"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", study):
        raise ValueError("Study name must be a safe directory name")
    workspace = base / study
    return OutputPaths(
        base=base,
        root_parquet=workspace / "compact",
        datasets=workspace / "datasets",
        splits=workspace / "datasets" / "splits",
        models_work=workspace / "runs",
        models_final=workspace / "releases",
        ablation=workspace / "ablation",
    )


def load_analysis(path: str | Path | None = None) -> AnalysisConfig:
    """Load studies/cg_bbc/analysis.yaml (or `path`) into an AnalysisConfig.

    With no explicit path, a missing file just yields the defaults; an
    explicitly given path that does not exist raises. Unknown keys in the
    YAML raise TypeError so typos are caught rather than ignored.
    """
    p = Path(path) if path is not None else default_study_directory() / "analysis.yaml"
    if not p.exists():
        if path is not None:
            raise FileNotFoundError(f"Analysis config not found: {p}")
        return AnalysisConfig()

    raw = yaml.safe_load(p.read_text()) or {}
    unknown = set(raw) - {"physics", "training", "splits", "evaluation"}
    if unknown:
        raise ValueError(f"Unknown analysis sections: {sorted(unknown)}; put selection cuts in extraction.yaml")
    physics = raw.get("physics", {}) or {}
    training_raw = dict(raw.get("training", {}) or {})
    scan_raw = training_raw.pop("scan", {}) or {}

    return AnalysisConfig(
        lumi=float(physics.get("lumi", AnalysisConfig.lumi)),
        training=TrainingConfig(scan=ScanConfig(**scan_raw), **training_raw),
        splits=SplitConfig(**(raw.get("splits", {}) or {})),
        evaluation=EvaluationConfig(**(raw.get("evaluation", {}) or {})),
    )


def fit_parameters(analysis: AnalysisConfig, overrides=()) -> dict:
    """Model hyperparameters for a study fit.

    analysis.yaml's training settings without the per-fit seed and the legacy scan,
    changed only by explicit KEY=VALUE overrides that the calling study records.
    """
    params = asdict(analysis.training)
    params.pop("scan")
    params.pop("seed")
    for item in overrides:
        key, separator, raw = item.partition("=")
        if not separator or key not in params:
            raise ValueError(f"Invalid training override {item!r}; use KEY=VALUE with KEY in {sorted(params)}")
        value, expected = yaml.safe_load(raw), type(params[key])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (expected is int and not isinstance(value, int)):
            raise ValueError(f"Training override {key} needs a {expected.__name__}, got {raw!r}")
        params[key] = expected(value)
    return params
