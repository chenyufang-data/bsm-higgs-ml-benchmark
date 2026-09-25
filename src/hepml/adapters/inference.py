#!/usr/bin/env python3
"""XGBoost model loading and inference file adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError as e:
    raise SystemExit("xgboost is required for inference. Install: pip install xgboost") from e

try:
    import joblib
except ImportError:
    joblib = None

from hepml.adapters.configuration import output_paths
from hepml.domain.artifacts import split_meta_filename
from hepml.log import get_logger

log = get_logger(__name__)


def read_features(path: Path) -> list[str]:
    feats = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not feats:
        raise ValueError(f"No features found in {path}")
    return feats


def read_frozen_record(path: Path, hypothesis_mass: float | None = None) -> dict:
    record = json.loads(path.read_text())
    if 'conditioning' in record:
        from hepml.application.conditioning import condition_frame

        condition_frame(pd.DataFrame(),hypothesis_mass,record['conditioning'])
        selected = record['hypotheses'][str(int(hypothesis_mass))]
        return dict(selected, conditioning=record['conditioning'], hypothesis_mass=float(hypothesis_mass))
    if hypothesis_mass is not None:
        raise ValueError('Hypothesis mass was supplied for an unconditional model')
    return record


def load_threshold(path: Path, objective: str | None = None, *, hypothesis_mass: float | None = None) -> float:
    record = read_frozen_record(path,hypothesis_mass)
    if objective is not None:
        evaluation = record.get("evaluation", {})
        if evaluation.get("schema_version") != 2 or evaluation.get("threshold_source") != "validation":
            raise ValueError("Named objectives require frozen version-2 validation operating points")
        point = evaluation["operating_points"][objective]
        if point["status"] != "valid":
            raise ValueError(f"{objective}: no_valid_threshold")
        return float(point["threshold"])
    thr = record.get("threshold", None)
    if thr is None:
        raise KeyError(f"'threshold' not found in {path}")
    return float(thr)


def load_conditional_model(model_dir: Path):
    """Reload one shared model; every score call requires a supported scalar mass."""
    from hepml.application.conditioning import condition_frame
    from hepml.application.training import make_xyw

    metadata = json.loads((model_dir / 'metrics.json').read_text())
    conditioning = metadata['conditioning']
    _, _, predict = load_model(model_dir)

    def score(frame, mass):
        conditioned = condition_frame(frame,mass,conditioning)
        if not {'target','sample_weight'}.issubset(conditioned):
            # Prediction does not require labels or physical rates.
            conditioned = conditioned.assign(target=0, sample_weight=1.)
        X, _, _ = make_xyw(conditioned,metadata['features'])
        if not np.isfinite(X).all():
            raise ValueError('Nonfinite conditional inference inputs')
        return predict(X)

    return metadata, score


def load_best_iteration(model_dir: Path) -> int | None:
    """
    Recover the early-stopping best_iteration for this model.
    Prefer threshold.json (frozen releases), fall back to metrics.json
    (working dirs). Returns None if unavailable.
    """
    for fname, keys in (("threshold.json", ["best_iteration"]), ("metrics.json", ["xgb", "best_iteration"])):
        path = model_dir / fname
        if not path.exists():
            continue
        try:
            obj = json.loads(path.read_text())
            for k in keys:
                obj = obj[k]
            if obj is not None and int(obj) >= 0:
                return int(obj)
        except (KeyError, TypeError, ValueError):
            continue
    return None


def load_model(model_dir: Path):
    """
    Prefer model.ubj. Fall back to model.joblib.
    Returns a callable object that provides predict_proba or booster predict.
    """
    ubj_path = model_dir / "model.ubj"
    joblib_path = model_dir / "model.joblib"

    if ubj_path.exists():
        booster = xgb.Booster()
        booster.load_model(ubj_path.as_posix())

        # The raw booster keeps ALL trained trees; training-time scores and the
        # frozen threshold come from the early-stopped model, so we must slice
        # to best_iteration or the scores here diverge from the tuned ones.
        best_it = load_best_iteration(model_dir)
        if best_it is None:
            log.warning(
                "best_iteration not found in threshold.json/metrics.json; "
                "scoring with ALL trees (may differ from training-time scores)"
            )

        def predict_scores(X: np.ndarray) -> np.ndarray:
            dmat = xgb.DMatrix(X)
            if best_it is not None:
                p = booster.predict(dmat, iteration_range=(0, best_it + 1))
            else:
                p = booster.predict(dmat)
            return p.astype(np.float32)

        return "ubj", ubj_path, predict_scores

    if joblib_path.exists():
        if joblib is None:
            raise SystemExit("joblib not available but model.joblib exists. Install: pip install joblib")
        model = joblib.load(joblib_path)

        def predict_scores(X: np.ndarray) -> np.ndarray:
            # XGBClassifier honors best_iteration in predict_proba
            p = model.predict_proba(X)[:, 1]
            return p.astype(np.float32)

        return "joblib", joblib_path, predict_scores

    raise FileNotFoundError(f"No model found in {model_dir} (expected model.ubj or model.joblib)")


def load_split_meta(mass: int, splits_dir: Path | None = None) -> dict[str, Any]:
    """
    Load split metadata produced by prepare_ml's stratified split writer.
    Returns {} if not found (so predict still works standalone).
    """
    if splits_dir is None:
        splits_dir = output_paths().splits
    meta_path = splits_dir / split_meta_filename(mass)
    if not meta_path.exists():
        log.warning("Split meta not found: %s (skipping weighted_frac_*)", meta_path)
        return {}
    try:
        return json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Failed to read split meta: %s (%s)", meta_path, e)
        return {}


def read_input(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format: {fmt}. Choose parquet or csv.")


def write_output(df: pd.DataFrame, path: Path, fmt: str):
    if fmt == "parquet":
        df.to_parquet(path, index=False)
        return
    if fmt == "csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported format: {fmt}. Choose parquet or csv.")
