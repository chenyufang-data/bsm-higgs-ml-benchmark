from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hepml.domain.weights import compute_sample_weight

# raw ROOT-derived parquet schema
REQUIRED_META_RAW = ["label", "weight", "xs"]

# prepared ML schema (after prepare_ml.py rename + additions)
REQUIRED_META_PREP = ["target", "gen_weight", "evt_xs", "sample_weight", "xs_pb", "n_events_total"]


class SanityError(ValueError):
    """Raised when a sanity check fails."""


@dataclass(frozen=True)
class WeightSummary:
    n: int
    min: float
    max: float
    sum: float

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "min": self.min, "max": self.max, "sum": self.sum}


def require_columns(df: pd.DataFrame, cols: Sequence[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SanityError(
            f"{name}: missing required columns: {missing}. "
            f"Found (first 50): {list(df.columns[:50])}" + (" ..." if df.shape[1] > 50 else "")
        )


def require_numeric(df: pd.DataFrame, cols: Sequence[str], *, name: str) -> None:
    bad = []
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            bad.append((c, str(df[c].dtype)))
    if bad:
        msg = ", ".join([f"{c} (dtype={dt})" for c, dt in bad])
        raise SanityError(f"{name}: expected numeric dtype, got: {msg}")


def check_binary_01(df: pd.DataFrame, col: str, *, name: str) -> None:
    require_columns(df, [col], name=name)
    require_numeric(df, [col], name=name)
    vals = pd.unique(df[col].dropna())
    bad = [v for v in vals.tolist() if v not in {0, 1}]
    if bad:
        raise SanityError(f"{name}: {col} must be binary 0/1; found values: {sorted(bad)}")


def check_no_nan_inf(df: pd.DataFrame, cols: Sequence[str], *, name: str) -> dict[str, int]:
    require_columns(df, cols, name=name)
    require_numeric(df, cols, name=name)

    sub = df[list(cols)]
    n_nan = int(sub.isna().to_numpy().sum())
    arr = sub.to_numpy(dtype=float, copy=False)
    n_nonfinite = int((~np.isfinite(arr)).sum())
    n_inf = n_nonfinite - n_nan

    if n_nan or n_inf:
        raise SanityError(
            f"{name}: invalid values found in columns {list(cols)[:8]}{'...' if len(cols) > 8 else ''}: "
            f"NaN={n_nan}, Inf/NonFinite={n_inf}"
        )
    return {"n_nan": n_nan, "n_inf": n_inf}


def check_nonnegative_finite(df: pd.DataFrame, col: str, *, name: str, allow_zero_sum: bool = False) -> WeightSummary:
    require_columns(df, [col], name=name)
    require_numeric(df, [col], name=name)

    x = df[col].to_numpy(dtype=float, copy=False)
    if not np.isfinite(x).all():
        n_bad = int((~np.isfinite(x)).sum())
        raise SanityError(f"{name}: {col} has {n_bad} non-finite values (NaN/Inf).")
    if (x < 0).any():
        n_neg = int((x < 0).sum())
        raise SanityError(f"{name}: {col} has {n_neg} negative values; expected >= 0.")
    s = float(x.sum())
    if (s == 0.0) and (not allow_zero_sum):
        raise SanityError(f"{name}: {col} sums to 0. Check upstream weight/rate calculation.")

    return WeightSummary(n=int(len(x)), min=float(x.min()), max=float(x.max()), sum=s)


def sanity_root_parquet(
    df: pd.DataFrame,
    *,
    features: Sequence[str],
    name: str = "root_parquet",
) -> dict[str, Any]:
    """
    Sanity checks for root_outputs/*.parquet (raw).
    Expected columns:
      - features
      - label (0/1), weight, xs
    """
    require_columns(df, list(REQUIRED_META_RAW) + list(features), name=name)

    # features
    check_no_nan_inf(df, features, name=name)

    # meta
    check_binary_01(df, "label", name=name)
    w = check_nonnegative_finite(df, "weight", name=name + ".weight")
    xs = check_nonnegative_finite(df, "xs", name=name + ".xs")

    return {
        "stage": "root_parquet",
        "name": name,
        "n_rows": int(len(df)),
        "weight": w.to_dict(),
        "xs": xs.to_dict(),
    }


def sanity_prepared_ml(
    df: pd.DataFrame,
    *,
    features: Sequence[str],
    name: str = "prepared_ml",
    check_sample_weight_formula: bool = True,
    rtol: float = 1e-6,
    atol: float = 1e-12,
    lumi: float = 3000,
) -> dict[str, Any]:
    """
    Sanity checks for prepared ML parquet (after prepare_ml.py):
      label -> target
      weight -> gen_weight   (provenance only, not used in normalization)
      xs -> evt_xs           (provenance only)
      sample_weight = lumi * xs_pb / n_events_total
    (the formula itself lives in hepml.domain.weights.compute_sample_weight)
    """
    require_columns(df, list(REQUIRED_META_PREP) + list(features), name=name)

    # features
    check_no_nan_inf(df, features, name=name)

    # meta
    check_binary_01(df, "target", name=name)
    gw = check_nonnegative_finite(df, "gen_weight", name=name + ".gen_weight")
    er = check_nonnegative_finite(df, "evt_xs", name=name + ".evt_xs")
    sw = check_nonnegative_finite(df, "sample_weight", name=name + ".sample_weight")

    if check_sample_weight_formula:
        xs = df["xs_pb"].to_numpy(dtype=float, copy=False)
        n_tot = df["n_events_total"].to_numpy(dtype=float, copy=False)
        s = df["sample_weight"].to_numpy(dtype=float, copy=False)
        expected = compute_sample_weight(xs, n_tot, lumi=lumi)

        if not np.allclose(s, expected, rtol=rtol, atol=atol):
            diff = np.abs(s - expected)
            idx = np.argsort(diff)[-5:][::-1]
            examples = [
                {
                    "row": int(i),
                    "xs_pb": float(xs[i]),
                    "n_events_total": float(n_tot[i]),
                    "sample_weight": float(s[i]),
                    "expected": float(expected[i]),
                    "abs_diff": float(diff[i]),
                }
                for i in idx
            ]
            raise SanityError(
                f"{name}: sample_weight != lumi * xs_pb / n_events_total (rtol={rtol}, atol={atol}). "
                f"Top mismatches: {examples}"
            )

    return {
        "stage": "prepared_ml",
        "name": name,
        "n_rows": int(len(df)),
        "summaries": {
            "gen_weight": gw.to_dict(),
            "evt_xs": er.to_dict(),
            "sample_weight": sw.to_dict(),
        },
    }


def sanity_predictions(
    df_preds: pd.DataFrame,
    *,
    score_col: str = "score",
    name: str = "preds",
) -> dict[str, Any]:
    """
    Sanity checks for prediction outputs.
    Ensures score exists, finite, and within [0,1] (probability).
    """
    require_columns(df_preds, [score_col], name=name)
    require_numeric(df_preds, [score_col], name=name)

    s = df_preds[score_col].to_numpy(dtype=float, copy=False)
    if not np.isfinite(s).all():
        n_bad = int((~np.isfinite(s)).sum())
        raise SanityError(f"{name}: {score_col} has {n_bad} non-finite values (NaN/Inf).")

    lo, hi = float(s.min()), float(s.max())
    if lo < -1e-9 or hi > 1 + 1e-9:
        raise SanityError(
            f"{name}: {score_col} out of [0,1] range (min={lo:.6g}, max={hi:.6g}). "
            f"Did you save margin/logit instead of probability?"
        )

    q = np.quantile(s, [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]).tolist()
    return {
        "stage": "predictions",
        "name": name,
        "n_rows": int(len(df_preds)),
        "score_quantiles": {"q0": q[0], "q50": q[1], "q90": q[2], "q95": q[3], "q99": q[4], "q100": q[5]},
    }
