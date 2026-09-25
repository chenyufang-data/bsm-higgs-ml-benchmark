"""Assemble a benchmark in memory, preserving event identity and normalization."""

from __future__ import annotations

import pandas as pd

from hepml.domain.validation import sanity_prepared_ml
from hepml.log import get_logger

log = get_logger(__name__)


def registered_splits(frame: pd.DataFrame, registry: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Project an existing physical-event registry without assigning any new events."""
    if frame.event_id.duplicated().any() or registry.event_id.duplicated().any():
        raise ValueError("Duplicate physical events")
    registered = registry[registry.sample_key.isin(frame.sample_key.unique())]
    identity = frame[['event_id', 'sample_key']].sort_values('event_id').reset_index(drop=True)
    expected = registered[['event_id', 'sample_key']].sort_values('event_id').reset_index(drop=True)
    if not identity.equals(expected):
        raise ValueError("Benchmark membership differs from the registered samples")
    if not registered.split.isin(['train', 'val', 'test']).all():
        raise ValueError("Invalid registry split")
    return {name: frame[frame.event_id.isin(registered.loc[registered.split == name, 'event_id'])].reset_index(drop=True)
            for name in ('train', 'val', 'test')}


def assemble_dataset(
    frames: list[pd.DataFrame],
    sig_meta: dict,
    features: list[str],
    lumi: float,
) -> tuple[pd.DataFrame, dict]:
    sig_mass = str(sig_meta["mass"])
    df = pd.concat(frames, ignore_index=True)
    if "event_id" in df:
        if df["event_id"].isna().any() or df["event_id"].duplicated().any():
            raise ValueError("Mixed legacy/identified inputs or duplicate events across samples")

    # rename for clarity (raw generator branches kept as provenance only)
    df = df.rename(
        columns={
            "label": "target",
            "weight": "gen_weight",
            "xs": "evt_xs",
        }
    )

    # ---- HARD sanity checks ----
    report = sanity_prepared_ml(
        df,
        features=features,
        name=f"sig{sig_mass}_prepared",
        lumi=lumi,
    )

    # ---- summary ----
    counts = df["target"].value_counts(dropna=False)
    S = float(df.loc[df["target"] == 1, "sample_weight"].sum())
    B = float(df.loc[df["target"] == 0, "sample_weight"].sum())

    log.info("Built dataset for sig_%s: shape=%s", sig_mass, df.shape)
    log.info("Target counts: %s", counts.to_dict())
    log.info("Expected yields at lumi=%g (xs_ref): S=%.6g, B=%.6g", lumi, S, B)

    return df, report
