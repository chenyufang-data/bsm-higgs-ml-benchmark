"""One operating-point protocol for all models; filesystem-independent."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from hepml.domain.config import EvaluationConfig
from hepml.domain.metrics import (
    OBJECTIVES,
    binomial_interval,
    normalized_weights,
    operating_curve,
    safe_auc,
    sample_normalization,
    select_operating_points,
)
from hepml.domain.weights import cross_section_factors


def point_rows(metadata, **identity):
    rows = []
    for objective, point in metadata['operating_points'].items():
        values = dict.fromkeys(['S', 'B', 'S_over_B', *OBJECTIVES, 'background_neff',
                               'signal_efficiency', 'background_efficiency'])
        values.update(point.get('metrics', {}))
        rows.append(dict(values, **identity, objective=objective, status=point['status'],
                         threshold=point['threshold'], plateau_threshold=point['plateau_threshold']))
    return rows


def evaluation_weights(full, split, *, stored_lumi, target_lumi, signal_k_factor=1.0, background_k_factor=1.0):
    if not all(np.isfinite(value) and value > 0 for value in (stored_lumi, target_lumi)):
        raise ValueError("Luminosities must be finite and positive")
    normalization = sample_normalization(full, split)
    factors = cross_section_factors(split.target, signal_k_factor=signal_k_factor,
                                    background_k_factor=background_k_factor)
    return normalized_weights(split, normalization) * target_lumi / stored_lumi * factors, normalization


def evaluate_validation(full, predictions, config, *, stored_lumi):
    weights, normalization = evaluation_weights(full, predictions, stored_lumi=stored_lumi,
                                                  target_lumi=config.lumi_pb_inv,
                                                  signal_k_factor=config.signal_k_factor,
                                                  background_k_factor=config.background_k_factor)
    curve, processes = operating_curve(predictions, weights, config)
    points = select_operating_points(curve, plateau_fraction=config.plateau_fraction)
    metrics = dict(schema_version=2, yield_scope=config.yield_scope, lumi_pb_inv=config.lumi_pb_inv,
                   threshold_source="validation", config=asdict(config), normalization=normalization,
                   auc_weighted=safe_auc(predictions.target, predictions.bdt_score, weights),
                   auc_unweighted=safe_auc(predictions.target, predictions.bdt_score), operating_points=points)
    return metrics, curve, processes, weights


def evaluate_frozen(full, predictions, metadata, config, *, stored_lumi, split):
    """Apply saved validation thresholds verbatim; never optimize on this split."""
    if metadata["threshold_source"] != "validation" or metadata["schema_version"] != 2:
        raise ValueError("Require version-2 validation-selected operating points")
    if config.lumi_pb_inv != metadata["lumi_pb_inv"]:
        raise ValueError("Frozen evaluation luminosity changed")
    # Older version-2 releases had no k-factor fields and mean k=1, not today's study settings.
    recorded = asdict(EvaluationConfig(**metadata["config"]))
    recorded["background_systematics"] = tuple(recorded["background_systematics"])
    current = dict(asdict(config), background_systematics=tuple(config.background_systematics))
    if recorded != current:
        raise ValueError("Frozen evaluation configuration changed")
    weights, _ = evaluation_weights(full, predictions, stored_lumi=stored_lumi, target_lumi=config.lumi_pb_inv,
                                   signal_k_factor=config.signal_k_factor,
                                   background_k_factor=config.background_k_factor)
    valid_points = {name: point for name, point in metadata["operating_points"].items() if point["status"] == "valid"}
    if not valid_points:
        return pd.DataFrame()
    thresholds = np.unique([point["threshold"] for point in valid_points.values()])
    curve, _ = operating_curve(predictions, weights, config, thresholds=thresholds)
    rows = []
    for name, point in valid_points.items():
        row = curve.loc[curve.threshold == point["threshold"]].iloc[0].to_dict()
        rows.append(dict(row, objective=name, split=split, threshold_source="validation"))
    return pd.DataFrame(rows)


def bootstrap_validation(predictions, weights, curve, config, *, replicates, seed):
    """Within-process event resampling; fixed model and validation grid.

    MC bands condition on preselection rates and fitted scores. They are distinct
    from the 5/10% scenarios and from training-seed variation. Each background
    process keeps its original number of validation draws.
    """
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(predictions["sample"].to_numpy() == sample) for sample in predictions["sample"].unique()]
    order = np.argsort(predictions.bdt_score.to_numpy(), kind="stable")
    names = ["S_over_B", *OBJECTIVES]
    replicas = {name: [] for name in names}
    selected = []
    for index in range(replicates):
        counts = np.zeros(len(predictions), dtype=int)
        for group in groups:
            counts[group] = np.bincount(rng.integers(len(group), size=len(group)), minlength=len(group))
        boot, _ = operating_curve(predictions, weights, config, thresholds=curve.threshold.to_numpy(),
                                  multiplicity=counts, score_order=order, include_processes=False)
        for name in names:
            replicas[name].append(boot[name].to_numpy())
        points = select_operating_points(boot, plateau_fraction=config.plateau_fraction)
        for objective, point in points.items():
            selected.append(dict(replicate=index, objective=objective, status=point["status"], threshold=point["threshold"]))
    bands = pd.DataFrame({"threshold": curve.threshold})
    for name, values in replicas.items():
        array = np.asarray(values)
        valid = np.isfinite(array).all(axis=0)
        low, high = np.full(len(curve), np.nan), np.full(len(curve), np.nan)
        if valid.any():
            low[valid], high[valid] = np.quantile(array[:, valid], [.025, .975], axis=0)
        bands[f"{name}_lower"], bands[f"{name}_upper"] = low, high
        bands[f"{name}_finite_fraction"] = np.isfinite(array).mean(axis=0)
    return bands, pd.DataFrame(selected)


def weighted_efficiency_interval(weights, selected, *, z=1.959963984540054):
    """Selected weight fraction with a normal-approximation 95% interval, one weighted event per row."""
    weights, selected = np.asarray(weights, dtype=float), np.asarray(selected, dtype=bool)
    total = weights.sum()
    efficiency = weights[selected].sum() / total
    sigma = np.sqrt(np.sum(weights**2 * (selected - efficiency) ** 2)) / total
    return max(0.0, efficiency - z * sigma), min(1.0, efficiency + z * sigma)


def process_support(predictions, processes, normalization, config, *, stored_lumi):
    """Per-process efficiency intervals at a selected threshold.

    Uniform-weight samples get conditional binomial intervals; samples whose rows carry
    different weights (truth tagging's pass probabilities) get a normal-approximation
    interval from their weights.
    """
    rows = []
    for item in processes.to_dict("records"):
        group = predictions[predictions["sample"] == item["sample"]]
        if group.sample_weight.nunique() == 1:
            low, high = binomial_interval(int(round(item["mc_count"])), len(group), alpha=.05)
            method = "binomial"
        else:
            low, high = weighted_efficiency_interval(group.sample_weight,
                                                     group.bdt_score.to_numpy() >= item["threshold"])
            method = "weighted_normal"
        factor = config.signal_k_factor if item["target"] == 1 else config.background_k_factor
        pre_yield = normalization["samples"][item["sample"]]["full_weight_sum"] * config.lumi_pb_inv / stored_lumi * factor
        rows.append(dict(item, validation_mc=len(group), efficiency_lower=low, efficiency_upper=high,
                         yield_lower=low*pre_yield, yield_upper=high*pre_yield,
                         **({} if method == "binomial" else {"interval_method": method})))
    return pd.DataFrame(rows)


def roc_table(full, predictions, config, *, stored_lumi, split):
    weights, _ = evaluation_weights(full, predictions, stored_lumi=stored_lumi,
        target_lumi=config.lumi_pb_inv, signal_k_factor=config.signal_k_factor,
        background_k_factor=config.background_k_factor)
    fpr, tpr, thresholds = roc_curve(predictions.target, predictions.bdt_score, sample_weight=weights)
    return pd.DataFrame(dict(split=split, background_efficiency=fpr, signal_efficiency=tpr, threshold=thresholds))


def roc_tables(full, predictions, config, *, stored_lumi):
    return pd.concat([roc_table(full, frame, config, stored_lumi=stored_lumi, split=name)
                      for name, frame in predictions.items()], ignore_index=True)


@dataclass
class ValidationRecord:
    """One model's shared evaluation; every threshold comes from the validation split."""

    metadata: dict
    curve: pd.DataFrame
    processes: pd.DataFrame
    weights: np.ndarray
    bands: pd.DataFrame
    stability: pd.DataFrame
    frozen: dict[str, pd.DataFrame]


def validate_model(full, predictions, config, *, stored_lumi, replicates, seed) -> ValidationRecord:
    """Select operating points on predictions['val'], bootstrap them, and apply them to every split given."""
    metadata, curve, processes, weights = evaluate_validation(full, predictions["val"], config, stored_lumi=stored_lumi)
    bands, stability = bootstrap_validation(predictions["val"], weights, curve, config,
                                            replicates=replicates, seed=seed)
    frozen = {name: evaluate_frozen(full, frame, metadata, config, stored_lumi=stored_lumi, split=name)
              for name, frame in predictions.items()}
    return ValidationRecord(metadata, curve, processes, weights, bands, stability, frozen)


def selected_support(record, validation, config, *, stored_lumi):
    """(objective, per-process support) at each valid validation-selected operating point."""
    for objective, point in record.metadata["operating_points"].items():
        if point["status"] == "valid":
            selected = record.processes[record.processes.threshold == point["threshold"]]
            yield objective, process_support(validation, selected, record.metadata["normalization"], config,
                                             stored_lumi=stored_lumi)


def check_yield_closure(full, validation, weights, config, *, stored_lumi) -> None:
    """Validation weights reproduce each class's full-sample yield with its k-factor applied once."""
    for target, factor in [(1, config.signal_k_factor), (0, config.background_k_factor)]:
        expected = full.loc[full.target == target, "sample_weight"].sum() * config.lumi_pb_inv / stored_lumi * factor
        actual = weights[validation.target.to_numpy() == target].sum()
        if not np.isclose(actual, expected, rtol=1e-10, atol=0):
            raise ValueError(f"Class {target}: validation yield {actual} differs from full yield {expected}")
