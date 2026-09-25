"""Development-only coupling comparisons and explicit screening decisions."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from hepml.domain.metrics import binomial_interval, bootstrap_wasserstein, ks_radius, shape_distances


def bootstrap_counts(size, replicates, rng):
    return np.asarray([np.bincount(rng.integers(size, size=size), minlength=size)
                       for _ in range(replicates)], dtype=np.int32)


def distance_table(development, features, config, counts):
    """One family per stage, including all features, masses and coupling pairs."""
    rows, bootstrap_errors = [], []
    masses = sorted({mass for mass, _ in development})
    pairs = list(combinations(config["couplings"], 2))
    family_size = len(masses) * len(pairs) * len(features)
    for mass in masses:
        ref = development[mass, config["couplings"][0]]
        for a, b in pairs:
            left, right = development[mass, a], development[mass, b]
            radius = ks_radius(len(left), len(right), alpha=config["stage_alpha"] / family_size)
            for feature in features:
                scale = float(np.subtract(*np.quantile(ref[feature], [.75, .25])))
                x, y = left[feature].to_numpy(), right[feature].to_numpy()
                distances = shape_distances(x, y, reference_iqr=scale)
                replicas = bootstrap_wasserstein(x, y, counts[mass, a], counts[mass, b])
                rows.append(dict(mass=mass, rho_a=a, rho_b=b, feature=feature,
                                 n_a=len(x), n_b=len(y), reference_iqr=scale, **distances,
                                 ks_lower=max(0., distances["ks"] - radius),
                                 ks_upper=min(1., distances["ks"] + radius),
                                 wasserstein_pointwise_lower=float(np.quantile(replicas, .025)),
                                 wasserstein_pointwise_upper=float(np.quantile(replicas, .975))))
                if scale > 0:
                    bootstrap_errors.append(np.abs(replicas - distances["wasserstein"]) / scale)
    # Event counts are shared across pairs/features, preserving their dependence.
    # Approximate simultaneous absolute-error bootstrap bands, conditional on
    # the fixed development reference IQR. KS uses conservative DKW bands instead.
    radius = float(np.quantile(np.max(bootstrap_errors, axis=0), 1 - config["stage_alpha"])) if bootstrap_errors else np.nan
    result = pd.DataFrame(rows)
    result["wasserstein_iqr_lower"] = (result.wasserstein_iqr - radius).clip(lower=0)
    result["wasserstein_iqr_upper"] = result.wasserstein_iqr + radius
    result["ks_followup"] = (result.ks_lower > 0) & (result.ks > config["ks_margin"])
    result["w1_followup"] = result.wasserstein_iqr_lower > config["wasserstein_iqr_margin"]
    result["within_margins"] = ((result.ks_upper < config["ks_margin"])
                                & (result.wasserstein_iqr_upper < config["wasserstein_iqr_margin"]))
    return result


def efficiency_tables(development, inventory, config):
    """ROOT-selection bookkeeping plus development-defined reference cut checks."""
    records = []
    for (mass, rho), frame in development.items():
        sample = inventory[mass, rho]
        records.append(dict(mass=mass, rho=rho, selection="compact_selection_given_ROOT",
                            passed=sample["n_selected"], total=sample["n_root"], cut=None))
        ref = development[mass, config["couplings"][0]]
        for feature in config["reference_cut_features"]:
            for quantile in config["reference_cut_quantiles"]:
                cut = float(ref[feature].quantile(quantile))
                records.append(dict(mass=mass, rho=rho, selection=f"{feature}_gt_rho01_q{quantile:g}",
                                    passed=int((frame[feature] > cut).sum()), total=len(frame), cut=cut))
    table = pd.DataFrame(records)
    table["efficiency"] = table.passed / table.total
    intervals = [binomial_interval(int(row.passed), int(row.total),
                                  alpha=config["stage_alpha"] / len(table)) for row in table.itertuples()]
    table[["lower", "upper"]] = intervals
    comparisons = []
    for (mass, selection), group in table.groupby(["mass", "selection"]):
        group = group.set_index("rho")
        for a, b in combinations(config["couplings"], 2):
            left, right = group.loc[a], group.loc[b]
            ratio = right.efficiency / left.efficiency if left.efficiency > 0 else np.nan
            lower = right.lower / left.upper if left.upper > 0 else np.nan
            upper = right.upper / left.lower if left.lower > 0 else np.nan
            margin = config["efficiency_relative_margin"]
            comparisons.append(dict(mass=mass, selection=selection, rho_a=a, rho_b=b,
                                    difference=right.efficiency-left.efficiency, ratio=ratio,
                                    ratio_lower=lower, ratio_upper=upper,
                                    followup=lower > 1 + margin or upper < 1 - margin,
                                    within_margin=lower > 1 - margin and upper < 1 + margin))
    return table, pd.DataFrame(comparisons)


def followup_reasons(distances, efficiencies, joint, config):
    reasons = []
    if (distances.ks_followup | distances.w1_followup).any():
        reasons.append("Marginal shape differences exceed the predeclared follow-up rule")
    if efficiencies.followup.any():
        reasons.append("Selection/reference-cut efficiency differences exceed the 5% margin")
    # Coarse permutation resolution is explicit; unresolved p-values do not pass.
    if ((joint.auc_lower > config["two_sample_auc_margin"])
            & (joint.permutation_p_adjusted <= config["stage_alpha"])).any():
        reasons.append("Joint-feature coupling classifier passes practical and permutation checks")
    return reasons


def audit_decision(distances, efficiencies, joint, *, production_complete, config):
    if followup_reasons(distances, efficiencies, joint, config):
        return "shape-relevant"
    if (production_complete and distances.within_margins.all() and efficiencies.within_margin.all()
            and (joint.auc_upper < config["two_sample_auc_margin"]).all()):
        return "provisionally rate-only"
    return "inconclusive"
