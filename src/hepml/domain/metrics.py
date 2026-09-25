"""Shared statistics helpers: Asimov significance and split fractions.

This is the only implementation of these quantities; train_bdt,
plot_bdt_diagnostics, summarize_inference and prepare_ml all import
from here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def shape_distances(x, y, *, reference_iqr: float) -> dict:
    """Unweighted empirical distances; constant per-sample weights cancel."""
    from scipy.stats import ks_2samp, wasserstein_distance

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not len(x) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Shape distances require nonempty finite samples")
    w1 = float(wasserstein_distance(x, y))
    return dict(ks=float(ks_2samp(x, y, method="asymp").statistic), wasserstein=w1,
                wasserstein_iqr=w1 / reference_iqr if reference_iqr > 0 else None)


def ks_radius(n_x: int, n_y: int, *, alpha: float) -> float:
    """Two empirical-CDF DKW bands, union-bounded for a supplied error budget.

    Valid with ties and no continuous-distribution KS p-value assumption.
    Assumes independent events; production lineage remains an external check.
    """
    return float(np.sqrt(np.log(4 / alpha) / (2 * n_x)) + np.sqrt(np.log(4 / alpha) / (2 * n_y)))


def binomial_interval(passed: int, total: int, *, alpha: float) -> tuple[float, float]:
    """Clopper-Pearson interval, including all-pass and zero-pass cases."""
    from scipy.stats import beta

    if total <= 0 or not 0 <= passed <= total or not 0 < alpha < 1:
        raise ValueError("Invalid binomial count or confidence level")
    return (float(beta.ppf(alpha / 2, passed, total - passed + 1)) if passed else 0.,
            float(beta.ppf(1 - alpha / 2, passed + 1, total - passed)) if passed < total else 1.)


def bootstrap_wasserstein(x, y, counts_x, counts_y) -> np.ndarray:
    """Event bootstrap distances on cached empirical support, in bounded blocks.

    Counts are shared across features/pairs to retain within-event dependence
    when constructing simultaneous bootstrap intervals for a whole audit stage.
    """
    x, y = np.asarray(x), np.asarray(y)
    order_x, order_y = np.argsort(x), np.argsort(y)
    grid = np.unique(np.concatenate([x, y]))
    pos_x = np.searchsorted(x[order_x], grid, side="right")
    pos_y = np.searchsorted(y[order_y], grid, side="right")
    results = []
    for start in range(0, len(counts_x), 20):
        cx, cy = counts_x[start:start + 20], counts_y[start:start + 20]
        cdf_x = np.pad(np.cumsum(cx[:, order_x], axis=1), ((0, 0), (1, 0)))[:, pos_x] / len(x)
        cdf_y = np.pad(np.cumsum(cy[:, order_y], axis=1), ((0, 0), (1, 0)))[:, pos_y] / len(y)
        results.extend((np.abs(cdf_x - cdf_y)[:, :-1] * np.diff(grid)).sum(axis=1))
    return np.asarray(results)


def asimov_z(S: float, B: float) -> float:
    """Asimov/AMS significance sqrt(2*((S+B)*ln(1+S/B) - S)).

    Returns 0.0 when S <= 0 or B <= 0: the formula diverges as B -> 0,
    and a selection with no surviving background is a statistics problem,
    not a discovery. Callers that need a guard against tiny B should
    apply explicit yield constraints (as the threshold scan does).
    """
    if S <= 0.0 or B <= 0.0:
        return 0.0
    r = S / B
    if r < 1e-7:
        # Preserve the legacy definition while avoiding subtraction of nearly
        # equal terms for tiny S/B. Ordinary/reference inputs are unchanged.
        return float(S / np.sqrt(B) * np.sqrt(1 - r / 3 + r * r / 6 - r**3 / 10))
    return float(np.sqrt(2.0 * ((S + B) * np.log1p(S / B) - S)))


def asimov_z_syst(S: float, B: float, delta_b: float) -> float:
    """Cowan counting-experiment Eq. 20; S/B already contain luminosity.

    sigma_b = delta_b * B. The constraint is the effective Poisson-control model,
    not a complete correlated nuisance likelihood. Zero B uses the legacy zero
    sentinel; the operating-point API separately marks it invalid.
    """
    if not all(np.isfinite(x) and x >= 0 for x in (S, B, delta_b)):
        raise ValueError("Yields and background uncertainty must be finite and non-negative")
    if delta_b == 0 or S == 0 or B == 0:
        return asimov_z(S, B)
    return float(significance_arrays(S, B, delta_b))


OBJECTIVES = ("Z_Asimov", "Z_syst_5pct", "Z_syst_10pct")


def significance_arrays(signal, background, delta_b=0.):
    """Vectorized form of the same counting formulas for scans/resampling."""
    s, b = np.broadcast_arrays(np.asarray(signal, dtype=float), np.asarray(background, dtype=float))
    if not np.isfinite(s).all() or not np.isfinite(b).all() or (s < 0).any() or (b < 0).any():
        raise ValueError("Invalid expected yields")
    if not np.isfinite(delta_b) or delta_b < 0:
        raise ValueError("Invalid background uncertainty")
    result = np.zeros_like(s)
    active = (s > 0) & (b > 0)
    ss, bb = s[active], b[active]
    r = ss / bb
    if delta_b == 0:
        small = r < 1e-7
        values = np.empty_like(r)
        values[small] = ss[small] / np.sqrt(bb[small]) * np.sqrt(1-r[small]/3+r[small]**2/6-r[small]**3/10)
        values[~small] = np.sqrt(2*((ss[~small]+bb[~small])*np.log1p(r[~small])-ss[~small]))
    else:
        t = delta_b**2 * bb
        if not np.isfinite(r).all() or not np.isfinite(t).all():
            raise ValueError("Yields/uncertainty exceed numerical range")
        a, u = t / (1+t), 1 / (1+t)
        small = r < .01
        values = np.empty_like(r)
        powers, series = np.zeros(small.sum()), np.zeros(small.sum())
        # Integral series for log[(1+r)/(1+a*r)], factoring 1-a^n
        # as u*(1+a+...+a^(n-1)) to preserve tiny systematic-limited Z.
        for n in range(1, 13):
            powers += a[small] ** (n-1)
            series += (-r[small]) ** (n-1) * powers / (n*(n+1))
        values[small] = ss[small] / np.sqrt(bb[small]) * np.sqrt(2*u[small]*series)
        ordinary = ~small & (t > 0)
        first = (ss[ordinary]+bb[ordinary]) * np.log1p(u[ordinary]*r[ordinary]/(1+a[ordinary]*r[ordinary]))
        second = bb[ordinary] * (np.log1p(a[ordinary]*r[ordinary])/t[ordinary])
        half = first-second
        tolerance = 32*np.finfo(float).eps*np.maximum(abs(first), abs(second))
        if not np.isfinite(half).all() or (half < -tolerance).any():
            raise ValueError("Invalid significance radicand")
        values[ordinary] = np.sqrt(2*np.maximum(0., half))
        underflow = ~small & (t == 0)
        values[underflow] = significance_arrays(ss[underflow], bb[underflow])
    result[active] = values
    return result


def yield_metrics(S: float, B: float) -> dict:
    """Validated expected-count metrics; finite JSON-compatible zero-B state."""
    if not all(np.isfinite(x) and x >= 0 for x in (S, B)):
        raise ValueError("Expected yields must be finite and non-negative")
    return dict(S=float(S), B=float(B), S_over_B=float(S / B) if B > 0 else None,
                Z_Asimov=asimov_z(S, B), Z_syst_5pct=asimov_z_syst(S, B, .05),
                Z_syst_10pct=asimov_z_syst(S, B, .10), valid=B > 0,
                validity_reason="ok" if B > 0 else "zero_background",
                asymptotic_warning=B < 10)


def operating_curve(frame, weights, config, *, thresholds=None, multiplicity=None,
                    score_order=None, include_processes=True):
    """Common exact score grid, cumulative yields and per-process MC support.

    Weights are full-yield evaluation weights, separate from stored/fit weights.
    Multiplicity is for a within-process event bootstrap; repeated observations
    contribute count*w^2, not (count*w)^2, to that replica's sum of squares.
    """
    scores = frame.bdt_score.to_numpy(dtype=float)
    weights = np.asarray(weights, dtype=float)
    y = frame.target.to_numpy()
    if (not len(frame) or len(weights) != len(frame) or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any() or not np.isfinite(weights).all()
            or (weights < 0).any() or set(y) != {0, 1}):
        raise ValueError("Invalid scores, weights or class population")
    if "event_id" in frame and frame.event_id.duplicated().any():
        raise ValueError("Evaluate unique physical events, not hypothesis replicas")
    if thresholds is None:
        thresholds = np.unique(np.r_[0., scores, np.nextafter(scores.max(), np.inf)])
    thresholds = np.asarray(thresholds, dtype=float)
    if not np.isfinite(thresholds).all() or (np.diff(thresholds) <= 0).any():
        raise ValueError("Thresholds must be finite and strictly increasing")
    counts = np.ones(len(frame)) if multiplicity is None else np.asarray(multiplicity, dtype=float)
    if counts.shape != weights.shape or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Invalid bootstrap multiplicities")
    order = np.argsort(scores, kind="stable") if score_order is None else score_order
    positions = np.searchsorted(scores[order], thresholds, side="left")

    def retained(values):
        # Reverse cumulative sums avoid cancellation in tiny tails.
        suffix = np.r_[np.cumsum(np.asarray(values)[order][::-1])[::-1], 0.]
        return suffix[positions]

    values = {"threshold": thresholds}
    process_rows = []
    all_processes_present = np.ones(len(thresholds), dtype=bool)
    for sample, group in frame.groupby("sample", sort=True):
        mask = frame["sample"].to_numpy() == sample
        target = int(group.target.iloc[0])
        if group.target.nunique() != 1:
            raise ValueError("A process mixes signal and background labels")
        n = retained(counts * mask)
        total = retained(weights * counts * mask)
        square = retained(weights**2 * counts * mask)
        if target == 0:
            all_processes_present &= n >= config.min_process_mc
        if include_processes:
            process_rows.append(pd.DataFrame(dict(threshold=thresholds, sample=sample, target=target,
                                                 mc_count=n, sum_weight=total, sum_weight_squared=square,
                                                 N_eff=np.divide(total**2, square, out=np.zeros_like(total), where=square > 0))))
    for label, prefix in ((1, "signal"), (0, "background")):
        mask = y == label
        total = retained(weights * counts * mask)
        square = retained(weights**2 * counts * mask)
        before = float((weights * counts * mask).sum())
        values["S" if label else "B"] = total
        values[f"{prefix}_efficiency"] = total / before if before > 0 else np.zeros(len(thresholds))
        values[f"{prefix}_mc"] = retained(counts * mask)
        values[f"{prefix}_sumw2"] = square
        values[f"{prefix}_neff"] = np.divide(total**2, square, out=np.zeros_like(total), where=square > 0)
    curve = pd.DataFrame(values)
    curve["S_over_B"] = np.divide(curve.S, curve.B, out=np.full(len(curve), np.nan), where=curve.B > 0)
    for objective, delta in zip(OBJECTIVES, (0., .05, .10)):
        curve[objective] = significance_arrays(curve.S, curve.B, delta)
    curve["valid"] = curve.B > 0
    curve["validity_reason"] = np.where(curve.valid, "ok", "zero_background")
    curve["asymptotic_warning"] = curve.B < 10
    curve["all_background_processes_supported"] = all_processes_present
    curve["eligible"] = (curve.valid & all_processes_present
                         & (curve.background_neff >= config.min_background_neff)
                         & (curve.signal_mc >= config.min_signal_mc))
    curve["support_reason"] = np.select(
        [~curve.valid, ~all_processes_present, curve.background_neff < config.min_background_neff,
         curve.signal_mc < config.min_signal_mc],
        ["zero_background", "missing_background_process", "background_neff_below_minimum", "signal_mc_below_minimum"],
        default="ok")
    return curve, pd.concat(process_rows, ignore_index=True) if process_rows else pd.DataFrame()


def select_operating_points(curve, *, plateau_fraction=.02, threshold_source="validation"):
    """Exact argmax on observed scores; smallest threshold wins exact ties."""
    if threshold_source != "validation":
        raise ValueError("Operating points must be selected on validation")
    points = {}
    for objective in OBJECTIVES:
        eligible = curve.loc[curve.eligible & np.isfinite(curve[objective]) & (curve[objective] > 0)]
        if eligible.empty:
            points[objective] = dict(status="no_valid_threshold", objective=objective,
                                     threshold_source=threshold_source, threshold=None, plateau_threshold=None)
            continue
        best = float(eligible[objective].max())
        raw = eligible.loc[eligible[objective] == best].sort_values("threshold").iloc[0]
        plateau = eligible.loc[eligible[objective] >= (1 - plateau_fraction) * best].sort_values("threshold").iloc[0]
        points[objective] = dict(status="valid", objective=objective, threshold_source=threshold_source,
                                 threshold=float(raw.threshold), plateau_threshold=float(plateau.threshold),
                                 metrics=raw.to_dict(), plateau_metrics=plateau.to_dict(), tie_rule="lowest_threshold")
    return points


def split_fracs_by_class(
    df_all: pd.DataFrame,
    df_split: pd.DataFrame,
    y_col: str = "target",
    w_col: str = "sample_weight",
) -> tuple[float, float]:
    """Weighted fraction of the full dataset contained in a split, per class.

    Returns (f_sig, f_bkg). Raises on non-positive totals so a broken
    weight column fails loudly instead of producing infinite scale-ups.
    """
    S_all = float(df_all.loc[df_all[y_col] == 1, w_col].sum())
    B_all = float(df_all.loc[df_all[y_col] == 0, w_col].sum())
    S_sp = float(df_split.loc[df_split[y_col] == 1, w_col].sum())
    B_sp = float(df_split.loc[df_split[y_col] == 0, w_col].sum())

    if S_all <= 0 or B_all <= 0:
        raise ValueError(f"Bad totals: S_all={S_all}, B_all={B_all}")
    if S_sp <= 0 or B_sp <= 0:
        raise ValueError(f"Bad splits: S_sp={S_sp}, B_sp={B_sp}")

    return (S_sp / S_all, B_sp / B_all)


def split_fracs_weighted(
    df_all: pd.DataFrame,
    df_split: pd.DataFrame,
    y_col: str = "target",
    w_col: str = "sample_weight",
) -> dict[str, float]:
    """Dict form of split_fracs_by_class, as persisted in split meta files."""
    f_sig, f_bkg = split_fracs_by_class(df_all, df_split, y_col=y_col, w_col=w_col)
    return {"f_sig": f_sig, "f_bkg": f_bkg}


def safe_auc(y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    from sklearn.metrics import roc_auc_score

    # If only one class present, roc_auc_score errors; return NaN
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))


def sample_normalization(df_all: pd.DataFrame, df_split: pd.DataFrame) -> dict:
    """Normalize each sample to its full *prepared* yield, before any score cut.

    This is evaluation bookkeeping, not a modification of physical or fit weights.
    A missing process cannot be extrapolated and must fail rather than disappear.
    """
    for frame in (df_all, df_split):
        if frame[["sample", "target", "sample_weight"]].isna().any().any():
            raise ValueError("Normalization requires non-null sample, target and sample_weight")
        weight = frame.sample_weight.to_numpy(dtype=float)
        if not np.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("Normalization requires finite non-negative physical weights")
    full = df_all.groupby("sample").sample_weight.sum()
    part = df_split.groupby("sample").sample_weight.sum()
    if set(full.index) != set(part.index):
        raise ValueError("Split must contain every reference sample and no unknown samples")
    samples = {}
    for sample, total in full.items():
        subtotal = float(part[sample])
        targets = pd.concat([
            df_all.loc[df_all["sample"] == sample, "target"],
            df_split.loc[df_split["sample"] == sample, "target"],
        ]).unique()
        if len(targets) != 1 or targets[0] not in (0, 1):
            raise ValueError(f"{sample}: inconsistent binary target")
        if total <= 0 or subtotal <= 0 or subtotal > total * (1 + 1e-12):
            raise ValueError(f"{sample}: invalid full/split weight totals {total}/{subtotal}")
        samples[str(sample)] = dict(
            target=int(targets[0]), full_weight_sum=float(total), split_weight_sum=subtotal,
            fraction=subtotal / float(total), factor=float(total) / subtotal,
        )
    return {"method": "per_sample_full_prepared_v1", "samples": samples}


def normalized_weights(df: pd.DataFrame, normalization: dict) -> np.ndarray:
    """Return full-yield evaluation weights in row order; never mutate the input."""
    if normalization.get("method") != "per_sample_full_prepared_v1":
        raise ValueError("Unknown evaluation normalization method")
    factors = df["sample"].map({k: v["factor"] for k, v in normalization["samples"].items()})
    weights = df.sample_weight.to_numpy(dtype=float) * factors.to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid normalized weights or unknown sample")
    return weights


def threshold_yields(y, scores, weights, *, threshold: float, lumi: float) -> dict:
    """Physics metrics at a supplied threshold using already-normalized weights."""
    selected = np.asarray(scores) >= threshold
    y, weights = np.asarray(y), np.asarray(weights)
    signal = float(weights[selected & (y == 1)].sum())
    background = float(weights[selected & (y == 0)].sum())
    return dict(threshold=float(threshold), signal_yield=signal, background_yield=background,
                signal_xs_pb=signal / lumi, background_xs_pb=background / lumi,
                significance=asimov_z(signal, background))


def weighted_significance_scan(
    y_true: np.ndarray,
    y_score: np.ndarray,
    w_xs: np.ndarray,  # sample_weight = lumi * merged_xs / n_generated
    *,
    lumi: float = 3000.0,
    frac_sig: float = 1.0,
    frac_bkg: float = 1.0,
    n_steps: int = 200,
    min_nS: int = 50,
    min_nB: int = 1000,
    min_B: float = 5.0,
    eps_plateau: float = 0.02,
) -> dict[str, float]:
    """
    Scan score thresholds and evaluate the Asimov Z of each selection.

    Yield definitions (w_xs already contains the lumi factor):
      S (efficiency-corrected xs) = sum(w_xs, selected sig) / lumi / frac_sig
      nS (expected events)        = lumi * S
    and likewise for background. Z is only evaluated where the weighted
    yields satisfy the min_nS/min_nB/min_B constraints. The chosen
    threshold is the lowest one on the Z plateau (within eps_plateau of
    the maximum), which is more stable than a raw argmax.
    """
    qs = np.linspace(0.0, 1.0, n_steps)
    thresholds = np.unique(np.quantile(y_score, qs))
    thresholds = thresholds[(thresholds > 0.0) & (thresholds < 1.0)]
    if len(thresholds) == 0:
        thresholds = np.array([0.5], dtype=float)

    thr_list, Z_list, S_list, B_list, nS_list, nB_list = [], [], [], [], [], []

    # denominators per class (avoid mixing sig/bkg)
    sig = y_true == 1
    bkg = y_true == 0

    for thr in thresholds:
        sel = y_score >= thr

        selS = sel & sig
        selB = sel & bkg

        S = float(w_xs[selS].sum() / lumi / frac_sig)
        B = float(w_xs[selB].sum() / lumi / frac_bkg)

        nS = lumi * S
        nB = lumi * B

        if (nS >= min_nS) and (nB >= min_nB) and (B >= min_B) and (S > 0.0):
            Z = asimov_z(nS, nB)
        else:
            Z = 0.0

        thr_list.append(float(thr))
        Z_list.append(float(Z))
        S_list.append(float(S))
        B_list.append(float(B))
        nS_list.append(nS)
        nB_list.append(nB)

    thr_arr = np.array(thr_list)
    Z_arr = np.array(Z_list)

    # robust choice: plateau threshold
    Zmax = float(Z_arr.max()) if len(Z_arr) else 0.0
    if Zmax <= 0:
        best_idx = int(np.argmax(Z_arr)) if len(Z_arr) else 0
        best_thr = float(thr_arr[best_idx]) if len(thr_arr) else 0.5
        method = "argmax_fallback"
    else:
        good = Z_arr >= (1.0 - eps_plateau) * Zmax
        best_thr = float(thr_arr[good].min())
        method = "plateau"

    # exact best at that chosen threshold
    i_star = int(np.where(thr_arr == best_thr)[0][0])

    return {
        "best_thr": best_thr,
        "best_Z": float(Z_list[i_star]),
        "best_S": float(S_list[i_star]),
        "best_B": float(B_list[i_star]),
        "best_nS": float(nS_list[i_star]),
        "best_nB": float(nB_list[i_star]),
        "method": method,
        "split_frac_sig": frac_sig,
        "split_frac_bkg": frac_bkg,
        "constraints": {"min_nS": min_nS, "min_nB": min_nB, "min_B": min_B, "eps_plateau": eps_plateau},
        "scan": {
            "thr": thr_list,
            "Z": Z_list,
            "S": S_list,
            "B": B_list,
            "nS": nS_list,
            "nB": nB_list,
        },
        # useful for debugging:
        "raw_argmax": {
            "thr": float(thr_arr[int(np.argmax(Z_arr))]),
            "Z": float(Zmax),
        },
    }


def rho_scan(
    score: np.ndarray,
    y: np.ndarray,
    wxs: np.ndarray,
    xs_by_rho: dict,
    xs_ref: float,
    frozen_thr: float,
    f_sig: float = 1.0,
    f_bkg: float = 1.0,
    n_steps: int = 200,
) -> list[dict]:
    """Per-rho_tc significance from pure xs rescaling (no retraining).

    Kinematics are rho-independent, so S(rho) = S_ref * xs(rho)/xs_ref while
    B is unchanged. For each rho this reports Z at the frozen threshold and
    the rho-optimal threshold found by re-scanning the STORED scores
    (unconstrained argmax over full-stat yields).
    """
    thresholds = np.unique(np.quantile(score, np.linspace(0.0, 1.0, n_steps)))
    sig = y == 1
    bkg = y == 0

    S_t = np.array([wxs[(score >= t) & sig].sum() / f_sig for t in thresholds])
    B_t = np.array([wxs[(score >= t) & bkg].sum() / f_bkg for t in thresholds])
    S_frozen = float(wxs[(score >= frozen_thr) & sig].sum() / f_sig)
    B_frozen = float(wxs[(score >= frozen_thr) & bkg].sum() / f_bkg)

    rows = []
    for rho, xs_entry in sorted(xs_by_rho.items(), key=lambda kv: float(kv[0])):
        xs = float(xs_entry[0] if isinstance(xs_entry, (list, tuple)) else xs_entry)
        scale = xs / xs_ref
        z_grid = np.array([asimov_z(s * scale, b) for s, b in zip(S_t, B_t)])
        i_best = int(np.argmax(z_grid))
        rows.append(
            {
                "rho_tc": float(rho),
                "xs_pb": xs,
                "scale": scale,
                "Z_frozen_thr": asimov_z(S_frozen * scale, B_frozen),
                "best_thr": float(thresholds[i_best]),
                "Z_best_thr": float(z_grid[i_best]),
            }
        )
    return rows


def background_efficiency_at(score, target, weight, signal_efficiency):
    """Weighted background efficiency and background N_eff at the loosest score threshold whose
    weighted signal efficiency reaches signal_efficiency (events are accepted by descending score)."""
    score, target, weight = (np.asarray(a) for a in (score, target, weight))
    signal = target == 1
    order = np.argsort(-score, kind="stable")
    reached = np.cumsum(np.where(signal, weight, 0.0)[order]) / weight[signal].sum()
    selected = order[: int(np.searchsorted(reached, signal_efficiency - 1e-12)) + 1]
    background = weight[selected[~signal[selected]]]
    total = background.sum()
    neff = total**2 / (background**2).sum() if len(background) else 0.0
    return float(total / weight[~signal].sum()), float(neff)
