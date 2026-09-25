"""Plots and small cross-validated two-sample models for the coupling audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def joint_diagnostic(left, right, features, config, *, mass, directory, family_size):
    """rho01 vs rho05 discrimination, confined to the development partition.

    Permutations refit the complete CV procedure. OOF score bootstrap bands are
    conditional diagnostics, not a model-uncertainty equivalence certificate.
    """
    from xgboost import XGBClassifier

    rng = np.random.default_rng(config["seed"] + int(mass))
    size = min(len(left), len(right))
    a = left.iloc[rng.choice(len(left), size, replace=False)]
    b = right.iloc[rng.choice(len(right), size, replace=False)]
    frame = pd.concat([a, b], ignore_index=True)
    X = frame[features].to_numpy(dtype=np.float32)
    y = np.repeat([0, 1], size)

    def score(labels):
        scores, folds = np.zeros(len(labels)), np.zeros(len(labels), dtype=int)
        splitter = StratifiedKFold(config["two_sample_folds"], shuffle=True, random_state=config["seed"])
        for fold, (train, val) in enumerate(splitter.split(X, labels)):
            model = XGBClassifier(n_estimators=60, max_depth=2, learning_rate=.08,
                                  objective="binary:logistic", tree_method="hist", n_jobs=2,
                                  random_state=config["seed"], eval_metric="logloss")
            model.fit(X[train], labels[train])
            scores[val] = model.predict_proba(X[val])[:, 1]
            folds[val] = fold
        return scores, folds

    scores, folds = score(y)
    auc = float(roc_auc_score(y, scores))
    null = []
    for permutation in range(config["two_sample_permutations"]):
        labels = rng.permutation(y)
        permuted, _ = score(labels)
        null.append(float(roc_auc_score(labels, permuted)))
    bootstrap = []
    for _ in range(config["bootstrap_replicates"]):
        index = np.concatenate([rng.integers(size, size=size), size + rng.integers(size, size=size)])
        bootstrap.append(roc_auc_score(y[index], scores[index]))
    alpha = config["stage_alpha"] / family_size
    lower, upper = np.quantile(bootstrap, [alpha / 2, 1 - alpha / 2])
    p = (1 + np.sum(np.asarray(null) >= auc)) / (len(null) + 1)
    pd.DataFrame(dict(event_id=frame.event_id, rho=frame.rho_tc, fold=folds,
                      label=y, oof_score=scores)).to_parquet(directory / f"joint_oof_m{mass}.parquet", index=False)
    pd.DataFrame(dict(permutation=np.arange(len(null)), auc=null)).to_csv(
        directory / f"joint_null_m{mass}.csv", index=False)
    return dict(mass=mass, n_per_rho=size, auc=auc, auc_lower=float(lower), auc_upper=float(upper),
                permutation_p=float(p), permutation_p_adjusted=min(1., float(p) * family_size),
                permutations=len(null), uncertainty="conditional OOF score bootstrap; no training uncertainty",
                model="XGBoost 60 trees, depth 2, learning_rate .08, 3-fold CV by default")


def plot_shapes(development, counts, features, config, directory, *, stage):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    histogram_rows, correlation_rows = [], []
    with PdfPages(directory / f"shape_overlays_{stage}.pdf") as overlays, \
            PdfPages(directory / f"shape_ratios_{stage}.pdf") as ratios, \
            PdfPages(directory / f"correlations_{stage}.pdf") as correlations:
        for mass in sorted({m for m, _ in development}):
            rhos = config["couplings"]
            for feature in features:
                combined = np.concatenate([development[mass, rho][feature].to_numpy() for rho in rhos])
                low, high = np.quantile(combined, config["display_quantiles"])
                if low == high:
                    low, high = low - .5, high + .5
                edges = np.linspace(low, high, config["histogram_bins"] + 1)
                mid = (edges[:-1] + edges[1:]) / 2
                histograms, replicas = {}, {}
                for rho in rhos:
                    values = development[mass, rho][feature].to_numpy()
                    bins = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(mid) - 1)
                    histograms[rho] = np.bincount(bins, minlength=len(mid)) / len(values)
                    replicas[rho] = np.asarray([np.bincount(bins, weights=row, minlength=len(mid)) / len(values)
                                                for row in counts[mass, rho]])
                    lo, hi = np.quantile(replicas[rho], [.025, .975], axis=0)
                    for i in range(len(mid)):
                        histogram_rows.append(dict(mass=mass, rho=rho, feature=feature, bin=i,
                                                   left=edges[i], right=edges[i+1], probability=histograms[rho][i],
                                                   lower=lo[i], upper=hi[i],
                                                   underflow=int((values < low).sum()), overflow=int((values > high).sum())))
                fig, ax = plt.subplots(figsize=(8, 4))
                ratio_fig, rax = plt.subplots(figsize=(8, 4))
                reference = histograms[rhos[0]]
                valid = reference > 0
                for rho in rhos:
                    lo, hi = np.quantile(replicas[rho], [.025, .975], axis=0)
                    line, = ax.plot(mid, histograms[rho], label=f"rho={rho:g}")
                    ax.fill_between(mid, lo, hi, color=line.get_color(), alpha=.18)
                    ratio = np.divide(histograms[rho], reference, out=np.full(len(mid), np.nan), where=valid)
                    rax.plot(mid, ratio, label=f"rho={rho:g}")
                    # Undefined reference bins remain missing, including bootstrap zeros.
                    stable = valid & (replicas[rhos[0]] > 0).all(axis=0)
                    if rho != rhos[0] and stable.any():
                        ratios_boot = replicas[rho][:, stable] / replicas[rhos[0]][:, stable]
                        lower, upper = np.quantile(ratios_boot, [.025, .975], axis=0)
                        rax.fill_between(mid[stable], lower, upper, alpha=.15)
                ax.set(title=f"m={mass}: {feature} (development only)", xlabel=feature, ylabel="Probability per bin")
                ax.legend()
                ax.text(.02, .98, "Tails folded into edge bins; pointwise 95% bootstrap bands",
                        transform=ax.transAxes, va="top", fontsize=8)
                rax.axhline(1, color="grey", linestyle="--")
                rax.set(title=f"m={mass}: ratio to rho={rhos[0]:g}; {int((~valid).sum())} undefined bins",
                         xlabel=feature, ylabel="Probability ratio")
                rax.legend()
                fig.tight_layout()
                ratio_fig.tight_layout()
                overlays.savefig(fig)
                ratios.savefig(ratio_fig)
                plt.close(fig)
                plt.close(ratio_fig)
            fig, axes = plt.subplots(1, len(rhos), figsize=(15, 5))
            reference = development[mass, rhos[0]][features].corr()
            for ax, rho in zip(np.atleast_1d(axes), rhos):
                corr = development[mass, rho][features].corr()
                cov = development[mass, rho][features].cov()
                for first in features:
                    for second in features:
                        correlation_rows.append(dict(mass=mass, rho=rho, first=first, second=second,
                                                     covariance=cov.loc[first, second], correlation=corr.loc[first, second],
                                                     correlation_difference=corr.loc[first, second] - reference.loc[first, second]))
                ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
                ax.set(title=f"m={mass}, rho={rho:g}", xticks=range(len(features)), yticks=range(len(features)))
                ax.set_xticklabels(features, rotation=90, fontsize=6)
                ax.set_yticklabels(features, fontsize=6)
            fig.tight_layout()
            correlations.savefig(fig)
            plt.close(fig)
    pd.DataFrame(histogram_rows).to_csv(directory / f"histograms_{stage}.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(directory / f"correlations_{stage}.csv", index=False)
