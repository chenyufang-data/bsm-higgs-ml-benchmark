"""XGBoost training adapter. Shared evaluation lives in domain.metrics."""

from __future__ import annotations

import numpy as np
import xgboost as xgb


def measured_train_xgb(*args, **kwargs):
    """Time a fit and sample whole-process resident memory, including native arrays."""
    import threading
    import time

    import psutil

    process = psutil.Process()
    samples = [process.memory_info().rss]
    done = threading.Event()

    def sample():
        while not done.wait(.02):
            samples.append(process.memory_info().rss)

    thread = threading.Thread(target=sample,daemon=True)
    start = time.perf_counter()
    thread.start()
    try:
        model = train_xgb(*args,**kwargs)
    finally:
        samples.append(process.memory_info().rss)
        done.set()
        thread.join()
    return model, dict(seconds=time.perf_counter()-start, rss_before_bytes=samples[0],
                       peak_sampled_rss_bytes=max(samples), peak_rss_delta_bytes=max(samples)-samples[0],
                       sampling_interval_seconds=.02)


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int = 42,
    max_depth: int = 4,
    n_estimators: int = 1200,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    early_stopping_rounds: int = 50,
    scale_pos_weight: float | None = None,
    tree_method: str = "hist",
    fit_weights_train: np.ndarray | None = None,
    fit_weights_val: np.ndarray | None = None,
    record_training: bool = False,
) -> xgb.XGBClassifier:
    params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        min_child_weight=min_child_weight,
        gamma=gamma,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=seed,
        tree_method=tree_method,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
    )
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)

    model = xgb.XGBClassifier(**params)

    fit_kwargs = {}
    if fit_weights_train is not None:
        fit_kwargs["sample_weight"] = fit_weights_train
        # keep early stopping consistent with the training objective
        fit_kwargs["sample_weight_eval_set"] = ([fit_weights_train, fit_weights_val]
                                                if record_training else [fit_weights_val])

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)] if record_training else [(X_val, y_val)],
        verbose=False,
        **fit_kwargs,
    )
    return model
