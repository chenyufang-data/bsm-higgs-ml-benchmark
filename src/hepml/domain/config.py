"""Typed analysis settings and sample metadata; no configuration I/O."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScanConfig:
    n_steps: int = 200
    min_nS: int = 50
    min_nB: int = 1000
    min_B: float = 5.0
    eps_plateau: float = 0.02


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    max_depth: int = 4
    n_estimators: int = 1200
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_weight: float = 1.0
    gamma: float = 0.0
    early_stopping_rounds: int = 50
    scan: ScanConfig = field(default_factory=ScanConfig)


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.1
    val_size: float = 0.1
    seed: int = 42


@dataclass(frozen=True)
class EvaluationConfig:
    schema_version: int = 2
    lumi_pb_inv: float = 500000.0
    signal_k_factor: float = 1.0
    background_k_factor: float = 1.0
    background_systematics: tuple[float, ...] = (0.05, 0.10)
    primary_objective: str = "Z_syst_5pct"
    yield_scope: str = "full_expected_per_process"
    threshold_source: str = "validation"
    min_background_neff: float = 100.0
    min_process_mc: int = 1
    min_signal_mc: int = 1
    plateau_fraction: float = 0.02

    def __post_init__(self):
        import math

        if self.schema_version != 2 or not math.isfinite(self.lumi_pb_inv) or self.lumi_pb_inv <= 0:
            raise ValueError("Invalid evaluation schema or luminosity")
        if any(not math.isfinite(k) or k <= 0 for k in (self.signal_k_factor, self.background_k_factor)):
            raise ValueError("Cross-section k-factors must be finite and positive")
        if tuple(self.background_systematics) != (.05, .10):
            raise ValueError("This evaluation schema reports fixed 5% and 10% scenarios")
        if self.primary_objective not in {"Z_Asimov", "Z_syst_5pct", "Z_syst_10pct"}:
            raise ValueError("Unknown primary objective")
        if self.threshold_source != "validation" or self.yield_scope != "full_expected_per_process":
            raise ValueError("New operating points require validation and per-process full yields")
        if (not math.isfinite(self.min_background_neff) or self.min_background_neff < 0
                or self.min_process_mc < 1 or self.min_signal_mc < 1 or not 0 <= self.plateau_fraction < 1):
            raise ValueError("Invalid MC support or plateau policy")


@dataclass(frozen=True)
class AnalysisConfig:
    # Which samples exist (backgrounds, masses) is defined by the sample
    # manifest, not here.
    lumi: float = 3000.0
    training: TrainingConfig = field(default_factory=TrainingConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
