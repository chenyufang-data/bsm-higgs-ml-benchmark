"""Small contracts used by application workflows; implementations live in adapters.

Study plugins are ordinary Python modules implementing StudyPlugin. Model trainers
are callables, so a new baseline does not need to inherit a framework base class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


class StudyPlugin(Protocol):
    FEATURES: list[str]
    RETAINED_COLUMNS: list[str]

    def derive_features(self, frame: pd.DataFrame) -> pd.DataFrame: ...


@dataclass
class Study:
    directory: Path
    config: dict
    plugin: StudyPlugin
    fingerprint: str

    @property
    def name(self) -> str:
        return self.config["name"]


class ScoreModel(Protocol):
    def predict_proba(self, values: np.ndarray) -> np.ndarray: ...


class ModelTrainer(Protocol):
    def __call__(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, **options: Any
    ) -> ScoreModel: ...
