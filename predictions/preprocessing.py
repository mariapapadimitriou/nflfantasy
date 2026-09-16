"""
Feature preprocessing shared by training and inference.

Training and prediction MUST see identical column order and identical scaling,
otherwise the served model silently operates on a different feature space than
the one it was fitted on. Both paths therefore go through a single
``FeaturePipeline``: ``fit_transform`` during training, ``transform`` at
prediction time, with the fitted statistics round-tripped through the database.

Missing values are left as NaN. XGBoost learns a default split direction for
them, which is strictly more informative than substituting a mean and pretending
the value was observed.
"""

import json
import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Selects, orders and standardizes model features.

    The column list is fixed at fit time. ``transform`` reindexes onto that
    exact list, so a feature missing from the prediction frame becomes an
    all-NaN column rather than shifting every downstream column left by one.
    """

    def __init__(self, features: Sequence[str], numeric_features: Sequence[str]):
        self.features: List[str] = list(features)
        # Preserve the caller's feature ordering rather than the numeric list's.
        numeric = set(numeric_features)
        self.numeric_features: List[str] = [f for f in self.features if f in numeric]
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}

    # ------------------------------------------------------------------ fit

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Capture per-feature mean and std from the raw, untransformed frame."""
        X = self._select(df)
        for feature in self.numeric_features:
            column = X[feature]
            mean = column.mean()
            std = column.std()
            # A constant or entirely-missing column is centered only, never divided.
            self.means[feature] = float(mean) if pd.notna(mean) else 0.0
            self.stds[feature] = float(std) if pd.notna(std) and std > 0 else 1.0
        return self

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------ transform

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return the model matrix as float64 with NaN preserved."""
        if not self.means and not self.stds:
            raise ValueError("FeaturePipeline.transform called before fit")

        X = self._select(df)
        for feature in self.numeric_features:
            mean = self.means.get(feature, 0.0)
            std = self.stds.get(feature, 1.0) or 1.0
            X[feature] = (X[feature] - mean) / std
        return X.to_numpy(dtype=np.float64)

    def _select(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex onto the fitted feature list, coercing everything numeric."""
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            logger.warning(
                "Features absent from frame, filled with NaN: %s", ", ".join(missing)
            )
        X = df.reindex(columns=self.features).copy()
        for feature in self.features:
            X[feature] = pd.to_numeric(X[feature], errors="coerce")
        return X

    # ------------------------------------------------------------ serialize

    def to_json(self) -> str:
        return json.dumps(
            {
                "features": self.features,
                "numeric_features": self.numeric_features,
                "means": self.means,
                "stds": self.stds,
            }
        )

    @classmethod
    def from_json(cls, payload: Optional[str]) -> Optional["FeaturePipeline"]:
        if not payload:
            return None
        try:
            state = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Could not decode stored FeaturePipeline state")
            return None

        pipeline = cls(state.get("features", []), state.get("numeric_features", []))
        pipeline.means = {k: float(v) for k, v in state.get("means", {}).items()}
        pipeline.stds = {k: float(v) or 1.0 for k, v in state.get("stds", {}).items()}
        return pipeline


class PlattCalibrator:
    """Logistic recalibration of raw model scores into usable probabilities."""

    def __init__(self, coef: float = 1.0, intercept: float = 0.0):
        self.coef = float(coef)
        self.intercept = float(intercept)

    @classmethod
    def fit(cls, probs, labels) -> Optional["PlattCalibrator"]:
        from sklearn.linear_model import LogisticRegression

        probs = np.asarray(probs, dtype=np.float64).reshape(-1, 1)
        labels = np.asarray(labels)
        if len(np.unique(labels)) < 2:
            logger.warning("Skipping calibration: validation set has a single class")
            return None

        model = LogisticRegression(solver="lbfgs", class_weight="balanced")
        model.fit(probs, labels)
        return cls(model.coef_.ravel()[0], model.intercept_[0])

    def apply(self, probs) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        logits = np.clip(self.coef * probs + self.intercept, -50, 50)
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict:
        return {"coef": self.coef, "intercept": self.intercept}

    @classmethod
    def from_dict(cls, payload: Optional[dict]) -> Optional["PlattCalibrator"]:
        if not payload:
            return None
        return cls(payload.get("coef", 1.0), payload.get("intercept", 0.0))
