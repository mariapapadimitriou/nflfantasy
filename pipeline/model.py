"""
Training and inference.

Training and prediction share one ``FeaturePipeline``, so the matrix the model
is fitted on and the matrix it scores are built by the same code with the same
fitted statistics. Keeping two copies of that logic is how a served model
quietly stops being the model that was validated.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score, roc_curve,
)

from .config import FEATURES, MIN_QB_CARRIES_EWMA, MIN_TOUCHES_EWMA, MODEL_PARAMS

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class FeaturePipeline:
    """Selects, orders and standardizes features.

    ``transform`` reindexes onto the feature list fixed at fit time, so a column
    missing from the prediction frame becomes all-NaN rather than shifting every
    later column one position to the left.
    """

    def __init__(self, features: List[str]):
        self.features = list(features)
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        # Measured on the raw frame, before any transform is applied to it.
        # Taking them afterwards yields mean 0 and std 1 for every column, and
        # prediction then silently stops scaling at all.
        X = self._select(df)
        for feature in self.features:
            mean, std = X[feature].mean(), X[feature].std()
            self.means[feature] = float(mean) if pd.notna(mean) else 0.0
            self.stds[feature] = float(std) if pd.notna(std) and std > 0 else 1.0
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._select(df)
        for feature in self.features:
            X[feature] = (X[feature] - self.means[feature]) / self.stds[feature]
        return X.to_numpy(dtype=np.float64)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def _select(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            logger.warning("Features absent, filled with NaN: %s", ", ".join(missing))
        X = df.reindex(columns=self.features).copy()
        for feature in self.features:
            X[feature] = pd.to_numeric(X[feature], errors="coerce")
        return X


class Calibrator:
    """Maps raw scores onto real-world frequencies."""

    def __init__(self, coef: float, intercept: float):
        self.coef, self.intercept = float(coef), float(intercept)

    @classmethod
    def fit(cls, probs, labels) -> Optional["Calibrator"]:
        if len(np.unique(labels)) < 2:
            return None
        # Deliberately unweighted. scale_pos_weight has already pushed the raw
        # scores towards a balanced distribution; fitting this with balanced
        # class weights too leaves the output balanced rather than calibrated,
        # inflating every probability by roughly 2.5x.
        model = LogisticRegression(solver="lbfgs")
        model.fit(np.asarray(probs, dtype=float).reshape(-1, 1), labels)
        return cls(model.coef_.ravel()[0], model.intercept_[0])

    def apply(self, probs) -> np.ndarray:
        logits = np.clip(self.coef * np.asarray(probs, dtype=float) + self.intercept, -50, 50)
        return 1.0 / (1.0 + np.exp(-logits))


def precision_at_k(y_true, y_prob, k: int) -> float:
    """Share of the top-k ranked players who actually scored."""
    y_true = np.asarray(y_true, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    top = np.argsort(np.asarray(y_prob))[::-1][: min(k, len(y_true))]
    return float(np.mean(y_true[top]))


def evaluate(y_true, y_prob, threshold: float, k: int) -> Dict[str, float]:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "precision_at_k": precision_at_k(y_true, y_prob, k),
        "base_rate": float(np.mean(y_true)),
        "k": k,
    }


def eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Players with enough recent usage to be credible scorers.

    Applied identically to training and prediction, so the model is fitted on
    the population it will actually be asked to rank. Quarterbacks are judged on
    rushing volume, since a passing touchdown does not count for the thrower.
    """
    def above(column: str, minimum: float) -> pd.Series:
        if column not in df.columns:
            return pd.Series(False, index=df.index)
        return df[column].notna() & (df[column] >= minimum)

    is_qb = df["position"] == "QB"
    red_zone = above("red_zone_touches_ewma", 0.01)
    skill = ~is_qb & (above("touches_ewma", MIN_TOUCHES_EWMA) | red_zone)
    quarterback = is_qb & (above("carries_ewma", MIN_QB_CARRIES_EWMA) | red_zone)
    return df[skill | quarterback].copy()


class TouchdownModel:
    """Gradient-boosted touchdown model for one week."""

    def __init__(self):
        self.booster: Optional[xgb.Booster] = None
        self.pipeline: Optional[FeaturePipeline] = None
        self.calibrator: Optional[Calibrator] = None
        self.threshold = 0.5
        self.metrics: Dict[str, dict] = {}
        self.importance: Dict[str, float] = {}

    def train(self, history: pd.DataFrame) -> "TouchdownModel":
        df = eligible(history[history["touchdown"].notna()])
        if df["touchdown"].nunique() < 2:
            raise ValueError("Training data contains only one class")

        self.pipeline = FeaturePipeline(FEATURES)
        X = self.pipeline.fit_transform(df)
        y = df["touchdown"].astype(int).to_numpy()

        # Chronological: weekly rows for one player are not independent.
        order = np.lexsort((df["week"].to_numpy(), df["season"].to_numpy()))
        X, y = X[order], y[order]
        n = len(y)
        test_start = int(n * (1 - MODEL_PARAMS["test_size"]))
        val_start = int(test_start * (1 - MODEL_PARAMS["val_size"]))

        splits = {
            "train": (X[:val_start], y[:val_start]),
            "val": (X[val_start:test_start], y[val_start:test_start]),
            "test": (X[test_start:], y[test_start:]),
        }
        matrices = {k: xgb.DMatrix(a, label=b) for k, (a, b) in splits.items()}
        _, y_val = splits["val"]
        _, y_test = splits["test"]

        logger.info(
            "Training on %s rows (val %s, test %s)",
            len(splits["train"][1]), len(y_val), len(y_test),
        )

        params = self._tune(matrices, splits)
        self.booster = xgb.train(
            params,
            matrices["train"],
            num_boost_round=MODEL_PARAMS["num_boost_round"],
            evals=[(matrices["val"], "val")],
            early_stopping_rounds=MODEL_PARAMS["early_stopping_rounds"],
            verbose_eval=False,
        )

        val_prob = self.booster.predict(matrices["val"])
        self.calibrator = Calibrator.fit(val_prob, y_val)
        val_prob = self._calibrate(val_prob)
        self.threshold = self._best_threshold(y_val, val_prob)

        k = MODEL_PARAMS["precision_at_k"]
        self.metrics = {
            "validation": evaluate(y_val, val_prob, self.threshold, k),
            "test": evaluate(
                y_test, self._calibrate(self.booster.predict(matrices["test"])), self.threshold, k
            ),
        }
        self.metrics["training_rows"] = int(n)

        gains = self.booster.get_score(importance_type="gain")
        total = sum(gains.values()) or 1.0
        self.importance = {
            name: gains.get(f"f{i}", 0.0) / total
            for i, name in enumerate(self.pipeline.features)
        }
        return self

    def _tune(self, matrices: dict, splits: dict) -> dict:
        _, y_train = splits["train"]
        _, y_val = splits["val"]
        # Weighting the positive class handles the ~19% base rate directly, and
        # unlike resampling it tolerates the NaN the matrix deliberately carries.
        base = {
            "objective": MODEL_PARAMS["objective"],
            "eval_metric": MODEL_PARAMS["eval_metric"],
            "scale_pos_weight": float(np.sum(y_train == 0)) / max(int(np.sum(y_train == 1)), 1),
        }

        def objective(trial):
            params = dict(
                base,
                max_depth=trial.suggest_int("max_depth", 3, 6),
                min_child_weight=trial.suggest_int("min_child_weight", 2, 12),
                eta=trial.suggest_float("eta", 0.01, 0.15, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1.0, 20.0, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
                subsample=trial.suggest_float("subsample", 0.7, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            )
            booster = xgb.train(
                params, matrices["train"],
                num_boost_round=MODEL_PARAMS["num_boost_round"],
                evals=[(matrices["val"], "val")],
                early_stopping_rounds=MODEL_PARAMS["early_stopping_rounds"],
                verbose_eval=False,
            )
            probs = booster.predict(matrices["val"])
            # Optimising AUC directly: the product ranks a shortlist, so
            # ordering is what matters, not accuracy at an arbitrary cut-off.
            return -roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else 0.0

        study = optuna.create_study(direction="minimize")
        study.optimize(
            objective,
            n_trials=MODEL_PARAMS["optuna_trials"],
            timeout=MODEL_PARAMS["optuna_timeout"],
        )
        logger.info("Best validation AUC %.4f", -study.best_value)
        return dict(base, **study.best_trial.params)

    @staticmethod
    def _best_threshold(y_true, y_prob) -> float:
        _, _, thresholds = roc_curve(y_true, y_prob)
        scored = [
            (f1_score(y_true, (y_prob >= t).astype(int), zero_division=0), float(t))
            for t in thresholds if np.isfinite(t)
        ]
        return max(scored)[1] if scored else 0.5

    def _calibrate(self, probs) -> np.ndarray:
        return self.calibrator.apply(probs) if self.calibrator else np.asarray(probs)

    def predict(self, upcoming: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Score a week, returning players ranked by probability."""
        df = eligible(upcoming).reset_index(drop=True)
        if df.empty:
            raise ValueError("No eligible players to score")

        X = self.pipeline.transform(df)
        df["probability"] = self._calibrate(self.booster.predict(xgb.DMatrix(X)))
        return df.sort_values("probability", ascending=False).reset_index(drop=True), X

    def explain(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Per-player SHAP contributions, aligned to the pipeline's features."""
        try:
            import shap

            values = shap.TreeExplainer(self.booster).shap_values(X)
            return np.asarray(values)
        except Exception:
            logger.warning("SHAP unavailable; shipping without explanations", exc_info=True)
            return None
