"""
Training and inference for the touchdown model.

Both paths share one ``FeaturePipeline`` and one set of eligibility filters, so
the matrix the model is fitted on and the matrix it scores are built by the same
code with the same fitted statistics.
"""

import json
import logging
import os
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import MODEL_PARAMS
from .filters import filter_individual_performance
from .models import MLModel
from .preprocessing import FeaturePipeline, PlattCalibrator

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

IDENTITY_COLUMNS = [
    "season", "week", "player_id", "player_name", "team",
    "against", "report_status", "injury_status", "played", "position",
]


def precision_at_k(y_true, y_prob, k: int) -> float:
    """Share of the top-k ranked players who actually scored.

    This is what the product is judged on: the shortlist is a ranking, so
    ranking quality matters more than accuracy at an arbitrary threshold.
    """
    y_true = np.asarray(y_true, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    k = min(k, len(y_true))
    top = np.argsort(np.asarray(y_prob))[::-1][:k]
    return float(np.mean(y_true[top]))


def evaluate(y_true, y_prob, threshold: float, k: int) -> Dict[str, float]:
    """Threshold metrics plus the ranking metrics that matter here."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        f"precision_at_{k}": precision_at_k(y_true, y_prob, k),
        "base_rate": float(np.mean(y_true)),
    }
    # AUC is undefined when every label is the same.
    metrics["auc"] = (
        roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    )
    return metrics


class NFLTouchdownModel:
    """XGBoost touchdown model for one season and week."""

    def __init__(self, season: int, week: int):
        self.season = season
        self.week = week
        self.model: Optional[xgb.Booster] = None
        self.pipeline: Optional[FeaturePipeline] = None
        self.calibrator: Optional[PlattCalibrator] = None
        self.feature_importance: Dict[str, dict] = {}
        self.metrics: Dict[str, dict] = {}
        self.optimal_threshold: float = MODEL_PARAMS["prediction_threshold"]

    def model_exists(self) -> bool:
        return MLModel.objects.filter(season=self.season, week=self.week).exists()

    # -------------------------------------------------------------- training

    def train(
        self,
        df: pd.DataFrame,
        features: list,
        numeric_features: list,
        save_model: bool = True,
    ) -> Tuple[bool, str]:
        """Fit the model on completed games and store it."""
        try:
            df = self._prepare_training_frame(df)
        except ValueError as error:
            return False, str(error)

        self.pipeline = FeaturePipeline(features, numeric_features)
        X = self.pipeline.fit_transform(df)
        y = df["touchdown"].astype(int).to_numpy()

        splits = self._split(df, X, y)
        if splits is None:
            return False, (
                f"Not enough data to train: {len(df)} rows after filtering, "
                "and every split needs both classes present."
            )
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits

        logger.info(
            "Training on %s rows (val %s, test %s), positive rate %.3f",
            len(y_train), len(y_val), len(y_test), y_train.mean(),
        )

        dtrain, dval, dtest = (
            xgb.DMatrix(features_, label=labels)
            for features_, labels in ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        )

        params = self._tune(dtrain, dval, y_val, y_train)
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=MODEL_PARAMS["num_boost_round"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=MODEL_PARAMS["early_stopping_rounds"],
            verbose_eval=False,
        )

        val_prob = self.model.predict(dval)
        if MODEL_PARAMS.get("enable_platt_scaling", False):
            self.calibrator = PlattCalibrator.fit(val_prob, y_val)
        val_prob = self._calibrate(val_prob)

        self.optimal_threshold = self._choose_threshold(y_val, val_prob)
        k = MODEL_PARAMS["precision_at_k"]
        self.metrics = {
            "validation": evaluate(y_val, val_prob, self.optimal_threshold, k),
            "test": evaluate(
                y_test, self._calibrate(self.model.predict(dtest)), self.optimal_threshold, k
            ),
        }
        self._calculate_feature_importance()

        if save_model:
            self._save(len(df))
            self._export_feature_importance()
        else:
            logger.info("save_model=False: trained for evaluation only, nothing stored")

        test = self.metrics["test"]
        return True, (
            f"Model for Season {self.season}, Week {self.week} trained on {len(df)} rows. "
            f"Test AUC {test['auc']:.3f}, precision@{k} {test[f'precision_at_{k}']:.3f} "
            f"(base rate {test['base_rate']:.3f})."
        )

    def _prepare_training_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restrict training data to labelled rows from the eligible population."""
        for column in ("player_id", "touchdown"):
            if column not in df.columns:
                raise ValueError(f"Missing required column: {column}")

        # Rows at or beyond the prediction week cannot inform it.
        if {"season", "week"} <= set(df.columns):
            before = len(df)
            df = df[
                (df["season"] < self.season)
                | ((df["season"] == self.season) & (df["week"] < self.week))
            ].copy()
            if before != len(df):
                logger.info("Excluded %s rows at or after the target week", before - len(df))

        # An unplayed game has no outcome to learn from.
        df = df[df["touchdown"].notna()].copy()

        # The same eligibility rule the prediction path uses, so the model is
        # fitted on the population it will actually be asked to rank.
        df = filter_individual_performance(df)

        if df.empty:
            raise ValueError("No training rows remain after filtering")
        if df["touchdown"].nunique() < 2:
            raise ValueError("Training data contains only one class")
        return df

    def _split(self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> Optional[tuple]:
        """Split into train, validation and test.

        Chronologically by default. Weekly rows for the same player are not
        independent, so a random split puts a player's later games in training
        and his earlier ones in test, flattering every metric.
        """
        if MODEL_PARAMS.get("temporal_split", True) and {"season", "week"} <= set(df.columns):
            order = np.lexsort((df["week"].to_numpy(), df["season"].to_numpy()))
            X, y = X[order], y[order]
            n = len(y)
            test_start = int(n * (1 - MODEL_PARAMS["test_size"]))
            val_start = int(test_start * (1 - MODEL_PARAMS["val_size"]))
            parts = (
                (X[:val_start], y[:val_start]),
                (X[val_start:test_start], y[val_start:test_start]),
                (X[test_start:], y[test_start:]),
            )
        else:
            from sklearn.model_selection import train_test_split

            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=MODEL_PARAMS["test_size"],
                random_state=MODEL_PARAMS["random_state"],
                stratify=y,
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=MODEL_PARAMS["val_size"],
                random_state=MODEL_PARAMS["random_state"],
                stratify=y_temp,
            )
            parts = ((X_train, y_train), (X_val, y_val), (X_test, y_test))

        if any(len(labels) == 0 or len(np.unique(labels)) < 2 for _, labels in parts):
            return None
        return parts

    def _tune(self, dtrain, dval, y_val, y_train) -> dict:
        """Search hyperparameters, then return the best set."""
        # Weighting the positive class handles the ~20% base rate directly.
        # SMOTE cannot be applied here because the matrix deliberately carries
        # NaN, which XGBoost uses but synthetic interpolation would destroy.
        positives = max(int(np.sum(y_train == 1)), 1)
        scale_pos_weight = float(np.sum(y_train == 0)) / positives

        base = {
            "objective": MODEL_PARAMS["objective"],
            "eval_metric": MODEL_PARAMS["eval_metric"],
            "scale_pos_weight": scale_pos_weight,
        }

        def objective(trial):
            params = dict(
                base,
                max_depth=trial.suggest_int("max_depth", 3, 6),
                min_child_weight=trial.suggest_int("min_child_weight", 2, 10),
                eta=trial.suggest_float("eta", 0.01, 0.15, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1.0, 20.0, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
                subsample=trial.suggest_float("subsample", 0.7, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
                colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.7, 1.0),
            )
            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=MODEL_PARAMS["num_boost_round"],
                evals=[(dval, "val")],
                early_stopping_rounds=MODEL_PARAMS["early_stopping_rounds"],
                verbose_eval=False,
            )
            return self._objective_score(y_val, booster.predict(dval))

        study = optuna.create_study(
            direction="minimize", study_name=f"td_s{self.season}_w{self.week}"
        )
        study.optimize(
            objective,
            n_trials=MODEL_PARAMS["optuna_trials"],
            timeout=MODEL_PARAMS["optuna_timeout"],
        )
        logger.info("Best trial scored %.4f with %s", study.best_value, study.best_params)

        return dict(base, **study.best_trial.params)

    @staticmethod
    def _objective_score(y_true, y_prob) -> float:
        """Lower is better, so ranking and accuracy metrics are negated."""
        metric = MODEL_PARAMS.get("optimization_metric", "logloss")
        if metric == "logloss":
            return log_loss(y_true, y_prob, labels=[0, 1])
        if metric == "auc":
            if len(np.unique(y_true)) < 2:
                return 0.0
            return -roc_auc_score(y_true, y_prob)

        y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
        scorers = {
            "accuracy": accuracy_score,
            "f1": f1_score,
            "precision": lambda t, p: precision_score(t, p, zero_division=0),
            "recall": lambda t, p: recall_score(t, p, zero_division=0),
        }
        scorer = scorers.get(metric)
        if scorer is None:
            return log_loss(y_true, y_prob, labels=[0, 1])
        return -scorer(y_true, y_pred)

    @staticmethod
    def _choose_threshold(y_true, y_prob) -> float:
        """Pick the probability cut-off that maximises F1 on validation."""
        if not MODEL_PARAMS.get("optimize_threshold", False):
            return MODEL_PARAMS["prediction_threshold"]

        _, _, thresholds = roc_curve(y_true, y_prob)
        scores = [
            (f1_score(y_true, (y_prob >= t).astype(int), zero_division=0), t)
            for t in thresholds
            if np.isfinite(t)
        ]
        if not scores:
            return MODEL_PARAMS["prediction_threshold"]
        return float(max(scores)[1])

    def _calibrate(self, probs: np.ndarray) -> np.ndarray:
        return self.calibrator.apply(probs) if self.calibrator else np.asarray(probs)

    # ----------------------------------------------------------- persistence

    def _save(self, training_records: int) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as handle:
            path = handle.name
        try:
            self.model.save_model(path)
            with open(path, "rb") as handle:
                model_bytes = handle.read()
        finally:
            os.unlink(path)

        validation = self.metrics.get("validation", {})
        test = self.metrics.get("test", {})

        MLModel.objects.update_or_create(
            season=self.season,
            week=self.week,
            defaults={
                "model_file": model_bytes,
                # Preprocessing is fully described by the pipeline state; the
                # legacy pickled preprocessor columns are no longer used.
                "imputer_file": b"",
                "scaler_file": b"",
                "encoder_file": b"",
                "pipeline_state": self.pipeline.to_json(),
                "calibrator": json.dumps(
                    self.calibrator.to_dict() if self.calibrator else None
                ),
                "metrics": json.dumps(self.metrics),
                "scaler_means": json.dumps(self.pipeline.means),
                "scaler_stds": json.dumps(self.pipeline.stds),
                "feature_importance": json.dumps(self.feature_importance),
                "training_records": training_records,
                "optimal_threshold": self.optimal_threshold,
                "validation_accuracy": validation.get("accuracy"),
                "validation_f1": validation.get("f1"),
                "validation_auc": validation.get("auc"),
                "test_accuracy": test.get("accuracy"),
                "test_f1": test.get("f1"),
                "test_auc": test.get("auc"),
            },
        )

    def load(self) -> bool:
        """Restore the booster and its preprocessing state."""
        try:
            record = MLModel.objects.get(season=self.season, week=self.week)
        except MLModel.DoesNotExist:
            return False

        self.pipeline = FeaturePipeline.from_json(record.pipeline_state)
        if self.pipeline is None:
            logger.error(
                "Model for %s week %s predates the shared preprocessing pipeline "
                "and cannot be scored safely; retrain it.",
                self.season,
                self.week,
            )
            return False

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as handle:
            path = handle.name
            handle.write(record.model_file)
        try:
            self.model = xgb.Booster()
            self.model.load_model(path)
        finally:
            os.unlink(path)

        self.calibrator = PlattCalibrator.from_dict(_load_json(record.calibrator))
        self.feature_importance = _load_json(record.feature_importance) or {}
        self.metrics = _load_json(record.metrics) or {}
        self.optimal_threshold = (
            float(record.optimal_threshold)
            if record.optimal_threshold is not None
            else MODEL_PARAMS["prediction_threshold"]
        )
        return True

    # ------------------------------------------------------------ inference

    def predict(
        self, current_week: pd.DataFrame, features: list, numeric_features: list,
        include_shap: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[dict]]:
        """Score the supplied players and return them ranked by probability."""
        if not self.load():
            raise ValueError(
                f"No usable model for Season {self.season}, Week {self.week}. Train it first."
            )
        if "player_id" not in current_week.columns:
            raise ValueError("Missing required column: player_id")

        current_week = filter_individual_performance(current_week)
        if current_week.empty:
            raise ValueError("No players remain after the individual performance filter")

        # The pipeline reindexes onto its fitted feature list, so column order
        # and count match training even if the frame differs.
        X = self.pipeline.transform(current_week)
        probabilities = self._calibrate(self.model.predict(xgb.DMatrix(X)))

        output = current_week[
            [c for c in IDENTITY_COLUMNS if c in current_week.columns]
        ].copy()
        output["probability"] = probabilities
        if "touchdown" in current_week.columns:
            output["touchdown"] = current_week["touchdown"]

        # Reported only where the game has been played; before kickoff there is
        # no outcome, and a 0 would read as "did not score".
        if "played" in output.columns:
            output["played"] = np.where(output["played"] == 1, "Y", "N")

        output = output.sort_values("probability", ascending=False).reset_index(drop=True)

        shap_values = self._explain(X, current_week) if include_shap else None
        return output, shap_values

    def _explain(self, X: np.ndarray, current_week: pd.DataFrame) -> Optional[dict]:
        """Per-player SHAP contributions keyed by feature name."""
        if not SHAP_AVAILABLE:
            return None
        try:
            explainer = shap.TreeExplainer(self.model)
            contributions = explainer.shap_values(X)
            base_value = float(np.ravel(explainer.expected_value)[0])
        except Exception:
            logger.exception("SHAP explanation failed; returning predictions without it")
            return None

        # Column j of X is self.pipeline.features[j] by construction, so the
        # names cannot drift out of step with the contributions.
        names = self.pipeline.features
        player_ids = current_week["player_id"].to_numpy()
        missing = np.isnan(X)

        return {
            str(player_id): {
                "contributions": {
                    name: (0.0 if missing[i, j] else float(contributions[i, j]))
                    for j, name in enumerate(names)
                },
                "base_value": base_value,
            }
            for i, player_id in enumerate(player_ids)
        }

    # ---------------------------------------------------------- diagnostics

    def _calculate_feature_importance(self) -> None:
        """Gain, weight and cover per feature, keyed by real feature names."""
        names = self.pipeline.features
        scores = {
            kind: self.model.get_score(importance_type=kind)
            for kind in ("gain", "weight", "cover")
        }
        self.feature_importance = {
            name: {kind: scores[kind].get(f"f{index}", 0.0) for kind in scores}
            for index, name in enumerate(names)
        }

    def _export_feature_importance(self) -> None:
        """Write the gain/weight/cover table alongside the project by default.

        The destination is a setting so tests can redirect it; training used to
        drop a CSV into the repository root on every run, including from the
        test suite.
        """
        if not self.feature_importance:
            return
        from django.conf import settings

        directory = getattr(settings, "FEATURE_IMPORTANCE_DIR", None)
        if directory is None:
            return

        frame = pd.DataFrame(
            [{"feature": name, **values} for name, values in self.feature_importance.items()]
        ).sort_values("gain", ascending=False)

        os.makedirs(directory, exist_ok=True)
        path = os.path.join(
            str(directory), f"feature_importance_s{self.season}_w{self.week}.csv"
        )
        frame.to_csv(path, index=False)
        logger.info("Wrote feature importance to %s", path)


def _load_json(payload: Optional[str]):
    if not payload:
        return None
    try:
        return json.loads(payload)
    except (TypeError, ValueError):
        return None


def train_model(
    df: pd.DataFrame, season: int, week: int, features: list, numeric_features: list
) -> Tuple[bool, str]:
    return NFLTouchdownModel(season, week).train(df, features, numeric_features)


def predict_week(
    season: int,
    week: int,
    current_week: pd.DataFrame,
    features: list,
    numeric_features: list,
    include_shap: bool = True,
) -> Tuple[pd.DataFrame, Optional[dict]]:
    return NFLTouchdownModel(season, week).predict(
        current_week, features, numeric_features, include_shap=include_shap
    )
