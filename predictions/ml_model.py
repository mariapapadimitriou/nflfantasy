"""
Model Training and Prediction Module
"""

import logging
import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import optuna
from typing import Tuple, Optional, List
import io
import tempfile
import json
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.getLogger(__name__).warning("SHAP not available. Feature explanations disabled.")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    log_loss, classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression

from .config import REPORT_STATUS_ORDER, MODEL_PARAMS, POSITIONS, CATEGORICAL_FEATURES
from .models import MLModel


logger = logging.getLogger(__name__)


class NFLTouchdownModel:
    """XGBoost model for NFL touchdown prediction"""

    def __init__(self, season: int, week: int):
        self.season = season
        self.week = week

        self.imputer = None
        self.scaler = None
        self.encoder = None
        self.model = None
        self.features_with_missing = []  # Features that should keep NaN for inapplicable positions
        self.scaler_means = {}  # Store scaling means for each feature
        self.scaler_stds = {}  # Store scaling stds for each feature
        self.feature_importance = {}  # Store feature importance scores
        self.feature_names = []  # Store feature names in order
        self.calibrator_params: Optional[dict] = None  # Platt scaling parameters (coef/intercept)

    def model_exists(self) -> bool:
        """Check if model for this season/week exists in database"""
        return MLModel.objects.filter(season=self.season, week=self.week).exists()

    def train(
        self,
        df: pd.DataFrame,
        features: list,
        numeric_features: list,
        save_model: bool = True,
        comparison_label: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Train the model on historical data

        Args:
            df: Training dataframe
            features: List of feature columns
            numeric_features: List of numeric feature columns

        Returns:
            Tuple of (success, message)
        """
        self.calibrator_params = None

        try:
            
            # CRITICAL: Ensure no data leakage - exclude current week from training data
            # This prevents the model from seeing future data during training
            if "season" in df.columns and "week" in df.columns:
                before_count = len(df)
                # Exclude current week: (season < self.season) OR (season == self.season AND week < self.week)
                df = df[
                    (df["season"] < self.season) | 
                    ((df["season"] == self.season) & (df["week"] < self.week))
                ].copy()
                after_count = len(df)
                if before_count == after_count and before_count > 0:
                    # Check if any records match current week (shouldn't happen, but verify)
                    current_week_records = df[(df["season"] == self.season) & (df["week"] == self.week)]
                    if len(current_week_records) > 0:
                        return (False, f"Data leakage detected: {len(current_week_records)} records from current week ({self.season}, week {self.week}) found in training data")
            
            # Validate required columns
            if "player_id" not in df.columns:
                return (False, "Missing required column: player_id")
            if "touchdown" not in df.columns:
                return (False, "Missing required column: touchdown")
            
            # Filter out players with low individual performance (same filter as prediction)
            # This ensures the model doesn't learn patterns from players who only have high probabilities
            # due to team stats rather than individual performance
            if MODEL_PARAMS.get("require_individual_performance", True):
                min_perf_threshold = MODEL_PARAMS.get("min_individual_performance_threshold", 0.0)
                before_count = len(df)
                
                # Key individual player performance features (non-normalized raw values)
                individual_features = [
                    "touches_ewma",
                    "total_yards_ewma",
                    "total_touchdowns_ewma",
                    "red_zone_touches_ewma",
                ]
                
                # Check which features exist in the dataframe
                available_individual_features = [f for f in individual_features if f in df.columns]
                
                if available_individual_features:
                    # Check if at least one individual feature is above threshold
                    has_individual_performance = None
                    for feature in available_individual_features:
                        feature_values = df[feature]
                        # Check if feature is not NaN and >= threshold
                        feature_mask = feature_values.notna() & (feature_values >= min_perf_threshold)
                        if has_individual_performance is None:
                            has_individual_performance = feature_mask
                        else:
                            has_individual_performance = has_individual_performance | feature_mask
                    
                    # Additional check: Require at least 2 individual features to be above threshold
                    # This prevents players with only one barely-above-threshold feature from passing
                    feature_count_above_threshold = pd.Series(0, index=df.index)
                    for feature in available_individual_features:
                        feature_values = df[feature]
                        feature_mask = feature_values.notna() & (feature_values >= min_perf_threshold)
                        feature_count_above_threshold += feature_mask.astype(int)
                    
                    # Require at least 2 features above threshold (or at least 1 if only 1-2 features available)
                    min_features_required = min(2, len(available_individual_features))
                    has_multiple_features = feature_count_above_threshold >= min_features_required
                    has_individual_performance = has_individual_performance & has_multiple_features
                    
                    # Filter to keep only players with meaningful individual performance
                    df = df[has_individual_performance].copy()
                    
                    after_count = len(df)
            
            # Filter features to only those that exist in the dataframe
            available_features = [f for f in features if f in df.columns]
            missing_features = [f for f in features if f not in df.columns]
            
            
            if not available_features:
                return (
                    False,
                    f"No features available. Missing all {len(features)} features. Available columns: {list(df.columns)[:20]}",
                )
            
            # Prepare data
            # Include position for imputation but exclude it from features
            # Position is needed for position-aware imputation but is NOT a model feature
            features_to_select = available_features.copy()
            if "position" in df.columns and "position" not in features_to_select:
                # Add position temporarily for imputation, will be removed before encoding
                features_to_select.append("position")
            
            X = df[["player_id"] + features_to_select].copy()
            y = df["touchdown"].copy()
            player_ids = X["player_id"].values
            X = X.drop("player_id", axis=1)
            
            # Store position column for imputation, then remove from X if it's not a feature
            position_col_for_imputation = None
            if "position" in X.columns and "position" not in available_features:
                position_col_for_imputation = X["position"].copy()
                X = X.drop("position", axis=1)
            
            # Update numeric_features to only include those that exist
            available_numeric_features = [f for f in numeric_features if f in X.columns]

            # Setup preprocessors
            # Identify categorical features that exist in the dataframe
            categorical_features_to_encode = []
            ordinal_categories = []
            
            # Check for report_status (ordinal)
            if "report_status" in X.columns:
                categorical_features_to_encode.append("report_status")
                ordinal_categories.append(REPORT_STATUS_ORDER)
            
            # Note: defense categorical feature removed - high cardinality causes overfitting
            # Defense quality is captured by defensive EWMA features instead
            
            # Note: position is still used internally for position-aware imputation but is NOT a feature

            # Position-aware imputation strategy
            # For features that don't apply to a position (e.g., receiving stats for RBs), keep NaN
            # XGBoost can handle missing values natively, which is better than imputing with 0
            # This allows the model to learn that missing = feature doesn't apply
            
            # Define position-specific feature groups based on feature specifications
            # These features should be NaN for inapplicable positions (not 0)
            # - WR/TE receiving features: applicable to WR/TE, NaN for RB/QB
            # - RB/QB rushing features: applicable to RB/QB, NaN for WR/TE
            # - WR/TE rushing features: NaN for WR/TE (rushing is less significant)
            # - RB receiving features: NaN for RB (receiving is less significant)
            # - QB receiving features: NaN for QB (QBs never catch)
            
            # WR/TE receiving features (most significant for WR/TE)
            wr_te_receiving_features = [
                "targets_ewma",
                "receptions_ewma", 
                "receiving_yards_ewma",
                "receiving_touchdowns_ewma",
            ]
            
            # RB/QB rushing features (most significant for RB/QB)
            rb_qb_rushing_features = [
                "carries_ewma",
                "rushing_yards_ewma",
                "rushing_touchdowns_ewma",
            ]
            
            # Legacy position groups (kept for compatibility)
            wr_te_features = []  # No WR/TE-only features needed
            rb_only_features = []  # No RB-only features needed
            rb_qb_features = ["carries_ewma"]  # RB/QB rushing
            qb_only_features = []  # No QB-only features needed
            
            # General features (apply to all positions) - these should be imputed normally
            # touches_ewma, red_zone_touches_ewma, red_zone_touch_share_ewma, team context, etc.
            
            # Ensure we're not including categorical features in numeric processing
            numeric_only_features = [f for f in available_numeric_features if f not in categorical_features_to_encode]
            
            # Validate numeric features exist
            if numeric_only_features:
                # Create a copy to work with
                X_imputed = X[numeric_only_features].copy()
                
                # Get position column if available (for imputation, not as a feature)
                position_col = position_col_for_imputation if position_col_for_imputation is not None else None
                if position_col is None and "position" in df.columns:
                    # Fallback: try to get from original df
                    position_col = df["position"]
                
                # Track which features should keep NaN for inapplicable positions
                features_to_keep_nan = set()
                
                # For each feature, impute only for positions where it's applicable
                for feature in numeric_only_features:
                    if feature in X_imputed.columns:
                        # Determine which positions this feature applies to
                        applicable_positions = None
                        
                        # WR/TE receiving features (most significant for WR/TE, NaN for RB/QB)
                        if feature in wr_te_receiving_features:
                            applicable_positions = ["WR", "TE"]
                            features_to_keep_nan.add(feature)
                        # RB/QB rushing features (most significant for RB/QB, NaN for WR/TE)
                        elif feature in rb_qb_rushing_features:
                            applicable_positions = ["RB", "QB"]
                            features_to_keep_nan.add(feature)
                        # Legacy position groups (for backward compatibility)
                        elif feature in wr_te_features:
                            applicable_positions = ["WR", "TE"]
                            features_to_keep_nan.add(feature)
                        elif feature in rb_only_features:
                            applicable_positions = ["RB"]
                            features_to_keep_nan.add(feature)
                        elif feature in rb_qb_features:
                            applicable_positions = ["RB", "QB"]
                            features_to_keep_nan.add(feature)
                        elif feature in qb_only_features:
                            applicable_positions = ["QB"]
                            features_to_keep_nan.add(feature)
                        
                        if applicable_positions and position_col is not None:
                            # Only impute for applicable positions, keep NaN for others
                            mask = position_col.isin(applicable_positions)
                            if mask.any():
                                # Calculate mean only for applicable positions
                                mean_val = X_imputed.loc[mask, feature].mean()
                                if pd.notna(mean_val):
                                    X_imputed.loc[mask & X_imputed[feature].isna(), feature] = mean_val
                                else:
                                    # If no valid values, use 0 for applicable positions
                                    X_imputed.loc[mask & X_imputed[feature].isna(), feature] = 0
                            # For non-applicable positions, keep NaN (XGBoost will handle it)
                        else:
                            # General feature (team, defense, touches, etc.) - use mean imputation for all
                            mean_val = X_imputed[feature].mean()
                            if pd.notna(mean_val):
                                X_imputed[feature] = X_imputed[feature].fillna(mean_val)
                            else:
                                X_imputed[feature] = X_imputed[feature].fillna(0)
                
                # Replace with imputed values
                X[numeric_only_features] = X_imputed
                
                # Store which features should keep NaN for prediction
                self.features_with_missing = list(features_to_keep_nan)
                
                # Store imputation strategy for later use
                self.imputer = SimpleImputer(strategy="mean")  # Keep for compatibility
                self.scaler = StandardScaler()

                # Scale the data - StandardScaler can handle NaN (it ignores them during fit, but we need to handle them)
                # For scaling, we'll use a custom approach that handles NaN
                for feature in numeric_only_features:
                    if feature in X.columns:
                        # Calculate mean and std only on non-NaN values
                        mean_val = X[feature].mean()
                        std_val = X[feature].std()
                        if pd.notna(std_val) and std_val > 0:
                            X[feature] = (X[feature] - mean_val) / std_val
                        elif pd.notna(mean_val):
                            # If std is 0 or NaN, just center it
                            X[feature] = X[feature] - mean_val

                # Store scaling parameters for later
                self.scaler_means = {feat: X[feat].mean() for feat in numeric_only_features if feat in X.columns}
                self.scaler_stds = {feat: X[feat].std() for feat in numeric_only_features if feat in X.columns}

            # Create encoder with all categorical features
            if categorical_features_to_encode:
                transformers = []
                for i, cat_feature in enumerate(categorical_features_to_encode):
                    encoder = OrdinalEncoder(categories=[ordinal_categories[i]])
                    transformers.append(
                        (
                            f"ordinal_{cat_feature}",
                            encoder,
                            [cat_feature],
                        )
                    )
                self.encoder = ColumnTransformer(
                    transformers=transformers,
                    remainder="passthrough",
                )
            else:
                # No categorical features, just use passthrough
                self.encoder = ColumnTransformer(
                    transformers=[],
                    remainder="passthrough",
                )

            X_encoded = self.encoder.fit_transform(X)

            # Split data into train/validation/test (3-way split)
            # First split: separate test set
            X_temp, X_test, y_temp, y_test, player_ids_temp, player_ids_test = train_test_split(
                X_encoded,
                y,
                player_ids,
                test_size=MODEL_PARAMS["test_size"],
                random_state=MODEL_PARAMS["random_state"],
            )
            
            # Second split: separate train and validation from remaining data
            val_size_adjusted = MODEL_PARAMS["val_size"] / (1 - MODEL_PARAMS["test_size"])
            X_train, X_val, y_train, y_val, player_ids_train, player_ids_val = train_test_split(
                X_temp,
                y_temp,
                player_ids_temp,
                test_size=val_size_adjusted,
                random_state=MODEL_PARAMS["random_state"],
            )
            
            # Handle NaN values before SMOTE (SMOTE doesn't accept NaN)
            # XGBoost can handle NaN natively, so we'll temporarily replace NaN with a sentinel value
            # then restore NaN after SMOTE but before XGBoost
            NAN_SENTINEL = -999999.0  # Large negative value that XGBoost will treat as missing
            
            # Convert to numpy array for easier NaN handling
            X_train_np = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
            X_test_np = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
            
            # Replace NaN with sentinel value before SMOTE
            X_train_np = np.where(np.isnan(X_train_np), NAN_SENTINEL, X_train_np)

            # Apply SMOTE
            smote = SMOTE(random_state=MODEL_PARAMS["random_state"])
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_np, y_train)
            
            # Restore NaN values after SMOTE (for XGBoost to handle natively)
            # Convert sentinel back to NaN
            X_train_resampled = np.where(X_train_resampled == NAN_SENTINEL, np.nan, X_train_resampled)
            
            # Also handle validation and test sets NaN with sentinel for XGBoost
            X_val_np = np.array(X_val) if not isinstance(X_val, np.ndarray) else X_val
            X_test_np = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
            X_val_np = np.where(np.isnan(X_val_np), NAN_SENTINEL, X_val_np)
            X_test_np = np.where(np.isnan(X_test_np), NAN_SENTINEL, X_test_np)

            # Create DMatrix (XGBoost will handle NaN natively)
            # Note: We use the sentinel value which XGBoost treats as missing
            dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled, enable_categorical=False)
            dval = xgb.DMatrix(X_val_np, label=y_val, enable_categorical=False)
            dtest = xgb.DMatrix(X_test_np, label=y_test, enable_categorical=False)
            
            # Store validation set for threshold optimization
            self.X_val = X_val_np
            self.y_val = y_val

            # Hyperparameter optimization with Optuna

            def objective(trial):
                param = {
                    "objective": MODEL_PARAMS["objective"],
                    "eval_metric": MODEL_PARAMS["eval_metric"],
                    # Tree structure - more conservative to prevent overfitting
                    "max_depth": trial.suggest_int("max_depth", 3, 6),  # Further reduced (3-6 instead of 3-8)
                    "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),  # Increased minimum
                    # Learning rate and regularization - stronger regularization
                    "eta": trial.suggest_float("eta", 0.01, 0.15, log=True),  # Lower max learning rate
                    "lambda": trial.suggest_float("lambda", 1.0, 20.0, log=True),  # Higher L2 regularization range
                    "alpha": trial.suggest_float("alpha", 0.1, 10.0, log=True),  # Higher L1 regularization range
                    # Sampling for regularization
                    "subsample": trial.suggest_float("subsample", 0.7, 1.0),  # Row sampling
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),  # Column sampling
                    "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 1.0),  # Per-level sampling
                    # Class imbalance
                    "scale_pos_weight": len(y_train_resampled[y_train_resampled == 0])
                    / len(y_train_resampled[y_train_resampled == 1]),
                }

                # Train with early stopping on validation set
                evals = [(dtrain, "train"), (dval, "val")]
                model = xgb.train(
                    param,
                    dtrain,
                    num_boost_round=MODEL_PARAMS["num_boost_round"],
                    evals=evals,
                    early_stopping_rounds=MODEL_PARAMS["early_stopping_rounds"],
                    verbose_eval=False,
                )
                
                # Get predictions on validation set
                y_pred_prob = model.predict(dval)
                
                # Calculate metric based on config
                optimization_metric = MODEL_PARAMS.get("optimization_metric", "logloss")
                if optimization_metric == "logloss":
                    score = log_loss(y_val, y_pred_prob)
                elif optimization_metric == "accuracy":
                    y_pred_binary = (y_pred_prob > 0.5).astype(int)
                    score = -accuracy_score(y_val, y_pred_binary)  # Negative for minimization
                elif optimization_metric == "f1":
                    y_pred_binary = (y_pred_prob > 0.5).astype(int)
                    score = -f1_score(y_val, y_pred_binary)  # Negative for minimization
                elif optimization_metric == "precision":
                    y_pred_binary = (y_pred_prob > 0.5).astype(int)
                    score = -precision_score(y_val, y_pred_binary, zero_division=0)  # Negative for minimization
                elif optimization_metric == "recall":
                    y_pred_binary = (y_pred_prob > 0.5).astype(int)
                    score = -recall_score(y_val, y_pred_binary)  # Negative for minimization
                else:
                    score = log_loss(y_val, y_pred_prob)
                
                return score

            study = optuna.create_study(
                direction="minimize",
                study_name=f"nfl_td_model_s{self.season}_w{self.week}",
            )
            study.optimize(
                objective,
                n_trials=MODEL_PARAMS["optuna_trials"],
                timeout=MODEL_PARAMS["optuna_timeout"],
                show_progress_bar=True,
            )

            # Train final model with best params
            best_params = study.best_trial.params.copy()
            best_params.update(
                {
                    "objective": MODEL_PARAMS["objective"],
                    "eval_metric": MODEL_PARAMS["eval_metric"],
                    "scale_pos_weight": len(y_train_resampled[y_train_resampled == 0])
                    / len(y_train_resampled[y_train_resampled == 1]),
                }
            )

            # Configure XGBoost to handle missing values natively
            # XGBoost treats NaN as missing and learns optimal split directions
            best_params["missing"] = np.nan  # Explicitly set missing value handling
            
            # Train final model with early stopping on validation set
            evals = [(dtrain, "train"), (dval, "val")]
            self.model = xgb.train(
                best_params,
                dtrain,
                num_boost_round=MODEL_PARAMS["num_boost_round"],
                evals=evals,
                early_stopping_rounds=MODEL_PARAMS["early_stopping_rounds"],
                verbose_eval=10,  # Print every 10 rounds
            )
            
            # Generate raw predictions for calibration/evaluation
            y_val_pred_prob_raw = self.model.predict(dval)
            y_test_pred_prob_raw = self.model.predict(dtest)
            train_pred_prob_raw = self.model.predict(dtrain)

            # Fit Platt scaling (calibration) on validation predictions if enabled
            self._fit_platt_scaler(y_val_pred_prob_raw, y_val)

            # Apply calibration (no-op if calibrator is None)
            y_val_pred_prob = self._apply_platt_scaling(y_val_pred_prob_raw)
            y_test_pred_prob = self._apply_platt_scaling(y_test_pred_prob_raw)
            train_pred_prob = self._apply_platt_scaling(train_pred_prob_raw)

            # Optimize prediction threshold based on calibrated validation probabilities
            if MODEL_PARAMS.get("optimize_threshold", False):
                fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_prob)

                # Find threshold that maximizes F1 score
                best_f1 = 0
                best_threshold = MODEL_PARAMS["prediction_threshold"]
                for threshold in thresholds:
                    y_pred_binary = (y_val_pred_prob >= threshold).astype(int)
                    f1 = f1_score(y_val, y_pred_binary)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold

                self.optimal_threshold = best_threshold
            else:
                self.optimal_threshold = MODEL_PARAMS["prediction_threshold"]

            # Evaluate on test set using calibrated probabilities
            y_test_pred_binary = (y_test_pred_prob >= self.optimal_threshold).astype(int)

            # Also evaluate on validation set for comparison
            y_val_pred_binary = (y_val_pred_prob >= self.optimal_threshold).astype(int)

            # Calculate comprehensive metrics
            test_accuracy = accuracy_score(y_test, y_test_pred_binary)
            test_precision = precision_score(y_test, y_test_pred_binary, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred_binary)
            test_f1 = f1_score(y_test, y_test_pred_binary)
            test_logloss = log_loss(y_test, y_test_pred_prob)
            test_auc = roc_auc_score(y_test, y_test_pred_prob)

            val_accuracy = accuracy_score(y_val, y_val_pred_binary)
            val_precision = precision_score(y_val, y_val_pred_binary, zero_division=0)
            val_recall = recall_score(y_val, y_val_pred_binary)
            val_f1 = f1_score(y_val, y_val_pred_binary)
            val_logloss = log_loss(y_val, y_val_pred_prob)
            val_auc = roc_auc_score(y_val, y_val_pred_prob)

            # Check for overfitting (difference between train and validation)
            train_logloss = log_loss(y_train_resampled, train_pred_prob)
            train_accuracy = accuracy_score(y_train_resampled, (train_pred_prob >= self.optimal_threshold).astype(int))
            
            # Calculate feature importance
            self._calculate_feature_importance(X_encoded, available_features)

            # Save model and preprocessors to database
            # XGBoost save_model requires a file path, so use temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_model:
                tmp_model_path = tmp_model.name
                self.model.save_model(tmp_model_path)
                with open(tmp_model_path, 'rb') as f:
                    model_bytes = f.read()
                os.unlink(tmp_model_path)  # Clean up temp file
            
            # Serialize preprocessors to bytes
            imputer_buffer = io.BytesIO()
            joblib.dump(self.imputer, imputer_buffer)
            imputer_bytes = imputer_buffer.getvalue()
            
            scaler_buffer = io.BytesIO()
            joblib.dump(self.scaler, scaler_buffer)
            scaler_bytes = scaler_buffer.getvalue()
            
            encoder_buffer = io.BytesIO()
            joblib.dump(self.encoder, encoder_buffer)
            encoder_bytes = encoder_buffer.getvalue()
            
            # Save or update in database
            features_with_missing_payload = {
                "features_with_missing": self.features_with_missing,
                "calibrator": self.calibrator_params,
            }

            ml_model, created = MLModel.objects.update_or_create(
                season=self.season,
                week=self.week,
                defaults={
                    'model_file': model_bytes,
                    'imputer_file': imputer_bytes,
                    'scaler_file': scaler_bytes,
                    'encoder_file': encoder_bytes,
                    'features_with_missing': json.dumps(features_with_missing_payload),
                    'scaler_means': json.dumps(self.scaler_means),
                    'scaler_stds': json.dumps(self.scaler_stds),
                    'feature_importance': json.dumps(self.feature_importance),
                    'training_records': len(df),
                    'optimal_threshold': self.optimal_threshold,
                    'validation_accuracy': val_accuracy,
                    'validation_f1': val_f1,
                    'test_accuracy': test_accuracy,
                    'test_f1': test_f1,
                }
            )

            if save_model:
                # Export feature importance to CSV for analysis
                self._export_feature_importance()
            
            return (
                True,
                f"Model for Season {self.season}, Week {self.week} trained successfully!",
            )

        except Exception as e:
            error_msg = f"Error training model: {str(e)}"
            return False, error_msg

    def load(self) -> bool:
        """Load trained model and preprocessors from database"""
        try:
            if not self.model_exists():
                return False

            ml_model = MLModel.objects.get(season=self.season, week=self.week)
            
            # Load model from bytes - XGBoost load_model requires a file path
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='wb') as tmp_model:
                tmp_model_path = tmp_model.name
                tmp_model.write(ml_model.model_file)

            self.model = xgb.Booster()
            self.model.load_model(tmp_model_path)
            os.unlink(tmp_model_path)  # Clean up temp file
            
            # Load preprocessors from bytes
            imputer_buffer = io.BytesIO(ml_model.imputer_file)
            self.imputer = joblib.load(imputer_buffer)
            
            scaler_buffer = io.BytesIO(ml_model.scaler_file)
            self.scaler = joblib.load(scaler_buffer)
            
            encoder_buffer = io.BytesIO(ml_model.encoder_file)
            self.encoder = joblib.load(encoder_buffer)
            
            # Load position-aware imputation/scaling parameters if available
            self.features_with_missing = []
            self.calibrator_params = None
            if hasattr(ml_model, 'features_with_missing') and ml_model.features_with_missing:
                try:
                    payload = json.loads(ml_model.features_with_missing)
                    if isinstance(payload, dict):
                        self.features_with_missing = payload.get("features_with_missing", [])
                        self.calibrator_params = payload.get("calibrator")
                    else:
                        self.features_with_missing = payload
                except json.JSONDecodeError:
                    self.features_with_missing = []
                    self.calibrator_params = None
            
            if hasattr(ml_model, 'scaler_means') and ml_model.scaler_means:
                self.scaler_means = json.loads(ml_model.scaler_means)
            else:
                self.scaler_means = {}
            
            if hasattr(ml_model, 'scaler_stds') and ml_model.scaler_stds:
                self.scaler_stds = json.loads(ml_model.scaler_stds)
            else:
                self.scaler_stds = {}

            # Load feature importance if available
            if hasattr(ml_model, 'feature_importance') and ml_model.feature_importance:
                try:
                    self.feature_importance = json.loads(ml_model.feature_importance)
                except:
                    self.feature_importance = {}
            else:
                self.feature_importance = {}
            
            # Load optimal threshold if available
            if hasattr(ml_model, 'optimal_threshold') and ml_model.optimal_threshold is not None:
                self.optimal_threshold = float(ml_model.optimal_threshold)
            else:
                self.optimal_threshold = MODEL_PARAMS["prediction_threshold"]
            
            return True

        except MLModel.DoesNotExist:
            return False
        except Exception as e:
            return False
    
    def _fit_platt_scaler(self, val_probs: np.ndarray, val_labels: np.ndarray) -> None:
        """Fit Platt scaling (logistic regression) on validation predictions."""
        self.calibrator_params = None

        if not MODEL_PARAMS.get("enable_platt_scaling", False):
            return

        val_probs = np.asarray(val_probs, dtype=np.float64)
        val_labels = np.asarray(val_labels)

        if len(np.unique(val_labels)) < 2:
            return

        try:
            calibrator = LogisticRegression(solver="lbfgs", class_weight="balanced")
            calibrator.fit(val_probs.reshape(-1, 1), val_labels)

            coef = float(calibrator.coef_.ravel()[0])
            intercept = float(calibrator.intercept_[0])
            self.calibrator_params = {"coef": coef, "intercept": intercept}
        except Exception as exc:
            self.calibrator_params = None

    def _apply_platt_scaling(self, probs: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to probability predictions, if available."""
        if not self.calibrator_params:
            return probs

        probs = np.asarray(probs, dtype=np.float64)
        coef = float(self.calibrator_params.get("coef", 0.0))
        intercept = float(self.calibrator_params.get("intercept", 0.0))
        logits = np.clip(coef * probs + intercept, -50, 50)
        calibrated = 1.0 / (1.0 + np.exp(-logits))
        return calibrated

    def _calculate_feature_importance(self, X_encoded, feature_names):
        """Calculate feature importance using multiple methods"""
        try:
            # Get feature names - need to handle encoded features
            if hasattr(self.encoder, 'get_feature_names_out'):
                encoded_feature_names = self.encoder.get_feature_names_out(feature_names)
            else:
                # Fallback: use indices
                encoded_feature_names = [f"feature_{i}" for i in range(X_encoded.shape[1])]
            
            self.feature_names = list(encoded_feature_names)
            
            # 1. XGBoost built-in feature importance (gain)
            importance_gain = self.model.get_score(importance_type='gain')
            importance_weight = self.model.get_score(importance_type='weight')
            importance_cover = self.model.get_score(importance_type='cover')
            
            # Convert to feature names (XGBoost uses f0, f1, f2...)
            feature_importance_dict = {}
            for i, feat_name in enumerate(encoded_feature_names):
                xgb_key = f"f{i}"
                feature_importance_dict[feat_name] = {
                    'gain': importance_gain.get(xgb_key, 0),
                    'weight': importance_weight.get(xgb_key, 0),
                    'cover': importance_cover.get(xgb_key, 0),
                }
            
            self.feature_importance = feature_importance_dict
            
            # 2. SHAP values (if available) - more accurate for feature importance
            if SHAP_AVAILABLE:
                try:
                    # Use a sample of data for SHAP (too expensive on full dataset)
                    sample_size = min(1000, len(X_encoded))
                    sample_indices = np.random.choice(len(X_encoded), sample_size, replace=False)
                    X_sample = X_encoded[sample_indices]
                    
                    explainer = shap.TreeExplainer(self.model)
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Calculate mean absolute SHAP values per feature
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]  # For binary classification, get first class
                    
                    mean_shap = np.abs(shap_values).mean(axis=0)
                    
                    # Add SHAP importance to feature importance dict
                    for i, feat_name in enumerate(encoded_feature_names):
                        if feat_name in self.feature_importance:
                            self.feature_importance[feat_name]['shap'] = float(mean_shap[i])
                        else:
                            self.feature_importance[feat_name] = {'shap': float(mean_shap[i])}
                except Exception as e:
                    pass
            
        except Exception as e:
            pass
    
    def _export_feature_importance(self):
        """Export feature importance to CSV for analysis"""
        try:
            import os
            from django.conf import settings
            
            if not self.feature_importance:
                return
            
            # Prepare data for export
            importance_data = []
            for feat_name, importance in self.feature_importance.items():
                row = {'feature': feat_name}
                row.update(importance)
                importance_data.append(row)
            
            df_importance = pd.DataFrame(importance_data)
            
            # Sort by SHAP importance if available, otherwise by gain
            if 'shap' in df_importance.columns:
                df_importance = df_importance.sort_values('shap', ascending=False)
            elif 'gain' in df_importance.columns:
                df_importance = df_importance.sort_values('gain', ascending=False)
            
            # Export to CSV
            base_dir = settings.BASE_DIR if hasattr(settings, 'BASE_DIR') else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_path = os.path.join(base_dir, f'feature_importance_s{self.season}_w{self.week}.csv')
            df_importance.to_csv(csv_path, index=False)
            
        except Exception as e:
            pass

    def predict(
        self, current_week: pd.DataFrame, features: list, numeric_features: list, 
        include_shap: bool = True
    ) -> Tuple[pd.DataFrame, Optional[dict]]:
        """
        Make predictions for current week

        Args:
            current_week: Current week dataframe
            features: List of feature columns
            numeric_features: List of numeric feature columns

        Returns:
            DataFrame with predictions
        """

        if not self.load():
            raise ValueError(
                f"Model for Season {self.season}, Week {self.week} not found. Please train the model first."
            )

        # Apply minimum usage filter to improve precision (filter out inactive players)
        # Position-aware: QBs use different criteria than skill position players
        # NOTE: For future weeks, this filter uses EWMA features from historical data (weeks 1-9)
        from .config import MODEL_PARAMS
        if MODEL_PARAMS.get("min_usage_filter", True):
            min_touches = MODEL_PARAMS.get("min_touches_ewma", 3.0)
            if "touches_ewma" in current_week.columns and "position" in current_week.columns:
                before_count = len(current_week)
                
                # Separate filters for QBs vs other positions
                is_qb = current_week["position"] == "QB"
                is_skill_position = ~is_qb
                
                # For skill positions (WR/TE/RB): use touches_ewma or red_zone_touches_ewma
                has_min_touches = current_week["touches_ewma"].notna() & (current_week["touches_ewma"] >= min_touches)
                has_red_zone = False
                if "red_zone_touches_ewma" in current_week.columns:
                    has_red_zone = current_week["red_zone_touches_ewma"].notna() & (current_week["red_zone_touches_ewma"] > 0)
                
                # For QBs: must have rushing attempts (carries_ewma) since they score rushing TDs
                # QBs with low rushing attempts shouldn't be predicted
                qb_has_rushing = False
                if "carries_ewma" in current_week.columns:
                    # QBs need at least some rushing attempts to be legitimate TD threats
                    qb_min_carries = 2.0  # Lower threshold for QBs (they don't rush as much)
                    qb_has_rushing = current_week["carries_ewma"].notna() & (current_week["carries_ewma"] >= qb_min_carries)
                qb_has_red_zone = False
                if "red_zone_touches_ewma" in current_week.columns:
                    # QBs can also pass if they have red zone rushing attempts
                    qb_has_red_zone = current_week["red_zone_touches_ewma"].notna() & (current_week["red_zone_touches_ewma"] > 0)
                
                # Combine: skill positions use touches/red_zone, QBs use carries/red_zone
                skill_position_mask = is_skill_position & (has_min_touches | has_red_zone)
                qb_mask = is_qb & (qb_has_rushing | qb_has_red_zone)
                keep_mask = skill_position_mask | qb_mask
                
                current_week = current_week[keep_mask].copy()
                after_count = len(current_week)
                if before_count > after_count:
                    removed_qbs = (is_qb & ~qb_mask).sum()
                    removed_skill = (is_skill_position & ~skill_position_mask).sum()

        try:
            # Validate required columns
            if "player_id" not in current_week.columns:
                raise ValueError("Missing required column: player_id")
            
            # Filter features to only those that exist in the dataframe
            available_features = [f for f in features if f in current_week.columns]
            missing_features = [f for f in features if f not in current_week.columns]
            
            
            if not available_features:
                raise ValueError(
                    f"No features available. Missing all {len(features)} features. Available columns: {list(current_week.columns)[:20]}"
                )
            
            # Prepare data
            # Include position for imputation but exclude it from features
            features_to_select = available_features.copy()
            if "position" in current_week.columns and "position" not in features_to_select:
                features_to_select.append("position")
            
            X = current_week[["player_id"] + features_to_select].copy()
            player_ids = X["player_id"].values
            X = X.drop("player_id", axis=1)

            # Store position column for imputation, then remove from X if it's not a feature
            position_col_for_imputation = None
            if "position" in X.columns and "position" not in available_features:
                position_col_for_imputation = X["position"].copy()
                X = X.drop("position", axis=1)
            
            # Track which features were NaN BEFORE preprocessing
            # This is important for SHAP - we don't want to show contributions for features that don't apply
            # Store as dict: {feature_name: Series of booleans indicating NaN per row}
            original_nan_mask = {}
            for col in X.columns:
                # Check if column has NaN values (works for numeric and object types)
                if X[col].isna().any():
                    original_nan_mask[col] = X[col].isna()
            
            # Update numeric_features to only include those that exist
            available_numeric_features = [f for f in numeric_features if f in X.columns]
            
            # Filter out categorical features from numeric processing
            # Note: position is used internally for imputation but is NOT a feature
            categorical_features = ["report_status"]  # defense removed
            numeric_only_features = [f for f in available_numeric_features if f not in categorical_features]

            # Apply preprocessing with position-aware handling (same as training)
            if numeric_only_features:
                # Use position-aware imputation similar to training
                X_imputed = X[numeric_only_features].copy()
                
                # Get position column if available (for imputation, not as a feature)
                position_col = position_col_for_imputation if position_col_for_imputation is not None else None
                if position_col is None and "position" in current_week.columns:
                    # Fallback: try to get from original current_week
                    position_col = current_week["position"]
                
                # Apply same position-aware imputation logic as training
                # WR/TE receiving features (most significant for WR/TE, NaN for RB/QB)
                wr_te_receiving_features = [
                    "targets_ewma",
                    "receptions_ewma", 
                    "receiving_yards_ewma",
                    "receiving_touchdowns_ewma",
                ]
                
                # RB/QB rushing features (most significant for RB/QB, NaN for WR/TE)
                rb_qb_rushing_features = [
                    "carries_ewma",
                    "rushing_yards_ewma",
                    "rushing_touchdowns_ewma",
                ]
                
                # Legacy position groups (for backward compatibility)
                wr_te_features = []
                rb_only_features = []
                rb_qb_features = ["carries_ewma"]
                qb_only_features = []
                
                for feature in numeric_only_features:
                    if feature in X_imputed.columns:
                        applicable_positions = None
                        
                        # WR/TE receiving features
                        if feature in wr_te_receiving_features:
                            applicable_positions = ["WR", "TE"]
                        # RB/QB rushing features
                        elif feature in rb_qb_rushing_features:
                            applicable_positions = ["RB", "QB"]
                        # Legacy position groups
                        elif feature in wr_te_features:
                            applicable_positions = ["WR", "TE"]
                        elif feature in rb_only_features:
                            applicable_positions = ["RB"]
                        elif feature in rb_qb_features:
                            applicable_positions = ["RB", "QB"]
                        elif feature in qb_only_features:
                            applicable_positions = ["QB"]
                        
                        if applicable_positions and position_col is not None:
                            # Only impute for applicable positions
                            mask = position_col.isin(applicable_positions)
                            if mask.any() and feature in self.scaler_means:
                                # Use training mean for imputation
                                mean_val = self.scaler_means[feature]
                                if pd.notna(mean_val):
                                    X_imputed.loc[mask & X_imputed[feature].isna(), feature] = mean_val
                                else:
                                    X_imputed.loc[mask & X_imputed[feature].isna(), feature] = 0
                            # For non-applicable positions, keep NaN (already set in data_manager)
                        else:
                            # General feature (team, defense, touches, etc.) - use training mean
                            if feature in self.scaler_means:
                                mean_val = self.scaler_means[feature]
                                if pd.notna(mean_val):
                                    X_imputed[feature] = X_imputed[feature].fillna(mean_val)
                                else:
                                    X_imputed[feature] = X_imputed[feature].fillna(0)
                
                X[numeric_only_features] = X_imputed
                
                # Apply scaling using stored parameters (preserves NaN)
                for feature in numeric_only_features:
                    if feature in X.columns and feature in self.scaler_means and feature in self.scaler_stds:
                        mean_val = self.scaler_means[feature]
                        std_val = self.scaler_stds[feature]
                        if pd.notna(std_val) and std_val > 0:
                            # Scale non-NaN values, preserve NaN
                            mask = X[feature].notna()
                            X.loc[mask, feature] = (X.loc[mask, feature] - mean_val) / std_val
                        elif pd.notna(mean_val):
                            # If std is 0 or NaN, just center it
                            mask = X[feature].notna()
                            X.loc[mask, feature] = X.loc[mask, feature] - mean_val
            
            X_encoded = self.encoder.transform(X)

            # Predict
            dnew = xgb.DMatrix(X_encoded)
            y_pred_prob = self.model.predict(dnew)
            y_pred_prob = self._apply_platt_scaling(y_pred_prob)

            # Calculate SHAP values for feature explanations
            shap_values = None
            shap_base_value = None
            if include_shap and SHAP_AVAILABLE:
                try:
                    # Create SHAP explainer
                    explainer = shap.TreeExplainer(self.model)
                    shap_values_array = explainer.shap_values(X_encoded)
                    shap_base_value = explainer.expected_value
                    
                    # Convert to dict for easier JSON serialization
                    # shap_values_array shape: (n_samples, n_features)
                    # Note: Use available_features since that's what was actually used for prediction
                    shap_values = {}
                    for i, player_id in enumerate(player_ids):
                        feature_contributions = {}
                        for j, feature in enumerate(available_features):
                            # Get SHAP contribution
                            shap_contrib = float(shap_values_array[i][j])
                            
                            # CRITICAL: If the feature was NaN in the original data (doesn't apply),
                            # set SHAP contribution to 0 to avoid misleading explanations
                            # This prevents showing feature contributions when the value was actually NaN
                            if feature in original_nan_mask and original_nan_mask[feature].iloc[i]:
                                shap_contrib = 0.0
                            
                            feature_contributions[feature] = shap_contrib
                        shap_values[str(player_id)] = {
                            'contributions': feature_contributions,
                            'base_value': float(shap_base_value)
                        }
                    pass
                except Exception as e:
                    shap_values = None

            # Create output dataframe
            predictions_df = pd.DataFrame(
                {"player_id": player_ids, "probability": y_pred_prob}
            )

            predictions_df_sorted = predictions_df.sort_values(
                by="probability", ascending=False
            )

            # Merge with player information
            output = pd.merge(
                predictions_df_sorted,
                current_week[
                    [
                        "season",
                        "week",
                        "player_id",
                        "player_name",
                        "team",
                        "against",
                        "report_status",
                        "injury_status",
                        "played",
                        "position",
                    ]
                ],
                how="inner",
                on="player_id",
            )

            # Add actual touchdowns if available
            output = pd.merge(
                output,
                current_week[["player_id", "touchdown"]],
                how="left",
                on="player_id",
            )
            
            # Filter out players with high probabilities primarily due to team stats but low individual performance
            # This ensures players need meaningful individual stats, not just be on a good team
            if MODEL_PARAMS.get("require_individual_performance", True):
                min_perf_threshold = MODEL_PARAMS.get("min_individual_performance_threshold", 0.0)
                before_count = len(output)
                
                # Key individual player performance features (non-normalized raw values)
                # These represent actual player usage and performance, not team context
                individual_features = [
                    "touches_ewma",
                    "total_yards_ewma",
                    "total_touchdowns_ewma",
                    "red_zone_touches_ewma",
                ]
                
                # Check which features exist in current_week
                available_individual_features = [f for f in individual_features if f in current_week.columns]
                
                if available_individual_features:
                    # Merge individual features into output
                    individual_features_df = current_week[["player_id"] + available_individual_features].copy()
                    output = pd.merge(output, individual_features_df, on="player_id", how="left")
                    
                    # Calculate a composite individual performance score
                    # Player must have at least one feature above threshold AND
                    # Average of non-NaN individual features should be meaningful
                    has_individual_performance = None
                    for feature in available_individual_features:
                        feature_values = output[feature]
                        # Check if feature is not NaN and >= threshold
                        feature_mask = feature_values.notna() & (feature_values >= min_perf_threshold)
                        if has_individual_performance is None:
                            has_individual_performance = feature_mask
                        else:
                            has_individual_performance = has_individual_performance | feature_mask
                    
                    # Additional check: Require at least 2 individual features to be above threshold
                    # This prevents players with only one barely-above-threshold feature from passing
                    # Count how many individual features are above threshold for each player
                    feature_count_above_threshold = pd.Series(0, index=output.index)
                    for feature in available_individual_features:
                        feature_values = output[feature]
                        feature_mask = feature_values.notna() & (feature_values >= min_perf_threshold)
                        feature_count_above_threshold += feature_mask.astype(int)
                    
                    # Require at least 2 features above threshold (or at least 1 if only 1-2 features available)
                    min_features_required = min(2, len(available_individual_features))
                    has_multiple_features = feature_count_above_threshold >= min_features_required
                    has_individual_performance = has_individual_performance & has_multiple_features
                    
                    # Filter to keep only players with meaningful individual performance
                    output = output[has_individual_performance].copy()
                    
                    # Drop the individual feature columns from output (they were just for filtering)
                    output = output.drop(columns=available_individual_features, errors='ignore')
                    
                    after_count = len(output)

            # Format output
            output["touchdown"] = np.where(
                output["played"] == 1, output["touchdown"], np.nan
            )
            output["played"] = np.where(output["played"] == 1, "Y", "N")

            # Evaluate if actual results available
            filtered_td = [x for x in output["touchdown"].values if pd.notna(x)]
            if len(filtered_td) > 0:
                filtered_probs = [
                    output["probability"].values[i]
                    for i in range(len(output["touchdown"].values))
                    if pd.notna(output["touchdown"].values[i])
                ]

                # Use optimal threshold if available, otherwise use default
                threshold = getattr(self, 'optimal_threshold', MODEL_PARAMS["prediction_threshold"])
                y_pred_binary = [
                    1 if prob >= threshold else 0
                    for prob in filtered_probs
                ]

                pass

            return output, shap_values

        except Exception as e:
            error_msg = f"Error making predictions: {str(e)}"
            raise


def train_model(
    df: pd.DataFrame, season: int, week: int, features: list, numeric_features: list
) -> Tuple[bool, str]:
    """Train model wrapper function"""
    model = NFLTouchdownModel(season, week)
    return model.train(df, features, numeric_features)


def predict_week(
    season: int,
    week: int,
    current_week: pd.DataFrame,
    features: list,
    numeric_features: list,
    include_shap: bool = True,
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Predict week wrapper function"""
    model = NFLTouchdownModel(season, week)
    return model.predict(current_week, features, numeric_features, include_shap=include_shap)


def retrain_model(
    season: int, week: int, df: pd.DataFrame, features: list, numeric_features: list
) -> Tuple[bool, str]:
    """Retrain model wrapper function"""
    model = NFLTouchdownModel(season, week)

    if model.model_exists():
        return (
            False,
            f"Model for Season {season}, Week {week} already exists. Use the confirmation dialog to overwrite.",
        )

    return model.train(df, features, numeric_features)


def compare_breakout_feature_sets(
    df: pd.DataFrame,
    season: int,
    week: int,
    combined_features: List[str],
    combined_numeric_features: List[str],
) -> None:
    """
    Compare model performance between combined vs. split breakout feature sets.

    Prints evaluation summaries to the console for both configurations.
    """
    required_split_features = [
        "recent_rushing_td_breakout",
        "recent_receiving_td_breakout",
        "recent_rushing_yards_breakout",
        "recent_receiving_yards_breakout",
    ]

    missing_split = [feat for feat in required_split_features if feat not in df.columns]
    if missing_split:
        return

    combined_feature_list = combined_features
    combined_numeric_list = combined_numeric_features

    # Build split feature lists by replacing combined breakout features
    split_features: List[str] = []
    split_numeric_features: List[str] = []

    for feat in combined_features:
        if feat == "recent_total_breakout_tds_position_normalized":
            split_features.extend(["recent_rushing_td_breakout", "recent_receiving_td_breakout"])
        elif feat == "recent_total_breakout_yards_position_normalized":
            split_features.extend(["recent_rushing_yards_breakout", "recent_receiving_yards_breakout"])
        else:
            split_features.append(feat)

    for feat in combined_numeric_features:
        if feat == "recent_total_breakout_tds_position_normalized":
            split_numeric_features.extend(["recent_rushing_td_breakout", "recent_receiving_td_breakout"])
        elif feat == "recent_total_breakout_yards_position_normalized":
            split_numeric_features.extend(["recent_rushing_yards_breakout", "recent_receiving_yards_breakout"])
        else:
            split_numeric_features.append(feat)


    scenarios = [
        ("Combined Breakout Features", combined_feature_list, combined_numeric_list),
        ("Split Breakout Features", split_features, split_numeric_features),
    ]

    for label, feats, num_feats in scenarios:

        model = NFLTouchdownModel(season, week)
        success, message = model.train(df.copy(), feats, num_feats, save_model=False, comparison_label=label)
        if not success:
            pass
        else:
            pass
