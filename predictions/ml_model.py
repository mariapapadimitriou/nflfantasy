"""
Model Training and Prediction Module
"""

import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import optuna
from typing import Tuple, Optional
import io
import tempfile
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[Warning] SHAP not available. Feature explanations will not be generated.")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import log_loss, classification_report, confusion_matrix

from .config import REPORT_STATUS_ORDER, MODEL_PARAMS
from .models import MLModel


class NFLTouchdownModel:
    """XGBoost model for NFL touchdown prediction"""

    def __init__(self, season: int, week: int):
        self.season = season
        self.week = week

        self.imputer = None
        self.scaler = None
        self.encoder = None
        self.model = None

    def model_exists(self) -> bool:
        """Check if model for this season/week exists in database"""
        return MLModel.objects.filter(season=self.season, week=self.week).exists()

    def train(
        self, df: pd.DataFrame, features: list, numeric_features: list
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
        print(f"\n{'='*60}")
        print(f"Training model for Season {self.season}, Week {self.week}")
        print(f"{'='*60}\n")

        try:
            # Prepare data
            X = df[["player_id"] + features].copy()
            y = df["touchdown"].copy()
            player_ids = X["player_id"].values
            X = X.drop("player_id", axis=1)

            # Setup preprocessors
            ordinal_features = ["report_status"]

            self.imputer = SimpleImputer(strategy="mean")
            self.scaler = StandardScaler()

            X[numeric_features] = self.imputer.fit_transform(X[numeric_features])
            X[numeric_features] = self.scaler.fit_transform(X[numeric_features])

            self.encoder = ColumnTransformer(
                transformers=[
                    (
                        "ordinal",
                        OrdinalEncoder(categories=[REPORT_STATUS_ORDER]),
                        ordinal_features,
                    )
                ],
                remainder="passthrough",
            )

            X_encoded = self.encoder.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test, _, _ = train_test_split(
                X_encoded,
                y,
                player_ids,
                test_size=MODEL_PARAMS["test_size"],
                random_state=MODEL_PARAMS["random_state"],
            )

            # Apply SMOTE
            print("[Training] Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=MODEL_PARAMS["random_state"])
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            # Create DMatrix
            dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled)
            dtest = xgb.DMatrix(X_test, label=y_test)

            # Hyperparameter optimization with Optuna
            print("[Training] Optimizing hyperparameters with Optuna...")

            def objective(trial):
                param = {
                    "objective": MODEL_PARAMS["objective"],
                    "eval_metric": MODEL_PARAMS["eval_metric"],
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "eta": trial.suggest_float("eta", 0.01, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "scale_pos_weight": len(y_train_resampled[y_train_resampled == 0])
                    / len(y_train_resampled[y_train_resampled == 1]),
                    "lambda": trial.suggest_float("lambda", 1e-8, 10.0),
                    "alpha": trial.suggest_float("alpha", 1e-8, 10.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                }

                model = xgb.train(
                    param, dtrain, num_boost_round=MODEL_PARAMS["num_boost_round"]
                )
                y_pred_prob = model.predict(dtest)
                loss = log_loss(y_test, y_pred_prob)
                return loss

            study = optuna.create_study(direction="minimize")
            study.optimize(
                objective,
                n_trials=MODEL_PARAMS["optuna_trials"],
                timeout=MODEL_PARAMS["optuna_timeout"],
                show_progress_bar=True,
            )

            # Train final model with best params
            print(f"[Training] Best parameters: {study.best_trial.params}")
            best_params = study.best_trial.params
            best_params.update(
                {
                    "objective": MODEL_PARAMS["objective"],
                    "eval_metric": MODEL_PARAMS["eval_metric"],
                    "scale_pos_weight": len(y_train_resampled[y_train_resampled == 0])
                    / len(y_train_resampled[y_train_resampled == 1]),
                }
            )

            self.model = xgb.train(
                best_params, dtrain, num_boost_round=MODEL_PARAMS["num_boost_round"]
            )

            # Evaluate
            y_pred_prob = self.model.predict(dtest)
            y_pred_binary = [
                1 if prob > MODEL_PARAMS["prediction_threshold"] else 0
                for prob in y_pred_prob
            ]

            print("\n[Evaluation] Model Performance:")
            print(classification_report(y_test, y_pred_binary))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred_binary))

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
            ml_model, created = MLModel.objects.update_or_create(
                season=self.season,
                week=self.week,
                defaults={
                    'model_file': model_bytes,
                    'imputer_file': imputer_bytes,
                    'scaler_file': scaler_bytes,
                    'encoder_file': encoder_bytes,
                    'training_records': len(df),
                }
            )

            print(f"\n[Success] Model saved to database for Season {self.season}, Week {self.week}")
            return (
                True,
                f"Model for Season {self.season}, Week {self.week} trained successfully!",
            )

        except Exception as e:
            error_msg = f"Error training model: {str(e)}"
            print(f"\n[Error] {error_msg}")
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

            print(f"[Model] Loaded model from database for Season {self.season}, Week {self.week}")
            return True

        except MLModel.DoesNotExist:
            print(f"[Error] Model not found in database for Season {self.season}, Week {self.week}")
            return False
        except Exception as e:
            print(f"[Error] Failed to load model: {str(e)}")
            return False

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
        print(f"\n{'='*60}")
        print(f"Predicting touchdowns for Season {self.season}, Week {self.week}")
        print(f"{'='*60}\n")

        if not self.load():
            raise ValueError(
                f"Model for Season {self.season}, Week {self.week} not found. Please train the model first."
            )

        try:
            # Prepare data
            X = current_week[["player_id"] + features].copy()
            player_ids = X["player_id"].values
            X = X.drop("player_id", axis=1)

            # Apply preprocessing
            X[numeric_features] = self.imputer.transform(X[numeric_features])
            X[numeric_features] = self.scaler.transform(X[numeric_features])
            X_encoded = self.encoder.transform(X)

            # Predict
            dnew = xgb.DMatrix(X_encoded)
            y_pred_prob = self.model.predict(dnew)

            # Calculate SHAP values for feature explanations
            shap_values = None
            shap_base_value = None
            if include_shap and SHAP_AVAILABLE:
                try:
                    print("[SHAP] Calculating feature contributions...")
                    # Create SHAP explainer
                    explainer = shap.TreeExplainer(self.model)
                    shap_values_array = explainer.shap_values(X_encoded)
                    shap_base_value = explainer.expected_value
                    
                    # Convert to dict for easier JSON serialization
                    # shap_values_array shape: (n_samples, n_features)
                    shap_values = {}
                    for i, player_id in enumerate(player_ids):
                        feature_contributions = {}
                        for j, feature in enumerate(features):
                            feature_contributions[feature] = float(shap_values_array[i][j])
                        shap_values[str(player_id)] = {
                            'contributions': feature_contributions,
                            'base_value': float(shap_base_value)
                        }
                    print(f"[SHAP] Calculated contributions for {len(shap_values)} players")
                except Exception as e:
                    print(f"[Warning] Failed to calculate SHAP values: {str(e)}")
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

                y_pred_binary = [
                    1 if prob > MODEL_PARAMS["prediction_threshold"] else 0
                    for prob in filtered_probs
                ]

                print("\n[Evaluation] Predictions vs Actual:")
                print(classification_report(filtered_td, y_pred_binary))
                print("\nConfusion Matrix:")
                print(confusion_matrix(filtered_td, y_pred_binary))

            print(f"\n[Success] Generated predictions for {len(output)} players")
            return output, shap_values

        except Exception as e:
            error_msg = f"Error making predictions: {str(e)}"
            print(f"\n[Error] {error_msg}")
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
