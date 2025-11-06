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
import json
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

from .config import REPORT_STATUS_ORDER, MODEL_PARAMS, POSITIONS, CATEGORICAL_FEATURES
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
        self.features_with_missing = []  # Features that should keep NaN for inapplicable positions
        self.scaler_means = {}  # Store scaling means for each feature
        self.scaler_stds = {}  # Store scaling stds for each feature
        self.feature_importance = {}  # Store feature importance scores
        self.feature_names = []  # Store feature names in order

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
            # Validate required columns
            if "player_id" not in df.columns:
                return (False, "Missing required column: player_id")
            if "touchdown" not in df.columns:
                return (False, "Missing required column: touchdown")
            
            # Filter features to only those that exist in the dataframe
            available_features = [f for f in features if f in df.columns]
            missing_features = [f for f in features if f not in df.columns]
            
            if missing_features:
                print(f"[Warning] Missing features in dataframe: {missing_features}")
                print(f"[Info] Using {len(available_features)} available features out of {len(features)} requested")
            
            if not available_features:
                return (
                    False,
                    f"No features available. Missing all {len(features)} features. Available columns: {list(df.columns)[:20]}",
                )
            
            # Prepare data
            X = df[["player_id"] + available_features].copy()
            y = df["touchdown"].copy()
            player_ids = X["player_id"].values
            X = X.drop("player_id", axis=1)
            
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
            
            # Check for position (ordinal)
            if "position" in X.columns:
                categorical_features_to_encode.append("position")
                ordinal_categories.append(POSITIONS)

            # Position-aware imputation strategy
            # For features that don't apply to a position (e.g., air_yards for RBs), keep NaN
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
            # targets_ewma, receptions_ewma, touches_ewma, red_zone_touches_ewma, 
            # red_zone_touch_share_ewma, td_streak_factor, regression_td_factor, etc.
            
            # Ensure we're not including categorical features in numeric processing
            numeric_only_features = [f for f in available_numeric_features if f not in categorical_features_to_encode]
            
            # Validate numeric features exist
            if not numeric_only_features:
                print(f"[Warning] No numeric features available after filtering. Available columns: {list(X.columns)}")
            else:
                # Create a copy to work with
                X_imputed = X[numeric_only_features].copy()
                
                # Get position column if available
                position_col = None
                if "position" in X.columns:
                    position_col = X["position"]
                
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
                    transformers.append(
                        (
                            f"ordinal_{cat_feature}",
                            OrdinalEncoder(categories=[ordinal_categories[i]]),
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

            # Split data
            X_train, X_test, y_train, y_test, _, _ = train_test_split(
                X_encoded,
                y,
                player_ids,
                test_size=MODEL_PARAMS["test_size"],
                random_state=MODEL_PARAMS["random_state"],
            )

            # Handle NaN values before SMOTE (SMOTE doesn't accept NaN)
            # XGBoost can handle NaN natively, so we'll temporarily replace NaN with a sentinel value
            # then restore NaN after SMOTE but before XGBoost
            print("[Training] Handling NaN values for SMOTE compatibility...")
            NAN_SENTINEL = -999999.0  # Large negative value that XGBoost will treat as missing
            
            # Convert to numpy array for easier NaN handling
            X_train_np = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
            X_test_np = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
            
            # Replace NaN with sentinel value before SMOTE
            X_train_np = np.where(np.isnan(X_train_np), NAN_SENTINEL, X_train_np)
            
            # Apply SMOTE
            print("[Training] Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=MODEL_PARAMS["random_state"])
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_np, y_train)
            
            # Restore NaN values after SMOTE (for XGBoost to handle natively)
            # Convert sentinel back to NaN
            X_train_resampled = np.where(X_train_resampled == NAN_SENTINEL, np.nan, X_train_resampled)
            
            # Also handle test set NaN with sentinel for XGBoost
            X_test_np = np.where(np.isnan(X_test_np), NAN_SENTINEL, X_test_np)

            # Create DMatrix (XGBoost will handle NaN natively)
            # Note: We use the sentinel value which XGBoost treats as missing
            dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled, enable_categorical=False)
            dtest = xgb.DMatrix(X_test_np, label=y_test, enable_categorical=False)

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

            # Configure XGBoost to handle missing values natively
            # XGBoost treats NaN as missing and learns optimal split directions
            best_params["missing"] = np.nan  # Explicitly set missing value handling
            
            self.model = xgb.train(
                best_params, dtrain, num_boost_round=MODEL_PARAMS["num_boost_round"]
            )

            # Evaluate
            y_pred_prob = self.model.predict(dtest)
            y_pred_binary = [
                1 if prob > MODEL_PARAMS["prediction_threshold"] else 0
                for prob in y_pred_prob
            ]
            
            # Calculate feature importance
            print("\n[Feature Importance] Calculating feature importance...")
            self._calculate_feature_importance(X_encoded, available_features)

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
                    'features_with_missing': json.dumps(self.features_with_missing),
                    'scaler_means': json.dumps(self.scaler_means),
                    'scaler_stds': json.dumps(self.scaler_stds),
                    'feature_importance': json.dumps(self.feature_importance),
                    'training_records': len(df),
                }
            )

            print(f"\n[Success] Model saved to database for Season {self.season}, Week {self.week}")
            
            # Export feature importance to CSV for analysis
            self._export_feature_importance()
            
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
            
            # Load position-aware imputation/scaling parameters if available
            if hasattr(ml_model, 'features_with_missing') and ml_model.features_with_missing:
                self.features_with_missing = json.loads(ml_model.features_with_missing)
            else:
                self.features_with_missing = []
            
            if hasattr(ml_model, 'scaler_means') and ml_model.scaler_means:
                self.scaler_means = json.loads(ml_model.scaler_means)
            else:
                self.scaler_means = {}
            
            if hasattr(ml_model, 'scaler_stds') and ml_model.scaler_stds:
                self.scaler_stds = json.loads(ml_model.scaler_stds)
            else:
                self.scaler_stds = {}

            print(f"[Model] Loaded model from database for Season {self.season}, Week {self.week}")
            
            # Load feature importance if available
            if hasattr(ml_model, 'feature_importance') and ml_model.feature_importance:
                try:
                    self.feature_importance = json.loads(ml_model.feature_importance)
                except:
                    self.feature_importance = {}
            else:
                self.feature_importance = {}
            
            return True

        except MLModel.DoesNotExist:
            print(f"[Error] Model not found in database for Season {self.season}, Week {self.week}")
            return False
        except Exception as e:
            print(f"[Error] Failed to load model: {str(e)}")
            return False
    
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
                print("[Feature Importance] Calculating SHAP values for feature importance...")
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
                    
                    print("[Feature Importance] SHAP values calculated successfully")
                except Exception as e:
                    print(f"[Warning] Could not calculate SHAP values: {e}")
            
            print(f"[Feature Importance] Calculated importance for {len(self.feature_importance)} features")
            
        except Exception as e:
            print(f"[Warning] Error calculating feature importance: {e}")
            import traceback
            traceback.print_exc()
    
    def _export_feature_importance(self):
        """Export feature importance to CSV for analysis"""
        try:
            import os
            from django.conf import settings
            
            if not self.feature_importance:
                print("[Warning] No feature importance data to export")
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
            
            print(f"[Feature Importance] Exported to {csv_path}")
            
            # Print top 20 features
            print("\n[Feature Importance] Top 20 Most Important Features:")
            print("=" * 80)
            top_col = 'shap' if 'shap' in df_importance.columns else 'gain'
            top_features = df_importance.head(20)
            for idx, row in top_features.iterrows():
                feat_name = row['feature']
                importance_val = row[top_col]
                print(f"  {feat_name:50s} {importance_val:10.4f}")
            
        except Exception as e:
            print(f"[Warning] Error exporting feature importance: {e}")
            import traceback
            traceback.print_exc()

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
            # Validate required columns
            if "player_id" not in current_week.columns:
                raise ValueError("Missing required column: player_id")
            
            # Filter features to only those that exist in the dataframe
            available_features = [f for f in features if f in current_week.columns]
            missing_features = [f for f in features if f not in current_week.columns]
            
            if missing_features:
                print(f"[Warning] Missing features in prediction dataframe: {missing_features}")
                print(f"[Info] Using {len(available_features)} available features out of {len(features)} requested")
            
            if not available_features:
                raise ValueError(
                    f"No features available. Missing all {len(features)} features. Available columns: {list(current_week.columns)[:20]}"
                )
            
            # Prepare data
            X = current_week[["player_id"] + available_features].copy()
            player_ids = X["player_id"].values
            X = X.drop("player_id", axis=1)
            
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
            categorical_features = ["position", "report_status"]
            numeric_only_features = [f for f in available_numeric_features if f not in categorical_features]

            # Apply preprocessing with position-aware handling (same as training)
            if numeric_only_features:
                # Use position-aware imputation similar to training
                X_imputed = X[numeric_only_features].copy()
                
                # Get position column if available
                position_col = None
                if "position" in X.columns:
                    position_col = X["position"]
                
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
                    # Note: Use available_features since that's what was actually used for prediction
                    shap_values = {}
                    for i, player_id in enumerate(player_ids):
                        feature_contributions = {}
                        for j, feature in enumerate(available_features):
                            # Get SHAP contribution
                            shap_contrib = float(shap_values_array[i][j])
                            
                            # CRITICAL: If the feature was NaN in the original data (doesn't apply),
                            # set SHAP contribution to 0 to avoid misleading explanations
                            # This prevents showing "air_yards_share_ewma +50%" when the value was actually NaN
                            if feature in original_nan_mask and original_nan_mask[feature].iloc[i]:
                                shap_contrib = 0.0
                            
                            feature_contributions[feature] = shap_contrib
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
