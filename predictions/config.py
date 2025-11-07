"""
Configuration file for NFL Touchdown Prediction App
Centralized configuration for easy maintenance and updates
"""
import os
from pathlib import Path

# Get BASE_DIR - try Django settings first, otherwise use path resolution
try:
    from django.conf import settings
    if settings.configured:
        BASE_DIR = settings.BASE_DIR
    else:
        # Django not configured yet, use path resolution
        BASE_DIR = Path(__file__).resolve().parent.parent
except (ImportError, AttributeError, RuntimeError):
    # Fallback if Django isn't available or not configured
    BASE_DIR = Path(__file__).resolve().parent.parent

# Data Storage Paths
# Note: Models and historical data are now stored in SQLite database
# All data is computed on-the-fly, no file caching needed

# Feature Configuration - Complete Feature Overhaul
FEATURES = [
    # Player Usage & Performance Metrics (EWMA) - All normalized by position to remove bias
    # Removed: "targets_ewma" - use position-normalized version
    "targets_ewma_position_normalized",  # targets normalized by position average
    # Removed: "receptions_ewma" - use reception_rate_ewma instead (more meaningful)
    # Removed: "carries_ewma" - doesn't apply to WR/TE, use targets_ewma for receiving usage instead
    # Removed: "reception_rate_ewma" - use position-normalized version
    "reception_rate_ewma_position_normalized",  # reception rate normalized by position average
    # Removed: "touches_ewma" - too RB/QB heavy, use position-specific features instead
    "touches_ewma_position_normalized",  # touches_ewma normalized by position average (removes position bias)
    # Combined yards and touchdowns (position-agnostic - works for all positions)
    # Removed: "total_yards_ewma" - use position-normalized version
    "total_yards_ewma_position_normalized",  # total yards normalized by position average
    # Removed: "total_touchdowns_ewma" - use position-normalized version
    "total_touchdowns_ewma_position_normalized",  # total touchdowns normalized by position average
    # Removed: "red_zone_touches_ewma" - too RB/QB heavy, use position-normalized version
    "red_zone_touches_ewma_position_normalized",  # red_zone_touches_ewma normalized by position average
    # Removed: "red_zone_touch_share_ewma" - use position-normalized version
    "red_zone_touch_share_ewma_position_normalized",
    
    # Breakout Game Indicators (capture recent surge in performance) - Position-normalized
    # Based on total touchdowns and total yards (position-agnostic)
    "recent_total_breakout_tds_position_normalized",  # Total TDs breakout normalized by position
    "recent_total_breakout_yards_position_normalized",  # Total yards breakout normalized by position
    "recent_breakout_game",  # 1 if any breakout occurred in last game
    
    # Team Context (EWMA)
    "team_play_volume_ewma",
    "team_total_red_zone_touches_ewma",
    "team_win_probability",
    "spread_line",
    
    # Defense Context (EWMA) - Only top importance features kept
    "def_ewma_TDs_allowed_per_game",
    "def_ewma_interceptions_per_game",
    "opponent_red_zone_def_rank",
    # Removed: def_ewma_yards_allowed_per_game (low importance ~0.027)
    # Removed: def_ewma_red_zone_completion_pct_allowed (low importance ~0.012)
    
    # QB Context (EWMA) - Only yardage stats, not TD counts (to avoid spurious correlations)
    "qb_passing_yards_ewma",
    "qb_rushing_yards_ewma",
    # Removed: qb_passing_TDs_ewma (causing spurious correlations)
    # Removed: qb_rushing_TDs_ewma (causing spurious correlations, especially for RBs)
    
    # Player Identity / Categorical Features
    # Note: defense removed - high cardinality (32 teams × seasons) causes overfitting
    # Defense quality is already captured by defensive EWMA features
]

NUMERIC_FEATURES = [
    # Player Usage & Performance Metrics (EWMA) - All normalized by position to remove bias
    # Removed: "targets_ewma" - use position-normalized version
    "targets_ewma_position_normalized",
    # Removed: "receptions_ewma" - use reception_rate_ewma instead (more meaningful)
    # Removed: "carries_ewma" - doesn't apply to WR/TE, use targets_ewma for receiving usage instead
    # Removed: "reception_rate_ewma" - use position-normalized version
    "reception_rate_ewma_position_normalized",
    # Removed: "touches_ewma" - too RB/QB heavy, use position-specific features instead
    "touches_ewma_position_normalized",
    # Combined yards and touchdowns (position-agnostic - works for all positions)
    # Removed: "total_yards_ewma" - use position-normalized version
    "total_yards_ewma_position_normalized",
    # Removed: "total_touchdowns_ewma" - use position-normalized version
    "total_touchdowns_ewma_position_normalized",
    # Removed: "red_zone_touches_ewma" - too RB/QB heavy, use position-normalized version
    "red_zone_touches_ewma_position_normalized",
    # Removed: "red_zone_touch_share_ewma" - use position-normalized version
    "red_zone_touch_share_ewma_position_normalized",
    
    # Breakout Game Indicators (capture recent surge in performance) - Position-normalized
    # Based on total touchdowns and total yards (position-agnostic)
    "recent_total_breakout_tds_position_normalized",  # Total TDs breakout normalized by position
    "recent_total_breakout_yards_position_normalized",  # Total yards breakout normalized by position
    "recent_breakout_game",  # 1 if any breakout occurred in last game
    
    # Team Context (EWMA)
    "team_play_volume_ewma",
    "team_total_red_zone_touches_ewma",
    "team_win_probability",
    "spread_line",
    
    # Defense Context (EWMA) - Only top importance features kept
    "def_ewma_TDs_allowed_per_game",
    "def_ewma_interceptions_per_game",
    "opponent_red_zone_def_rank",
    # Removed: def_ewma_yards_allowed_per_game (low importance ~0.027)
    # Removed: def_ewma_red_zone_completion_pct_allowed (low importance ~0.012)
    
    # QB Context (EWMA) - Only yardage stats, not TD counts (to avoid spurious correlations)
    "qb_passing_yards_ewma",
    "qb_rushing_yards_ewma",
    # Removed: qb_passing_TDs_ewma (causing spurious correlations)
    # Removed: qb_rushing_TDs_ewma (causing spurious correlations, especially for RBs)
]

CATEGORICAL_FEATURES = [
    # No categorical features - all numeric features used
]

# Player positions to include
POSITIONS = ["WR", "QB", "RB", "TE"]

# Report status categories (kept for reference, may not be used in new features)
REPORT_STATUS_ORDER = ["Healthy", "Minor", "Questionable", "Doubtful", "Out"]

# Model Configuration
# Target Variable: Binary (1 if player scored ANY touchdown, 0 otherwise)
# - TE/WR/RB: Can score rushing OR receiving touchdowns
# - QB: Can only score rushing touchdowns (QBs don't catch passes)
MODEL_PARAMS = {
    "objective": "binary:logistic",  # Binary classification: TD or no TD
    "eval_metric": "logloss",
    
    # Hyperparameter Optimization
    "optuna_trials": 150,  # Increased for better search
    "optuna_timeout": 900,  # Increased timeout (15 minutes)
    
    # Training Configuration
    "num_boost_round": 500,  # Increased max rounds (early stopping will prevent overfitting)
    "early_stopping_rounds": 50,  # Increased to 50 for more aggressive early stopping
    "early_stopping_patience": 50,  # Same as early_stopping_rounds for clarity
    "enable_platt_scaling": True,  # Calibrate probabilities with Platt scaling
    
    # Feature Selection
    "min_feature_importance": 0.01,  # Minimum SHAP importance to keep feature (0.01 = 1%)
    "use_top_n_features": None,  # If set, use only top N features by importance (None = use all)
    
    # Data Splitting
    "test_size": 0.2,  # 20% for test set
    "val_size": 0.15,  # 15% of training data for validation (from remaining 80%)
    "random_state": 42,
    
    # Prediction Threshold (can be optimized based on precision/recall tradeoff)
    "prediction_threshold": 0.5,
    
    # Model Selection Metric
    "optimization_metric": "precision",  # Options: "accuracy", "f1", "precision", "recall", "logloss"
    "optimize_threshold": True,  # Optimize prediction threshold based on validation set
    
    # Prediction Filtering (to improve precision)
    "min_touches_ewma": 3.0,  # Minimum touches_ewma to make predictions (filters out inactive players)
    "min_usage_filter": True,  # Apply minimum usage filters before prediction
}

# Feature Engineering Configuration
EWMA_ALPHA = 0.3  # Exponential decay factor (0-1, higher = more weight to recent games)
EWMA_WEEKS = 5  # Number of previous weeks to consider for EWMA

# Winsorization Configuration (to handle breakout games)
# Caps extreme values before EWMA calculation to prevent outliers from skewing averages
WINSORIZE_ENABLED = True  # Enable winsorization before EWMA
WINSORIZE_PERCENTILE = 0.95  # Cap values at 95th percentile (0.95 = cap at 95th percentile)

# Breakout Detection Configuration
# Flags recent breakout games separately so model can learn from them
BREAKOUT_ENABLED = True  # Enable breakout game indicators
BREAKOUT_TOTAL_TDS = 2  # Minimum total TDs (receiving + rushing) in a game to count as breakout
BREAKOUT_TOTAL_YARDS = 150  # Minimum total yards (receiving + rushing) in a game to count as breakout
# Data Loading Configuration
HISTORICAL_SEASONS = 2  # Number of previous seasons to use for training (season - 2, so for 2025 use 2023, 2024)
MAX_WEEK = 20

# Cache Configuration
CACHE_ENABLED = True
# Historical data is now stored in SQLite database via TrainingData model
