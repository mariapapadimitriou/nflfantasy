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

# Feature Configuration
FEATURES = [
    "home",
    "prev_receiving_touchdowns",
    "prev_receiving_yards",
    "prev_rushing_touchdowns",
    "prev_rushing_yards",
    "prev_receptions",
    "prev_targets",
    "report_status",
    "rolling_receiving_touchdowns",
    "rolling_receiving_yards",
    "rolling_rushing_touchdowns",
    "rolling_rushing_yards",
    "rolling_receptions",
    "rolling_targets",
    "touchdown_attempts",
    "rolling_touchdown_attempts",
    "red_zone_completion_pct",
    "rolling_red_zone_completion_pct",
    "rookie",
    "wp",
    "rolling_red_zone",
    "rolling_yapg",
    "prev_red_zone",
    "prev_red_zone_completion_pct",
    "qb_rolling_passing_tds",
    "qb_rolling_passing_yards",
    # Defensive stats (rolling)
    "rolling_def_yards_gained",
    "rolling_def_touchdown",
    "rolling_def_passing_yards",
    "rolling_def_rushing_yards",
    "rolling_def_points_allowed",
    # Defensive stats (previous season)
    "prev_def_yards_gained",
    "prev_def_touchdown",
    "prev_def_passing_yards",
    "prev_def_rushing_yards",
    "prev_def_points_allowed",
]

NUMERIC_FEATURES = [
    "prev_receiving_touchdowns",
    "prev_receiving_yards",
    "prev_rushing_touchdowns",
    "prev_rushing_yards",
    "prev_receptions",
    "prev_targets",
    "rolling_receiving_touchdowns",
    "rolling_receiving_yards",
    "rolling_rushing_touchdowns",
    "rolling_rushing_yards",
    "rolling_receptions",
    "rolling_targets",
    "touchdown_attempts",
    "rolling_touchdown_attempts",
    "red_zone_completion_pct",
    "rolling_red_zone_completion_pct",
    "wp",
    "rolling_red_zone",
    "rolling_yapg",
    "prev_red_zone",
    "prev_red_zone_completion_pct",
    "qb_rolling_passing_tds",
    "qb_rolling_passing_yards",
    # Defensive stats (rolling)
    "rolling_def_yards_gained",
    "rolling_def_touchdown",
    "rolling_def_passing_yards",
    "rolling_def_rushing_yards",
    "rolling_def_points_allowed",
    # Defensive stats (previous season)
    "prev_def_yards_gained",
    "prev_def_touchdown",
    "prev_def_passing_yards",
    "prev_def_rushing_yards",
    "prev_def_points_allowed",
]

# Player positions to include
POSITIONS = ["WR", "QB", "RB", "TE"]

# Report status categories
REPORT_STATUS_ORDER = ["Healthy", "Minor", "Questionable", "Doubtful", "Out"]

# Model Configuration
MODEL_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "optuna_trials": 100,
    "optuna_timeout": 600,
    "num_boost_round": 100,
    "test_size": 0.2,
    "random_state": 42,
    "prediction_threshold": 0.5,
}

# Data Loading Configuration
HISTORICAL_SEASONS = 3  # Number of previous seasons to load
MAX_WEEK = 20

# Cache Configuration
CACHE_ENABLED = True
# Historical data is now stored in SQLite database via TrainingData model

