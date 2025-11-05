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
    # Player Usage & Performance Metrics (EWMA)
    "targets_ewma", # player_stats
    "receptions_ewma", # player_stats
    "carries_ewma", # player_stats
    "touches_ewma", # player_stats
    "receiving_yards_ewma", # player_stats
    "receiving_touchdowns_ewma", # player_stats
    "rushing_yards_ewma", # player_stats
    "rushing_touchdowns_ewma", # player_stats
    "red_zone_touches_ewma", 
    "red_zone_touch_share_ewma",
    "end_zone_targets_ewma",
    "designed_rush_attempts_ewma",
    
    # Team Context (EWMA)
    "team_play_volume_ewma",
    "team_total_red_zone_touches_ewma",
    "team_total_air_yards_ewma",
    "team_win_probability",
    "spread_line",
    
    # Defense Context (EWMA)
    "def_ewma_yards_allowed_per_game",
    "def_ewma_TDs_allowed_per_game",
    "def_ewma_red_zone_completion_pct_allowed",
    "def_ewma_interceptions_per_game",
    "opponent_red_zone_def_rank",
    
    # QB Context (EWMA)
    "qb_rolling_rushing_yards_ewma",
    "qb_rolling_rushing_TDs_ewma",
    
    # Player Identity / Categorical Features
    "position",
]

NUMERIC_FEATURES = [
    # Player Usage & Performance Metrics (EWMA)
    "targets_ewma",
    "receptions_ewma",
    "carries_ewma",
    "touches_ewma",
    "receiving_yards_ewma",
    "receiving_touchdowns_ewma",
    "rushing_yards_ewma",
    "rushing_touchdowns_ewma",
    "red_zone_touches_ewma",
    "red_zone_touch_share_ewma",
    "end_zone_targets_ewma",
    "designed_rush_attempts_ewma",
    
    # Team Context (EWMA)
    "team_play_volume_ewma",
    "team_total_red_zone_touches_ewma",
    "team_total_air_yards_ewma",
    "team_win_probability",
    "spread_line",
    
    # Defense Context (EWMA)
    "def_ewma_yards_allowed_per_game",
    "def_ewma_TDs_allowed_per_game",
    "def_ewma_red_zone_completion_pct_allowed",
    "def_ewma_interceptions_per_game",
    "opponent_red_zone_def_rank",
    
    # QB Context (EWMA)
    "qb_rolling_rushing_yards_ewma",
    "qb_rolling_rushing_TDs_ewma",
]

CATEGORICAL_FEATURES = [
    "position",
]

# Player positions to include
POSITIONS = ["WR", "QB", "RB", "TE"]

# Report status categories (kept for reference, may not be used in new features)
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

# Feature Engineering Configuration
EWMA_ALPHA = 0.7  # Exponential decay factor (0-1, higher = more weight to recent games)
EWMA_WEEKS = 4  # Number of previous weeks to consider for EWMA
REGRESSION_LAMBDA = 0.3  # Lambda for regression_td_factor: EWMA + λ(xTD - EWMA)

# Data Loading Configuration
HISTORICAL_SEASONS = 3  # Number of previous seasons to load
MAX_WEEK = 20

# Cache Configuration
CACHE_ENABLED = True
# Historical data is now stored in SQLite database via TrainingData model
