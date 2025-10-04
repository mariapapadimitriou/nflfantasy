"""
Configuration file for NFL Touchdown Prediction App
Centralized configuration for easy maintenance and updates
"""
import os

# Data Storage Paths
DATA_DIR = "data"
MODEL_DIR = "Model"
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Feature Configuration
FEATURES = [
    'home',
    'prev_receiving_touchdowns', 
    'prev_receiving_yards', 
    'prev_rushing_touchdowns', 
    'prev_rushing_yards', 
    'report_status', 
    'rolling_receiving_touchdowns', 
    'rolling_receiving_yards', 
    'rolling_rushing_touchdowns',
    'rolling_rushing_yards', 
    'rookie', 
    'wp', 
    'rolling_red_zone', 
    'rolling_yapg', 
    'prev_red_zone', 
    'qb_rolling_passing_tds', 
    'qb_rolling_passing_yards'
]

NUMERIC_FEATURES = [
    'prev_receiving_touchdowns', 
    'prev_receiving_yards', 
    'prev_rushing_touchdowns', 
    'prev_rushing_yards', 
    'rolling_receiving_touchdowns', 
    'rolling_receiving_yards', 
    'rolling_rushing_touchdowns', 
    'rolling_rushing_yards', 
    'wp', 
    'rolling_red_zone', 
    'rolling_yapg', 
    'prev_red_zone', 
    'qb_rolling_passing_tds', 
    'qb_rolling_passing_yards'
]

# Player positions to include
POSITIONS = ['WR', 'QB', 'RB', 'TE']

# Report status categories
REPORT_STATUS_ORDER = ['Healthy', 'Minor', 'Questionable', 'Doubtful', 'Out']

# Model Configuration
MODEL_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'optuna_trials': 100,
    'optuna_timeout': 600,
    'num_boost_round': 100,
    'test_size': 0.2,
    'random_state': 42,
    'prediction_threshold': 0.5
}

# Data Loading Configuration
HISTORICAL_SEASONS = 3  # Number of previous seasons to load
MAX_WEEK = 20

# Cache Configuration
CACHE_ENABLED = True
HISTORICAL_DATA_FILE = os.path.join(CACHE_DIR, "historical_data.parquet")