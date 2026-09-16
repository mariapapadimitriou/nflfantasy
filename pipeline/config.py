"""
Configuration for the touchdown prediction pipeline.

Target: did this player score any touchdown in this game? WR/TE/RB can score
rushing or receiving; a quarterback only counts for rushing, since a passing
touchdown is credited to the receiver.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "web" / "data" / "predictions.json"

# Prior seasons of training data, in addition to the current one.
HISTORICAL_SEASONS = 2

POSITIONS = ["QB", "RB", "WR", "TE"]

# Ordered least to most severe; used as an ordinal model feature.
INJURY_SEVERITY = {
    "": 0,
    "Healthy": 0,
    "Questionable": 1,
    "Doubtful": 2,
    "Out": 3,
    "IR": 3,
    "PUP": 3,
}
# A player at or above this severity is removed from the shortlist entirely.
INJURY_EXCLUDE_AT = 2


# ---------------------------------------------------------------- features

# Rolling-usage features, each normalized against the player's position so a
# workload that is heavy for a receiver is not dwarfed by a routine carry count.
# The floor guards against dividing by a near-zero weekly average, so it must sit
# BELOW the feature's typical average -- a floor above it pins the divisor to a
# constant and the normalization silently degrades into the raw value.
POSITION_NORMALIZED = {
    "targets_ewma": 0.5,
    "touches_ewma": 0.5,
    "total_yards_ewma": 1.0,
    "total_touchdowns_ewma": 0.02,
    "reception_rate_ewma": 0.05,
    "red_zone_touches_ewma": 0.02,
    "red_zone_touch_share_ewma": 0.005,
    "snap_share_ewma": 0.05,
}
NORMALIZED_CAP = 5.0

FEATURES = [
    # Individual usage and production
    "targets_ewma_position_normalized",
    "touches_ewma_position_normalized",
    "total_yards_ewma_position_normalized",
    "total_touchdowns_ewma_position_normalized",
    "reception_rate_ewma_position_normalized",
    "red_zone_touches_ewma_position_normalized",
    "red_zone_touch_share_ewma_position_normalized",
    "snap_share_ewma_position_normalized",
    # Raw snap share as well: the normalized form says "busy for his position",
    # this says "on the field", and the two disagree in useful ways.
    "snap_share_ewma",
    "depth_chart_rank",
    # Recent breakout, expressed as multiples of the qualifying threshold. These
    # are sparse spikes, so they are scaled against an absolute bar rather than
    # a position average that is itself zero most weeks.
    "recent_breakout_tds_normalized",
    "recent_breakout_yards_normalized",
    # Game environment. The implied team total is the market's own view of how
    # many points this offense will score, which is the most direct statement
    # available about how many touchdowns are on offer.
    "implied_team_total",
    "game_total",
    "spread_line",
    "team_win_probability",
    "is_dome",
    "wind",
    "days_rest",
    # Team context
    "team_play_volume_ewma",
    "team_red_zone_volume_ewma",
    # Opposing defense
    "def_tds_allowed_ewma",
    "def_red_zone_tds_allowed_ewma",
    "def_yards_allowed_ewma",
    # Availability
    "injury_severity",
]


# ------------------------------------------------------------------- model

MODEL_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "num_boost_round": 400,
    "early_stopping_rounds": 40,
    "optuna_trials": 40,
    "optuna_timeout": 600,
    # Chronological splits. Weekly rows for one player are not independent, so a
    # random split leaks his later games into training and flatters every metric.
    "test_size": 0.2,
    "val_size": 0.15,
    "random_state": 42,
    # The product ranks a shortlist, so ranking quality is the headline metric.
    "precision_at_k": 10,
}

EWMA_ALPHA = 0.5
MIN_GAMES_PLAYED = 4  # before a player's rolling features mean anything

# Cap outliers before the rolling average, using only games already played.
WINSORIZE_PERCENTILE = 0.95
WINSORIZE_MIN_PERIODS = 8

BREAKOUT_TDS = 2
BREAKOUT_YARDS = 150

# Eligibility: a player needs real individual usage, not just a seat on a
# productive offense.
MIN_TOUCHES_EWMA = 3.0
MIN_QB_CARRIES_EWMA = 2.0
