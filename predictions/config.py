"""
Configuration for the NFL touchdown prediction app.

Target variable: 1 if the player scored any touchdown that game, else 0.
WR/TE/RB can score rushing or receiving; QBs only rushing, since a passing
touchdown is credited to the receiver.
"""

from pathlib import Path

try:
    from django.conf import settings

    BASE_DIR = settings.BASE_DIR if settings.configured else Path(__file__).resolve().parent.parent
except (ImportError, AttributeError, RuntimeError):
    BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- features

# Every model feature is numeric, so one list serves as both the feature set and
# the numeric subset. Usage and production metrics are position-normalized so a
# workload that is heavy for a WR is not dwarfed by an ordinary RB's carry count.
FEATURES = [
    # Player usage and production (EWMA, normalized against position peers)
    "targets_ewma_position_normalized",
    "reception_rate_ewma_position_normalized",
    "touches_ewma_position_normalized",
    "total_yards_ewma_position_normalized",
    "total_touchdowns_ewma_position_normalized",
    "red_zone_touches_ewma_position_normalized",
    "red_zone_touch_share_ewma_position_normalized",
    # Recent breakout indicators, scaled against the breakout threshold
    "recent_total_breakout_tds_normalized",
    "recent_total_breakout_yards_normalized",
    "recent_breakout_game",
    # Team context
    "team_play_volume_ewma",
    "team_total_red_zone_touches_ewma",
    "team_win_probability",
    "spread_line",
    # Opposing defense context
    "def_ewma_TDs_allowed_per_game",
    "def_ewma_interceptions_per_game",
    "opponent_red_zone_def_rank",
    # Supporting quarterback (yardage only; QB TD counts produced spurious
    # correlations, particularly inflating RB predictions)
    "qb_passing_yards_ewma",
    "qb_rushing_yards_ewma",
]

NUMERIC_FEATURES = FEATURES

# Raw (non-normalized) features used by the eligibility filters. These measure
# actual individual usage, so a player on a high-scoring offense cannot qualify
# on team context alone.
INDIVIDUAL_PERFORMANCE_FEATURES = [
    "touches_ewma",
    "total_yards_ewma",
    "total_touchdowns_ewma",
    "red_zone_touches_ewma",
]

POSITIONS = ["WR", "QB", "RB", "TE"]

REPORT_STATUS_ORDER = ["Healthy", "Minor", "Questionable", "Doubtful", "Out"]

# Statuses that rule a player out of the prediction pool entirely.
EXCLUDED_INJURY_STATUSES = ["Out", "IR", "PUP", "Doubtful", "Suspended", "NA"]


# ------------------------------------------------------------------- model

MODEL_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    # Hyperparameter search
    "optuna_trials": 150,
    "optuna_timeout": 900,
    # Training
    "num_boost_round": 500,
    "early_stopping_rounds": 50,
    "enable_platt_scaling": True,
    # Class imbalance. XGBoost's scale_pos_weight handles the ~20% positive rate
    # natively and, unlike SMOTE, tolerates NaN, so no synthetic rows are needed.
    "use_smote": False,
    # Splitting. Weekly observations of the same player are not independent, so
    # splits are chronological: the most recent weeks form validation and test.
    # A random split leaks a player's own future games into training.
    "temporal_split": True,
    "test_size": 0.2,
    "val_size": 0.15,
    "random_state": 42,
    # Thresholds and selection
    "prediction_threshold": 0.5,
    "optimization_metric": "auc",  # accuracy | f1 | precision | recall | logloss | auc
    "optimize_threshold": True,
    # The product ranks a shortlist, so ranking quality is the headline metric.
    "precision_at_k": 10,
    # Eligibility filters
    "min_usage_filter": True,
    "min_touches_ewma": 3.0,
    "min_qb_carries_ewma": 2.0,
    "min_red_zone_touches_ewma": 0.01,
    "require_individual_performance": True,
    "min_individual_performance_threshold": 0.0,
    "current_season_only_from_week": 3,
    "exclude_injured": True,
}


# ------------------------------------------------- feature engineering knobs

EWMA_ALPHA = 0.5  # higher weights recent games more heavily
EWMA_WEEKS = 4  # minimum games of history before a player is modelled

# Cap outliers before the EWMA so one huge game does not dominate the average.
# The cap is computed from prior games only, never the full series.
WINSORIZE_ENABLED = True
WINSORIZE_PERCENTILE = 0.95
WINSORIZE_MIN_PERIODS = 10

# Breakout detection: a standout prior game is carried as its own signal.
# These are sparse spikes, zero for most players in most weeks, so they are
# scaled against the threshold that defines them rather than against a position
# average. A position-week average of a mostly-zero column is itself mostly
# zero, which makes the division either a no-op or an explosion.
BREAKOUT_ENABLED = True
BREAKOUT_TOTAL_TDS = 2
BREAKOUT_TOTAL_YARDS = 150

# Position normalization. Each feature is divided by its position's average for
# that week. The divisor is floored to keep low-average features from exploding,
# so the floor must sit BELOW the feature's typical average or the division
# silently becomes a no-op. Floors are therefore set per feature.
POSITION_NORMALIZED_FEATURES = {
    "targets_ewma": 0.5,
    "touches_ewma": 0.5,
    "total_yards_ewma": 1.0,
    "total_touchdowns_ewma": 0.02,
    "reception_rate_ewma": 0.05,
    "red_zone_touches_ewma": 0.02,
    "red_zone_touch_share_ewma": 0.005,
}
POSITION_NORMALIZED_CAP = 5.0


# ------------------------------------------------------------------- data

HISTORICAL_SEASONS = 2  # prior seasons of training data, plus the current one
MAX_WEEK = 22  # includes playoffs
CACHE_ENABLED = True
