"""
Feature engineering for the NFL touchdown model.

Every rolling feature describes games that had already been played when the
prediction would have been made. Nothing here may read a row's own game or any
later one.
"""

import logging
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    BREAKOUT_ENABLED,
    BREAKOUT_TOTAL_TDS,
    BREAKOUT_TOTAL_YARDS,
    EWMA_ALPHA,
    WINSORIZE_ENABLED,
    WINSORIZE_MIN_PERIODS,
    WINSORIZE_PERCENTILE,
)
from .utils import american_odds_to_probability

logger = logging.getLogger(__name__)

GAME_KEYS = ["game_id", "season", "week"]
TEAM_GAME_KEYS = ["team"] + GAME_KEYS
PLAYER_GAME_KEYS = ["player_id"] + GAME_KEYS
PLAYER_ORDER = ["player_id", "season", "week"]
TEAM_ORDER = ["team", "season", "week"]


# --------------------------------------------------------------- primitives


def calculate_ewma_feature(
    series: pd.Series, alpha: float = EWMA_ALPHA, winsorize: bool = True
) -> pd.Series:
    """Exponentially weighted mean of a player's *prior* games.

    The series is shifted before anything else, so every subsequent step —
    including the outlier cap — can only see games already played. Capping
    against a quantile of the whole series would let a player's future
    production set the ceiling applied to his past.
    """
    past = series.shift(1)

    if winsorize and WINSORIZE_ENABLED:
        cap = past.expanding(min_periods=WINSORIZE_MIN_PERIODS).quantile(WINSORIZE_PERCENTILE)
        past = past.where(cap.isna(), past.clip(upper=cap))

    return past.ewm(alpha=alpha, adjust=False).mean()


def add_ewma_columns(
    df: pd.DataFrame,
    group_key: str,
    columns: Dict[str, str],
    alpha: float = EWMA_ALPHA,
) -> pd.DataFrame:
    """Add EWMA columns for a ``{source: destination}`` mapping.

    Sources absent from the frame produce an all-NaN destination, so downstream
    code can rely on the column existing without re-checking each one.
    """
    for source, destination in columns.items():
        if source in df.columns:
            df[destination] = df.groupby(group_key)[source].transform(
                lambda s: calculate_ewma_feature(s, alpha)
            )
        else:
            logger.debug("Source column %s missing; %s set to NaN", source, destination)
            df[destination] = np.nan
    return df


def fill_forward_with_fallback(
    df: pd.DataFrame,
    group_key: str,
    columns: Sequence[str],
    defaults: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Carry each group's last known value forward, then backfill the rest.

    Teams with no history at all fall back to the league median for the column,
    or to a supplied constant when the column is empty everywhere.
    """
    defaults = defaults or {}
    for column in columns:
        if column not in df.columns:
            continue
        df[column] = df.groupby(group_key)[column].ffill()
        if df[column].isna().any():
            median = df[column].median()
            fallback = median if pd.notna(median) else defaults.get(column, 0.0)
            df[column] = df[column].fillna(fallback)
    return df


def extend_to_scheduled_games(
    stats: pd.DataFrame,
    schedules: pd.DataFrame,
    columns: Sequence[str],
    defaults: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Project team-level stats onto every scheduled game, including future ones.

    An upcoming game has no play-by-play yet, so each team carries its most
    recent values forward. Without this, the week being predicted would have no
    team or defensive context at all.
    """
    if len(schedules) == 0 or "game_id" not in schedules.columns:
        return stats

    slate = pd.concat(
        [
            schedules[GAME_KEYS + [side]].rename(columns={side: "team"})
            for side in ("home_team", "away_team")
        ],
        ignore_index=True,
    ).drop_duplicates()

    available = [c for c in columns if c in stats.columns]
    slate = slate.merge(stats[TEAM_GAME_KEYS + available], on=TEAM_GAME_KEYS, how="left")
    slate = slate.sort_values(TEAM_ORDER)
    slate = fill_forward_with_fallback(slate, "team", available, defaults)

    logger.debug(
        "Extended team context to %s scheduled games (%s had prior stats)",
        len(slate),
        len(stats),
    )
    return slate[TEAM_GAME_KEYS + available].copy()


# ------------------------------------------------------------ player-level


def calculate_player_features(pbp: pd.DataFrame, weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """Build per-player, per-game rolling features."""
    features = weekly_stats.copy()

    if "game_id" not in features.columns and {"player_id", "game_id"} <= set(pbp.columns):
        game_info = pbp[PLAYER_GAME_KEYS].drop_duplicates()
        features = features.merge(game_info, on=PLAYER_ORDER, how="left")

    features = features.sort_values(PLAYER_ORDER).reset_index(drop=True)

    for column in ("carries", "targets"):
        if column not in features.columns:
            features[column] = np.nan

    # Touches stay NaN when the player recorded neither carries nor targets, so
    # "did not play" is distinguishable from "played and was not used".
    recorded = features["carries"].notna() | features["targets"].notna()
    features["touches"] = (
        features["carries"].fillna(0) + features["targets"].fillna(0)
    ).where(recorded)

    features["red_zone_touches"] = _red_zone_touches(pbp, features)

    features = add_ewma_columns(
        features,
        "player_id",
        {
            "targets": "targets_ewma",
            "receptions": "receptions_ewma",
            "carries": "carries_ewma",
            "touches": "touches_ewma",
            "red_zone_touches": "red_zone_touches_ewma",
            "receiving_yards": "receiving_yards_ewma",
            "receiving_tds": "receiving_touchdowns_ewma",
            "rushing_yards": "rushing_yards_ewma",
            "rushing_tds": "rushing_touchdowns_ewma",
        },
    )

    if BREAKOUT_ENABLED:
        features = _add_breakout_indicators(features)

    return features


def _red_zone_touches(pbp: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """Count each player's targets and carries inside the 20-yard line."""
    if "yardline_100" not in pbp.columns:
        return pd.Series(np.nan, index=features.index)

    counts = count_red_zone_plays(pbp)
    if counts is None:
        return pd.Series(np.nan, index=features.index)

    merged = features[PLAYER_GAME_KEYS].merge(counts, on=PLAYER_GAME_KEYS, how="left")
    touches = pd.Series(merged["red_zone_touches"].to_numpy(), index=features.index)

    # Someone who appeared but never reached the red zone had zero touches
    # there, which is real information. Someone who did not appear has none,
    # which is absence of information and must stay NaN.
    appeared = features["touches"].notna() if "touches" in features.columns else touches.notna()
    return touches.fillna(0).where(appeared)


def count_red_zone_plays(pbp: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Red-zone targets plus carries per player and game."""
    in_red_zone = pbp["yardline_100"] <= 20
    frames = []

    for id_column, play_column, label in (
        ("receiver_player_id", "pass", "rz_targets"),
        ("rusher_player_id", "rush", "rz_carries"),
    ):
        if id_column not in pbp.columns or play_column not in pbp.columns:
            continue
        plays = pbp[in_red_zone & (pbp[play_column] == 1) & pbp[id_column].notna()]
        counts = (
            plays.groupby([id_column] + GAME_KEYS)
            .size()
            .reset_index(name=label)
            .rename(columns={id_column: "player_id"})
        )
        frames.append(counts)

    if not frames:
        return None

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on=PLAYER_GAME_KEYS, how="outer")

    tallies = [c for c in ("rz_targets", "rz_carries") if c in combined.columns]
    combined["red_zone_touches"] = sum(combined[c].fillna(0) for c in tallies)
    return combined[PLAYER_GAME_KEYS + ["red_zone_touches"]]


def _add_breakout_indicators(features: pd.DataFrame) -> pd.DataFrame:
    """Flag a standout previous game as its own signal.

    Winsorization deliberately flattens these spikes out of the EWMA, so they
    are carried here instead of being lost.
    """
    for label, parts in (
        ("total_tds", ("receiving_tds", "rushing_tds")),
        ("total_yards", ("receiving_yards", "rushing_yards")),
    ):
        present = [c for c in parts if c in features.columns]
        features[label] = (
            sum(features[c].fillna(0) for c in present) if present else 0.0
        )

    for source, destination, minimum in (
        ("total_tds", "recent_total_breakout_tds", BREAKOUT_TOTAL_TDS),
        ("total_yards", "recent_total_breakout_yards", BREAKOUT_TOTAL_YARDS),
    ):
        previous = features.groupby("player_id")[source].shift(1)
        features[destination] = previous.where(previous >= minimum, 0)
        # Expressed as multiples of the qualifying threshold: 0 for no breakout,
        # 1.0 for one exactly at the bar, higher for a bigger game. The
        # threshold is an absolute scale, so this is already comparable across
        # positions without dividing by a near-zero group average.
        features[f"{destination}_normalized"] = features[destination] / minimum

    features["recent_breakout_game"] = (
        (features["recent_total_breakout_tds"] > 0)
        | (features["recent_total_breakout_yards"] > 0)
    ).astype(int)

    return features.drop(columns=["total_tds", "total_yards"])


# -------------------------------------------------------------- team-level


def calculate_team_context_features(
    player_features: pd.DataFrame, pbp: pd.DataFrame, schedules: pd.DataFrame
) -> pd.DataFrame:
    """Team play volume, red-zone volume, win probability and spread."""
    if "team" not in player_features.columns:
        raise ValueError("'team' column is required to build team context")

    if "posteam" in pbp.columns:
        volume = (
            pbp.groupby(["posteam"] + GAME_KEYS)
            .size()
            .reset_index(name="total_plays")
            .rename(columns={"posteam": "team"})
        )
    else:
        logger.warning("pbp has no 'posteam'; approximating play volume from player touches")
        volume = (
            player_features.groupby(TEAM_GAME_KEYS)["touches"]
            .sum()
            .reset_index(name="total_plays")
        )

    volume = volume.sort_values(TEAM_ORDER)
    volume = add_ewma_columns(volume, "team", {"total_plays": "team_play_volume_ewma"})

    red_zone = (
        player_features.groupby(TEAM_GAME_KEYS)["red_zone_touches"]
        .sum()
        .reset_index(name="team_total_red_zone_touches")
        .sort_values(TEAM_ORDER)
    )
    red_zone = add_ewma_columns(
        red_zone, "team", {"team_total_red_zone_touches": "team_total_red_zone_touches_ewma"}
    )

    context = volume[TEAM_GAME_KEYS + ["team_play_volume_ewma"]].merge(
        red_zone[TEAM_GAME_KEYS + ["team_total_red_zone_touches_ewma"]],
        on=TEAM_GAME_KEYS,
        how="outer",
    )

    defaults = {"team_play_volume_ewma": 65.0, "team_total_red_zone_touches_ewma": 0.0}
    context = context.sort_values(TEAM_ORDER)
    context = fill_forward_with_fallback(
        context, "team", list(defaults), defaults
    )
    context = extend_to_scheduled_games(context, schedules, list(defaults), defaults)

    return _add_game_odds(context, schedules)


def _add_game_odds(context: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Attach win probability and point spread from each team's perspective."""
    if "home_moneyline" not in schedules.columns:
        context["team_win_probability"] = 0.5
        context["spread_line"] = 0.0
        return context

    games = schedules[GAME_KEYS + ["home_team", "away_team", "home_moneyline", "spread_line"]].copy()
    games["home_wp"] = games["home_moneyline"].apply(american_odds_to_probability)

    home = games[GAME_KEYS + ["home_team", "home_wp", "spread_line"]].rename(
        columns={"home_team": "team", "home_wp": "team_win_probability"}
    )
    away = games[GAME_KEYS + ["away_team", "home_wp", "spread_line"]].rename(
        columns={"away_team": "team"}
    )
    # The spread is quoted from the home team's side; invert both for the away team.
    away["team_win_probability"] = 1 - away["home_wp"]
    away["spread_line"] = -away["spread_line"]
    away = away.drop(columns=["home_wp"])

    odds = pd.concat([home, away], ignore_index=True)
    context = context.merge(
        odds[TEAM_GAME_KEYS + ["team_win_probability", "spread_line"]],
        on=TEAM_GAME_KEYS,
        how="left",
    )
    context["team_win_probability"] = context["team_win_probability"].fillna(0.5)
    context["spread_line"] = context["spread_line"].fillna(0.0)
    return context


def calculate_team_shares(df: pd.DataFrame, team_context: pd.DataFrame) -> pd.DataFrame:
    """Each player's share of his team's red-zone work."""
    column = "team_total_red_zone_touches_ewma"
    if column in df.columns:
        df = df.drop(columns=[column])

    keys = TEAM_GAME_KEYS + ([column] if column in team_context.columns else [])
    df = df.merge(team_context[keys], on=TEAM_GAME_KEYS, how="left")

    for required in ("red_zone_touches_ewma", column):
        if required not in df.columns:
            df[required] = np.nan

    df["red_zone_touch_share_ewma"] = df["red_zone_touches_ewma"] / (
        df[column].fillna(0) + 1
    )
    return df


def calculate_defensive_features(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Rolling stats allowed by each defense, projected onto every scheduled game."""
    output_columns = [
        "def_ewma_yards_allowed_per_game",
        "def_ewma_TDs_allowed_per_game",
        "def_ewma_red_zone_completion_pct_allowed",
        "def_ewma_interceptions_per_game",
        "opponent_red_zone_def_rank",
    ]

    if "defteam" not in pbp.columns:
        logger.warning("pbp has no 'defteam'; defensive features unavailable")
        return pd.DataFrame(columns=TEAM_GAME_KEYS + output_columns)

    aggregations = {"yards_gained": "sum", "touchdown": "sum"}
    if "interception" in pbp.columns:
        aggregations["interception"] = "sum"

    stats = (
        pbp.groupby(["defteam"] + GAME_KEYS)
        .agg(aggregations)
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    stats["rz_completion_pct_allowed"] = _red_zone_completion_allowed(pbp, stats)
    stats = stats.sort_values(TEAM_ORDER)

    stats = add_ewma_columns(
        stats,
        "team",
        {
            "yards_gained": "def_ewma_yards_allowed_per_game",
            "touchdown": "def_ewma_TDs_allowed_per_game",
            "rz_completion_pct_allowed": "def_ewma_red_zone_completion_pct_allowed",
            "interception": "def_ewma_interceptions_per_game",
        },
    )

    defaults = {
        "def_ewma_yards_allowed_per_game": 350.0,
        "def_ewma_TDs_allowed_per_game": 2.0,
        "def_ewma_red_zone_completion_pct_allowed": 0.5,
        "def_ewma_interceptions_per_game": 1.0,
    }
    stats = fill_forward_with_fallback(stats, "team", list(defaults), defaults)
    stats = extend_to_scheduled_games(stats, schedules, list(defaults), defaults)

    # Rank 1 is the most generous red-zone defense, so a high value means a
    # tougher matchup. Ranked per week across the teams playing that week.
    stats["opponent_red_zone_def_rank"] = (
        stats.groupby(["season", "week"])["def_ewma_red_zone_completion_pct_allowed"]
        .rank(method="min", ascending=False)
        .fillna(16)
    )

    return stats[TEAM_GAME_KEYS + output_columns]


def _red_zone_completion_allowed(pbp: pd.DataFrame, stats: pd.DataFrame) -> pd.Series:
    """Completion rate each defense allowed inside the 20."""
    if not {"yardline_100", "complete_pass", "pass"} <= set(pbp.columns):
        return pd.Series(np.nan, index=stats.index)

    red_zone_passes = pbp[(pbp["yardline_100"] <= 20) & (pbp["pass"] == 1)]
    allowed = (
        red_zone_passes.groupby(["defteam"] + GAME_KEYS)["complete_pass"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"defteam": "team", "sum": "completions", "count": "attempts"})
    )
    allowed["rz_completion_pct_allowed"] = allowed["completions"] / (allowed["attempts"] + 1)

    merged = stats[TEAM_GAME_KEYS].merge(
        allowed[TEAM_GAME_KEYS + ["rz_completion_pct_allowed"]],
        on=TEAM_GAME_KEYS,
        how="left",
    )
    return merged["rz_completion_pct_allowed"].to_numpy()


def calculate_qb_stats(player_features: pd.DataFrame) -> pd.DataFrame:
    """Rolling passing and rushing yardage for each quarterback.

    Merged onto that QB's teammates later as supporting-cast context.
    """
    if "position" not in player_features.columns:
        return pd.DataFrame()

    quarterbacks = player_features[player_features["position"] == "QB"]
    if quarterbacks.empty:
        return pd.DataFrame()

    quarterbacks = quarterbacks.sort_values(PLAYER_ORDER).reset_index(drop=True)
    quarterbacks = add_ewma_columns(
        quarterbacks,
        "player_id",
        {
            "passing_yards": "qb_passing_yards_ewma",
            "rushing_yards": "qb_rushing_yards_ewma",
        },
    )

    return quarterbacks[
        PLAYER_ORDER + ["qb_passing_yards_ewma", "qb_rushing_yards_ewma"]
    ].copy()


# ------------------------------------------------------ position normalizing


def add_position_normalized_features(
    df: pd.DataFrame, specs: Dict[str, float], cap: float
) -> pd.DataFrame:
    """Rescale each feature against its position's average for that week.

    A WR with five touches and an RB with fifteen can both be heavily used for
    their position. Dividing by the position-week average makes them comparable.

    The divisor is floored to keep a near-zero average from producing absurd
    ratios, but the floor has to stay below the feature's typical average --
    a floor above it pins the divisor at a constant and the normalization
    silently degrades into the raw value.
    """
    if "position" not in df.columns:
        return df

    for feature, floor in specs.items():
        if feature not in df.columns:
            continue

        averages = df.groupby(["position", "season", "week"])[feature].transform("mean")
        if floor >= averages.median(skipna=True):
            logger.warning(
                "Normalization floor %.4f for %s exceeds its median position average "
                "(%.4f); normalization will be a no-op",
                floor,
                feature,
                averages.median(skipna=True),
            )

        normalized = df[feature] / averages.clip(lower=floor)
        df[f"{feature}_position_normalized"] = normalized.clip(upper=cap).where(
            df[feature].notna()
        )

    return df
