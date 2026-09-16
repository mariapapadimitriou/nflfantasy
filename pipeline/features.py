"""
Feature engineering.

Every rolling feature describes games that had already been played at the moment
the prediction would have been made. Nothing here may read a row's own game or
any later one -- that is the single invariant the whole model rests on.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import (
    BREAKOUT_TDS,
    BREAKOUT_YARDS,
    EWMA_ALPHA,
    INJURY_SEVERITY,
    MIN_GAMES_PLAYED,
    NORMALIZED_CAP,
    POSITION_NORMALIZED,
    POSITIONS,
    WINSORIZE_MIN_PERIODS,
    WINSORIZE_PERCENTILE,
)
from . import sources

logger = logging.getLogger(__name__)

PLAYER_WEEK = ["player_id", "season", "week"]
TEAM_WEEK = ["team", "season", "week"]

STAT_COLUMNS = [
    "carries", "targets", "receptions", "rushing_yards", "receiving_yards",
    "rushing_tds", "receiving_tds", "passing_yards",
]


# --------------------------------------------------------------- primitives


def ewma(series: pd.Series, alpha: float = EWMA_ALPHA) -> pd.Series:
    """Exponentially weighted mean over a player's *prior* games.

    Shifted first, so every later step -- including the outlier cap -- can only
    see games already played. Capping against a quantile of the whole series
    would let a player's future production set the ceiling applied to his past.
    """
    past = series.shift(1)
    cap = past.expanding(min_periods=WINSORIZE_MIN_PERIODS).quantile(WINSORIZE_PERCENTILE)
    past = past.where(cap.isna(), past.clip(upper=cap))
    return past.ewm(alpha=alpha, adjust=False).mean()


def add_ewma(df: pd.DataFrame, group: str, columns: Dict[str, str]) -> pd.DataFrame:
    """Add rolling columns for a ``{source: destination}`` mapping."""
    for source, destination in columns.items():
        if source in df.columns:
            df[destination] = df.groupby(group)[source].transform(ewma)
        else:
            logger.debug("%s missing; %s set to NaN", source, destination)
            df[destination] = np.nan
    return df


def position_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Rescale usage features against each position's average for that week."""
    for feature, floor in POSITION_NORMALIZED.items():
        if feature not in df.columns:
            continue
        average = df.groupby(["position", "season", "week"])[feature].transform("mean")

        typical = average.median(skipna=True)
        if pd.notna(typical) and floor >= typical:
            logger.warning(
                "Floor %.4f for %s sits above its median position average (%.4f); "
                "normalization will be a no-op",
                floor, feature, typical,
            )

        ratio = df[feature] / average.clip(lower=floor)
        df[f"{feature}_position_normalized"] = ratio.clip(upper=NORMALIZED_CAP).where(
            df[feature].notna()
        )
    return df


# ------------------------------------------------------------ frame assembly


def build(season: int, week: int) -> pd.DataFrame:
    """Assemble the full modelling frame across the loaded seasons."""
    seasons = list(range(season - 2, season + 1))

    schedules = sources.load("schedules", seasons)
    schedules = schedules[
        (schedules["season"] < season)
        | ((schedules["season"] == season) & (schedules["week"] <= week))
    ]
    pbp = sources.load("pbp", seasons)

    df = _player_games(sources.load("rosters_weekly", seasons), schedules)
    stats = _weekly_stats(sources.load("player_stats", seasons))

    df = _add_outcome(df, pbp, stats)
    df = _add_rolling_usage(df, stats, pbp, seasons)
    df = _add_team_context(df, pbp)
    df = _add_defense(df, pbp)
    df = _add_injuries(df, seasons)

    # A player's slice of his own offense's red-zone work. Needs both the
    # individual and the team rolling averages, so it lands after both.
    df["red_zone_touch_share_ewma"] = df["red_zone_touches_ewma"] / (
        df["team_red_zone_volume_ewma"].fillna(0) + 1
    )

    df = position_normalize(df)
    return _finalize(df)


def _player_games(rosters: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per rostered skill player per scheduled game.

    Weekly rosters carry the team a player was actually on that week, so a
    midseason trade resolves correctly instead of attaching him to both clubs'
    entire schedules.
    """
    rosters = rosters[
        rosters["position"].isin(POSITIONS) & (rosters["status"] == "ACT")
    ].copy()
    rosters = rosters.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
    rosters = rosters[
        ["player_id", "player_name", "position", "depth_chart_position", "team", "season", "week"]
    ].dropna(subset=["player_id"])

    games = _team_games(schedules)
    df = rosters.merge(games, on=TEAM_WEEK, how="inner")
    return df.drop_duplicates(subset=["player_id", "game_id"])


def _team_games(schedules: pd.DataFrame) -> pd.DataFrame:
    """Game context from each team's own perspective."""
    keep = [
        "game_id", "season", "week", "home_team", "away_team",
        "spread_line", "total_line", "home_moneyline", "away_moneyline",
        "roof", "wind", "home_rest", "away_rest",
    ]
    schedules = schedules[[c for c in keep if c in schedules.columns]].copy()

    sides = []
    for side, other, moneyline, rest in (
        ("home_team", "away_team", "home_moneyline", "home_rest"),
        ("away_team", "home_team", "away_moneyline", "away_rest"),
    ):
        frame = schedules.rename(columns={side: "team", other: "against"}).copy()
        frame["is_home"] = int(side == "home_team")
        frame["team_win_probability"] = frame[moneyline].apply(_implied_probability)
        frame["days_rest"] = frame[rest] if rest in frame.columns else np.nan
        # The spread is quoted from the home side; flip it for the away team so
        # a negative number always means "this team is favoured".
        if side == "away_team":
            frame["spread_line"] = -frame["spread_line"]
        sides.append(frame)

    games = pd.concat(sides, ignore_index=True)
    games["game_total"] = games["total_line"]
    # The market's own estimate of this offense's points: half the game total,
    # adjusted by half the spread. The most direct available statement about how
    # many touchdowns are actually on offer.
    games["implied_team_total"] = games["game_total"] / 2 - games["spread_line"] / 2
    games["is_dome"] = games["roof"].isin(["dome", "closed"]).astype(int)
    games["wind"] = pd.to_numeric(games.get("wind"), errors="coerce").fillna(0)

    return games[
        TEAM_WEEK + [
            "game_id", "against", "is_home", "spread_line", "game_total",
            "implied_team_total", "team_win_probability", "is_dome", "wind", "days_rest",
        ]
    ]


def _implied_probability(odds) -> float:
    """American moneyline to implied win probability, 0.5 when unpriced."""
    try:
        odds = int(odds)
    except (TypeError, ValueError):
        return 0.5
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def _weekly_stats(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Per-player weekly stat lines, restricted to the columns the model uses."""
    columns = PLAYER_WEEK + [c for c in STAT_COLUMNS if c in player_stats.columns]
    stats = player_stats[columns].copy()

    recorded = stats.get("carries", pd.Series(np.nan, index=stats.index)).notna() | stats.get(
        "targets", pd.Series(np.nan, index=stats.index)
    ).notna()
    stats["touches"] = (
        stats.get("carries", 0).fillna(0) + stats.get("targets", 0).fillna(0)
    ).where(recorded)

    for label, parts in (
        ("total_yards", ("rushing_yards", "receiving_yards")),
        ("total_touchdowns", ("rushing_tds", "receiving_tds")),
    ):
        present = [c for c in parts if c in stats.columns]
        stats[label] = sum(stats[c].fillna(0) for c in present) if present else np.nan

    if {"receptions", "targets"} <= set(stats.columns):
        stats["reception_rate"] = (
            stats["receptions"] / stats["targets"].replace(0, np.nan)
        ).clip(upper=1.0)
    else:
        stats["reception_rate"] = np.nan

    return stats


def _add_outcome(df: pd.DataFrame, pbp: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Attach the played flag and the three-valued touchdown label.

    A game that has not kicked off carries NaN rather than 0, so an upcoming
    week can never be mistaken for a week in which nobody scored -- not by the
    trainer, not by the evaluator, and not by the page.
    """
    appeared = stats[PLAYER_WEEK].drop_duplicates().assign(_appeared=1)
    df = df.merge(appeared, on=PLAYER_WEEK, how="left")

    completed = set(pbp.loc[pbp["game_id"].notna(), "game_id"].unique())
    df["played"] = np.where(df["game_id"].isin(completed) & df["_appeared"].notna(), 1, 0)

    scorers = (
        pbp[pbp["touchdown"] == 1][["td_player_id", "game_id"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"td_player_id": "player_id"})
        .assign(_scored=1)
    )
    df = df.merge(scorers, on=["player_id", "game_id"], how="left")
    df["touchdown"] = np.where(
        df["_scored"].notna(), 1.0, np.where(df["played"] == 1, 0.0, np.nan)
    )
    return df.drop(columns=["_appeared", "_scored"])


def _add_rolling_usage(
    df: pd.DataFrame, stats: pd.DataFrame, pbp: pd.DataFrame, seasons: List[int]
) -> pd.DataFrame:
    """Rolling individual usage, including red-zone work and snap share."""
    stats = stats.merge(_red_zone_touches(pbp), on=PLAYER_WEEK, how="left")
    stats = stats.merge(_snap_share(seasons), on=PLAYER_WEEK, how="left")
    stats = _extend_to_scheduled(stats, df)

    # Someone who appeared but never reached the red zone had zero touches
    # there, which is information; someone who did not appear has none, which
    # is the absence of it.
    stats["red_zone_touches"] = stats["red_zone_touches"].fillna(0).where(stats["touches"].notna())

    stats = stats.sort_values(PLAYER_WEEK).reset_index(drop=True)
    stats = add_ewma(
        stats,
        "player_id",
        {
            "targets": "targets_ewma",
            "touches": "touches_ewma",
            "carries": "carries_ewma",
            "total_yards": "total_yards_ewma",
            "total_touchdowns": "total_touchdowns_ewma",
            "reception_rate": "reception_rate_ewma",
            "red_zone_touches": "red_zone_touches_ewma",
            "snap_share": "snap_share_ewma",
        },
    )
    stats = _add_breakouts(stats)

    rolling = [c for c in stats.columns if c.endswith(("_ewma", "_normalized"))]
    return df.merge(stats[PLAYER_WEEK + rolling], on=PLAYER_WEEK, how="left")


def _extend_to_scheduled(stats: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add empty stat rows for games that have not been played.

    Rolling features are computed from the shifted series, so a player only
    receives a value for a week that exists as a row. Without a placeholder the
    upcoming week carries no usage history at all and every player is filtered
    out as unused. The row is left as NaN rather than zero -- a zero would enter
    the averages as a genuine scoreless outing.
    """
    scheduled = df[PLAYER_WEEK].drop_duplicates()
    missing = scheduled.merge(stats[PLAYER_WEEK], on=PLAYER_WEEK, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns=["_merge"])
    if missing.empty:
        return stats

    logger.info("Added %s placeholder rows for unplayed games", len(missing))
    return pd.concat([stats, missing], ignore_index=True)


def _red_zone_touches(pbp: pd.DataFrame) -> pd.DataFrame:
    """Targets plus carries inside the opponent's 20-yard line."""
    if "yardline_100" not in pbp.columns:
        return pd.DataFrame(columns=PLAYER_WEEK + ["red_zone_touches"])

    inside = pbp["yardline_100"] <= 20
    frames = []
    for id_column, play_column in (("receiver_player_id", "pass"), ("rusher_player_id", "rush")):
        if id_column not in pbp.columns or play_column not in pbp.columns:
            continue
        plays = pbp[inside & (pbp[play_column] == 1) & pbp[id_column].notna()]
        frames.append(
            plays.groupby([id_column, "season", "week"])
            .size()
            .reset_index(name="n")
            .rename(columns={id_column: "player_id"})
        )

    if not frames:
        return pd.DataFrame(columns=PLAYER_WEEK + ["red_zone_touches"])

    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby(PLAYER_WEEK)["n"].sum().reset_index(name="red_zone_touches")


def _snap_share(seasons: List[int]) -> pd.DataFrame:
    """Share of offensive snaps played, the cleanest single usage signal."""
    snaps = sources.load_optional(
        "snap_counts", seasons, ["pfr_player_id", "season", "week", "offense_pct", "player", "team"]
    )
    if snaps.empty:
        return pd.DataFrame(columns=PLAYER_WEEK + ["snap_share"])

    # Snap counts key on Pro Football Reference ids, so they are joined back to
    # gsis ids through the player crosswalk.
    players = sources.load_optional("players", seasons, ["gsis_id", "pfr_id"])
    if players.empty:
        return pd.DataFrame(columns=PLAYER_WEEK + ["snap_share"])

    snaps = snaps.merge(
        players.rename(columns={"pfr_id": "pfr_player_id", "gsis_id": "player_id"}),
        on="pfr_player_id",
        how="inner",
    )
    snaps["snap_share"] = pd.to_numeric(snaps["offense_pct"], errors="coerce")
    return snaps[PLAYER_WEEK + ["snap_share"]].dropna(subset=["player_id"]).drop_duplicates(
        subset=PLAYER_WEEK
    )


def _add_breakouts(stats: pd.DataFrame) -> pd.DataFrame:
    """Carry a standout previous game as its own signal.

    Winsorization deliberately flattens these spikes out of the rolling average,
    so they would otherwise be lost. Expressed as multiples of the threshold
    that defines them: an absolute scale, already comparable across positions,
    and immune to the near-zero weekly averages that make a sparse column
    impossible to normalize by position.
    """
    for source, destination, threshold in (
        ("total_touchdowns", "recent_breakout_tds", BREAKOUT_TDS),
        ("total_yards", "recent_breakout_yards", BREAKOUT_YARDS),
    ):
        previous = stats.groupby("player_id")[source].shift(1)
        stats[f"{destination}_normalized"] = previous.where(previous >= threshold, 0) / threshold
    return stats


def _add_team_context(df: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Rolling offensive volume for each team."""
    if "posteam" not in pbp.columns:
        df["team_play_volume_ewma"] = np.nan
        df["team_red_zone_volume_ewma"] = np.nan
        return df

    plays = pbp.groupby(["posteam", "season", "week"]).size().reset_index(name="plays")
    red_zone = (
        pbp[pbp["yardline_100"] <= 20]
        .groupby(["posteam", "season", "week"])
        .size()
        .reset_index(name="red_zone_plays")
    )
    team = plays.merge(red_zone, on=["posteam", "season", "week"], how="left")
    team = team.rename(columns={"posteam": "team"}).sort_values(TEAM_WEEK)
    team = add_ewma(
        team,
        "team",
        {"plays": "team_play_volume_ewma", "red_zone_plays": "team_red_zone_volume_ewma"},
    )

    columns = ["team_play_volume_ewma", "team_red_zone_volume_ewma"]
    df = df.merge(team[TEAM_WEEK + columns], on=TEAM_WEEK, how="left")
    return _carry_forward(df, "team", columns)


def _add_defense(df: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Rolling production allowed by the opposing defense."""
    columns = ["def_tds_allowed_ewma", "def_red_zone_tds_allowed_ewma", "def_yards_allowed_ewma"]
    if "defteam" not in pbp.columns:
        for column in columns:
            df[column] = np.nan
        return df

    allowed = (
        pbp.groupby(["defteam", "season", "week"])
        .agg(tds=("touchdown", "sum"), yards=("yards_gained", "sum"))
        .reset_index()
    )
    red_zone = (
        pbp[(pbp["yardline_100"] <= 20) & (pbp["touchdown"] == 1)]
        .groupby(["defteam", "season", "week"])
        .size()
        .reset_index(name="red_zone_tds")
    )
    allowed = allowed.merge(red_zone, on=["defteam", "season", "week"], how="left")
    allowed["red_zone_tds"] = allowed["red_zone_tds"].fillna(0)

    allowed = allowed.rename(columns={"defteam": "team"}).sort_values(TEAM_WEEK)
    allowed = add_ewma(
        allowed,
        "team",
        {
            "tds": "def_tds_allowed_ewma",
            "red_zone_tds": "def_red_zone_tds_allowed_ewma",
            "yards": "def_yards_allowed_ewma",
        },
    )

    # Joined on the opponent: these describe the defense this player faces.
    defense = allowed[TEAM_WEEK + columns].rename(columns={"team": "against"})
    df = df.merge(defense, on=["against", "season", "week"], how="left")
    return _carry_forward(df, "against", columns)


def _carry_forward(df: pd.DataFrame, group: str, columns: List[str]) -> pd.DataFrame:
    """Hold the last known value forward, then fall back to the league median.

    An upcoming game has no play-by-play of its own, so without this the week
    being predicted would carry no team or defensive context at all.
    """
    df = df.sort_values([group, "season", "week"])
    for column in columns:
        df[column] = df.groupby(group)[column].ffill()
        df[column] = df[column].fillna(df[column].median())
    return df


def _add_injuries(df: pd.DataFrame, seasons: List[int]) -> pd.DataFrame:
    """Attach the official injury report.

    The previous build displayed an availability status but never acted on it,
    so a player ruled out could still head the shortlist.
    """
    injuries = sources.load_optional(
        "injuries", seasons, ["gsis_id", "season", "week", "report_status"]
    )
    if injuries.empty:
        df["report_status"] = "Healthy"
        df["injury_severity"] = 0
        return df

    injuries = injuries.rename(columns={"gsis_id": "player_id"}).drop_duplicates(
        subset=PLAYER_WEEK
    )
    df = df.merge(injuries, on=PLAYER_WEEK, how="left")
    df["report_status"] = df["report_status"].fillna("Healthy").replace("", "Healthy")
    df["injury_severity"] = df["report_status"].map(INJURY_SEVERITY).fillna(0).astype(int)
    return df


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Depth-chart rank, history requirement and tidy-up."""
    # Rank within position group, e.g. RB1 vs RB3. Lower is better; unknown
    # depth is treated as buried rather than as a starter.
    df["depth_chart_rank"] = (
        df.groupby(["team", "season", "week", "position"])["snap_share_ewma"]
        .rank(ascending=False, method="min")
        .fillna(9)
    )

    appearances = df[df["played"] == 1].groupby("player_id").size()
    experienced = appearances[appearances >= MIN_GAMES_PLAYED].index
    before = len(df)
    df = df[df["player_id"].isin(experienced)].copy()
    logger.info("History filter kept %s of %s rows", len(df), before)

    return df.sort_values(PLAYER_WEEK).reset_index(drop=True)
