"""
Player eligibility filters.

These filters decide which players are modelled and which are scored. Training
and prediction have to agree: a filter applied on only one side teaches the
model on one population and serves another. Every filter therefore lives here
once and is called from both paths.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from .config import EWMA_WEEKS, INDIVIDUAL_PERFORMANCE_FEATURES, MODEL_PARAMS

logger = logging.getLogger(__name__)


@dataclass
class FilterReport:
    """Running tally of what each filter removed, for diagnostics."""

    steps: List[str] = field(default_factory=list)

    def record(self, name: str, before: int, after: int) -> None:
        if before != after:
            self.steps.append(f"{name}: {before} -> {after} ({before - after} removed)")
            logger.debug("Filter %s removed %s rows", name, before - after)

    def describe(self) -> str:
        return "; ".join(self.steps) if self.steps else "no rows removed"


def filter_individual_performance(
    df: pd.DataFrame, report: Optional[FilterReport] = None
) -> pd.DataFrame:
    """Require genuine individual usage, not just a seat on a productive offense.

    A player qualifies when at least two of the individual usage features are
    present and meet the threshold. Applied identically during training and
    prediction so the model never learns on a population it will not score.
    """
    if not MODEL_PARAMS.get("require_individual_performance", True):
        return df

    available = [f for f in INDIVIDUAL_PERFORMANCE_FEATURES if f in df.columns]
    if not available:
        return df

    threshold = MODEL_PARAMS.get("min_individual_performance_threshold", 0.0)
    qualifying = sum(
        (df[feature].notna() & (df[feature] >= threshold)).astype(int)
        for feature in available
    )
    required = min(2, len(available))

    before = len(df)
    df = df[qualifying >= required].copy()
    if report:
        report.record("individual_performance", before, len(df))
    return df


def filter_min_usage(
    df: pd.DataFrame, report: Optional[FilterReport] = None
) -> pd.DataFrame:
    """Drop players with too little recent usage to be credible TD threats.

    Quarterbacks are judged on rushing volume, since a passing touchdown does
    not count for the thrower under this target definition.
    """
    if not MODEL_PARAMS.get("min_usage_filter", True):
        return df
    if "touches_ewma" not in df.columns or "position" not in df.columns:
        return df

    def above(column: str, minimum: float) -> pd.Series:
        if column not in df.columns:
            return pd.Series(False, index=df.index)
        return df[column].notna() & (df[column] >= minimum)

    is_qb = df["position"] == "QB"
    has_red_zone = above("red_zone_touches_ewma", MODEL_PARAMS["min_red_zone_touches_ewma"])

    skill_ok = ~is_qb & (above("touches_ewma", MODEL_PARAMS["min_touches_ewma"]) | has_red_zone)
    qb_ok = is_qb & (above("carries_ewma", MODEL_PARAMS["min_qb_carries_ewma"]) | has_red_zone)

    before = len(df)
    df = df[skill_ok | qb_ok].copy()
    if report:
        report.record("min_usage", before, len(df))
    return df


def filter_min_history(
    df: pd.DataFrame,
    history: Optional[pd.DataFrame] = None,
    season: Optional[int] = None,
    week: Optional[int] = None,
    report: Optional[FilterReport] = None,
) -> pd.DataFrame:
    """Require enough prior games for the EWMA features to mean anything.

    Early in a season there are not yet enough games in the current year, so
    games from prior seasons count too. From week 4 onward only the current
    season counts, which keeps long-retired players out of the pool.
    """
    if "player_id" not in df.columns:
        return df

    if history is None or history.empty:
        # Without a history to count against, every player would show a single
        # appearance and be dropped. The data build already enforces a minimum
        # game count, so deferring to it is the safe answer here.
        return df

    relevant = history
    early_season = week is not None and week <= MODEL_PARAMS["current_season_only_from_week"]
    if not early_season and season is not None and "season" in history.columns:
        relevant = history[history["season"] == season]

    counts = (
        pd.concat([relevant[["player_id"]], df[["player_id"]]], ignore_index=True)
        .groupby("player_id")
        .size()
    )
    eligible = counts[counts >= EWMA_WEEKS].index

    before = len(df)
    df = df[df["player_id"].isin(eligible)].copy()
    if report:
        report.record("min_history", before, len(df))
    return df


def filter_played(
    df: pd.DataFrame, report: Optional[FilterReport] = None
) -> pd.DataFrame:
    """Drop players who did not appear, for weeks that have actually been played.

    ``played`` is NaN before kickoff, so an upcoming week passes through
    untouched rather than being emptied or, worse, treated as a week in which
    nobody scored.
    """
    if "played" not in df.columns or len(df) == 0:
        return df
    if df["played"].isna().all():
        return df

    before = len(df)
    df = df[df["played"] == 1].copy()
    if report:
        report.record("played", before, len(df))
    return df


def apply_prediction_filters(
    df: pd.DataFrame,
    history: Optional[pd.DataFrame] = None,
    season: Optional[int] = None,
    week: Optional[int] = None,
) -> tuple:
    """Run the full eligibility chain used before scoring a week.

    Returns the filtered frame and a report describing each step.
    """
    report = FilterReport()
    df = filter_min_history(df, history, season, week, report)
    df = filter_played(df, report)
    df = filter_min_usage(df, report)
    df = filter_individual_performance(df, report)
    return df, report
