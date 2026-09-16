"""
Data loading from nflverse via nflreadpy.

nflverse is the open data project behind nflfastR: free, no API key, updated
within hours of each game, and the same source the public NFL analytics
community works from. Everything the model needs -- play-by-play, weekly stat
lines, weekly rosters, snap counts, injury reports and closing betting lines --
comes from it, so there is one provenance to reason about rather than five.
"""

import logging
from functools import lru_cache
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# Reference tables that cover every season at once and take no season argument.
SEASONLESS = {"players", "teams"}


def _load(name: str, seasons: Tuple[int, ...]) -> pd.DataFrame:
    import nflreadpy as nfl

    loader = getattr(nfl, f"load_{name}")
    frame = (loader() if name in SEASONLESS else loader(seasons=list(seasons))).to_pandas()
    logger.info("Loaded %s: %s rows", name, len(frame))
    return frame


@lru_cache(maxsize=None)
def _cached(name: str, seasons: Tuple[int, ...]) -> pd.DataFrame:
    return _load(name, seasons)


def load(name: str, seasons: List[int]) -> pd.DataFrame:
    """Fetch an nflverse table, memoized per season set for this process."""
    return _cached(name, tuple(sorted(seasons))).copy()


def load_optional(name: str, seasons: List[int], columns: List[str]) -> pd.DataFrame:
    """Fetch a table that may be unpublished for the current season.

    Snap counts and injury reports lag the schedule, and early in a season they
    can be missing entirely. Returning an empty frame with the expected columns
    lets the caller merge unconditionally and get NaN, which the model reads as
    "unknown" rather than as a value.
    """
    try:
        frame = load(name, seasons)
    except Exception:
        logger.warning("Source %s unavailable; continuing without it", name, exc_info=True)
        return pd.DataFrame(columns=columns)

    missing = [c for c in columns if c not in frame.columns]
    if missing:
        logger.warning("Source %s is missing %s", name, ", ".join(missing))
        return pd.DataFrame(columns=columns)
    return frame[columns]


def current_target_week(season: int) -> int:
    """The earliest scheduled week that has not been played yet.

    Falls back to the last scheduled week once a season is complete, so the
    pipeline still produces output out of season.
    """
    schedules = load("schedules", [season])
    unplayed = schedules[schedules["away_score"].isna()]
    if unplayed.empty:
        return int(schedules["week"].max())
    return int(unplayed["week"].min())


def season_and_week(season: int = None, week: int = None) -> Tuple[int, int]:
    """Resolve the season and week to predict, defaulting to the live one."""
    from datetime import date

    if season is None:
        today = date.today()
        # An NFL season is labelled by the year it starts, and runs into the
        # following February.
        season = today.year if today.month >= 3 else today.year - 1
    if week is None:
        week = current_target_week(season)
    return season, week
