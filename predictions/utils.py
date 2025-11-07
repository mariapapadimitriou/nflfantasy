import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict


SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
SLEEPER_CACHE_TTL_SECONDS = 60 * 15  # 15 minutes
_sleeper_injury_cache = {
    "timestamp": None,
    "data": {}
}


def match_teams(team, abbr_list):
    # Try direct match
    if team in abbr_list.values:
        return team
    # Try lower-case match
    for abbr in abbr_list:
        if team.lower() == abbr.lower():
            return abbr
    # Try partial match
    for abbr in abbr_list:
        if team.lower() in abbr.lower():
            return abbr
    return team


def american_odds_to_probability(odds):
    odds = int(odds)
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)


def get_sleeper_injury_status_map(force_refresh: bool = False) -> Dict[str, str]:
    """
    Fetch injury status information from the Sleeper API.

    Args:
        force_refresh: If True, bypass the cached result and fetch fresh data.

    Returns:
        Mapping of GSIS player id -> injury status string.
    """
    global _sleeper_injury_cache

    # Return cached copy when available and fresh
    cache_timestamp = _sleeper_injury_cache.get("timestamp")
    if (
        not force_refresh
        and cache_timestamp is not None
        and datetime.utcnow() - cache_timestamp < timedelta(seconds=SLEEPER_CACHE_TTL_SECONDS)
        and _sleeper_injury_cache.get("data")
    ):
        return _sleeper_injury_cache["data"]

    try:
        response = requests.get(SLEEPER_PLAYERS_URL, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        # Log warning and fall back to cached data if available
        print(f"[Warning] Unable to fetch Sleeper injury data: {exc}")
        return _sleeper_injury_cache.get("data", {})

    injury_map: Dict[str, str] = {}

    if isinstance(payload, dict):
        for player in payload.values():
            gsis_id = player.get("gsis_id")
            if not gsis_id:
                continue

            status = player.get("injury_status")
            # Fall back to general status (Active, IR, etc.) when specific injury status absent
            if not status:
                status = player.get("status")

            # Default to Healthy when no status provided
            if not status:
                status = "Healthy"

            injury_map[str(gsis_id).upper()] = status

    _sleeper_injury_cache = {
        "timestamp": datetime.utcnow(),
        "data": injury_map
    }

    return injury_map

