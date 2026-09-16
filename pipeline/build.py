"""
Pipeline entry point: fetch, train, predict, and write the site's data file.

Run as ``python -m pipeline.build``. Everything expensive happens here, offline,
so the hosted site only ever has to read a small JSON document.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd

from . import features, sources
from .config import FEATURES, INJURY_EXCLUDE_AT, OUTPUT_PATH
from .model import TouchdownModel

logger = logging.getLogger(__name__)

# Plain-English names for the page. A driver labelled "red zone touches" is
# something a reader can act on; "red_zone_touches_ewma_position_normalized"
# is not.
FEATURE_LABELS = {
    "targets_ewma_position_normalized": "Targets vs position",
    "touches_ewma_position_normalized": "Touches vs position",
    "total_yards_ewma_position_normalized": "Yards vs position",
    "total_touchdowns_ewma_position_normalized": "Recent scoring",
    "reception_rate_ewma_position_normalized": "Catch rate",
    "red_zone_touches_ewma_position_normalized": "Red zone work",
    "red_zone_touch_share_ewma_position_normalized": "Red zone share",
    "snap_share_ewma_position_normalized": "Snap share vs position",
    "snap_share_ewma": "Snap share",
    "depth_chart_rank": "Depth chart rank",
    "recent_breakout_tds_normalized": "Breakout scoring game",
    "recent_breakout_yards_normalized": "Breakout yardage game",
    "implied_team_total": "Team implied points",
    "game_total": "Game total",
    "spread_line": "Point spread",
    "team_win_probability": "Win probability",
    "is_dome": "Indoor game",
    "wind": "Wind",
    "days_rest": "Days rest",
    "team_play_volume_ewma": "Team play volume",
    "team_red_zone_volume_ewma": "Team red zone volume",
    "def_tds_allowed_ewma": "Defense TDs allowed",
    "def_red_zone_tds_allowed_ewma": "Defense red zone TDs allowed",
    "def_yards_allowed_ewma": "Defense yards allowed",
    "injury_severity": "Injury report",
}

DRIVERS_PER_PLAYER = 6


def run(season: int = None, week: int = None, output=OUTPUT_PATH) -> dict:
    season, week = sources.season_and_week(season, week)
    logger.info("Building predictions for season %s week %s", season, week)

    frame = features.build(season, week)
    history = frame[
        (frame["season"] < season) | ((frame["season"] == season) & (frame["week"] < week))
    ]
    history = history[history["played"] == 1]
    upcoming = frame[(frame["season"] == season) & (frame["week"] == week)]

    if upcoming.empty:
        raise SystemExit(f"No players found for season {season} week {week}")

    # Anyone doubtful or worse is off the board; the report is why the feature
    # exists, and showing a ruled-out player at the top would be worse than
    # showing nothing.
    available = upcoming[upcoming["injury_severity"] < INJURY_EXCLUDE_AT]
    logger.info(
        "%s training rows; %s of %s players available", len(history), len(available), len(upcoming)
    )

    model = TouchdownModel().train(history)
    ranked, matrix = model.predict(available)
    contributions = model.explain(matrix)

    document = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season,
        "week": week,
        "model": _model_summary(model),
        "players": _players(ranked, contributions, model),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=1))
    logger.info("Wrote %s players to %s", len(document["players"]), output)
    return document


def _model_summary(model: TouchdownModel) -> dict:
    top = sorted(model.importance.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "test": _clean(model.metrics["test"]),
        "validation": _clean(model.metrics["validation"]),
        "training_rows": model.metrics["training_rows"],
        "features_used": len(FEATURES),
        "top_features": [
            {"feature": name, "label": FEATURE_LABELS.get(name, name), "importance": round(value, 4)}
            for name, value in top
        ],
    }


def _players(ranked: pd.DataFrame, contributions, model: TouchdownModel) -> List[dict]:
    names = model.pipeline.features
    players = []

    for position, row in enumerate(ranked.itertuples()):
        drivers = []
        if contributions is not None:
            # Column j of the matrix is names[j] by construction, so the labels
            # cannot drift out of step with the values.
            weights = contributions[position]
            order = np.argsort(np.abs(weights))[::-1][:DRIVERS_PER_PLAYER]
            drivers = [
                {
                    "label": FEATURE_LABELS.get(names[j], names[j]),
                    "effect": round(float(weights[j]), 4),
                    "value": _number(getattr(row, names[j], None)),
                }
                for j in order
                if abs(float(weights[j])) > 1e-6
            ]

        players.append(
            {
                "rank": position + 1,
                "id": str(row.player_id),
                "name": str(row.player_name),
                "position": str(row.position),
                "team": str(row.team),
                "against": str(row.against),
                "home": bool(getattr(row, "is_home", 0)),
                "probability": round(float(row.probability), 4),
                "status": str(getattr(row, "report_status", "Healthy")),
                "implied_total": _number(getattr(row, "implied_team_total", None)),
                "game_total": _number(getattr(row, "game_total", None)),
                "snap_share": _number(getattr(row, "snap_share_ewma", None)),
                "drivers": drivers,
            }
        )
    return players


def _clean(metrics: dict) -> dict:
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in metrics.items()}


def _number(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(value) else round(value, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NFL touchdown predictions")
    parser.add_argument("--season", type=int, help="defaults to the current season")
    parser.add_argument("--week", type=int, help="defaults to the next unplayed week")
    parser.add_argument("--output", type=str, help="path for predictions.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from pathlib import Path

    run(args.season, args.week, Path(args.output) if args.output else OUTPUT_PATH)


if __name__ == "__main__":
    main()
