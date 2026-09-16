"""
Django views for the NFL touchdown prediction app.
"""

import io
import json
import logging
import os
import traceback
from functools import wraps
from typing import Optional

import numpy as np
import pandas as pd
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_http_methods

from .config import EXCLUDED_INJURY_STATUSES, FEATURES, MAX_WEEK, MODEL_PARAMS, NUMERIC_FEATURES
from .data_manager import NFLDataManager
from .filters import apply_prediction_filters
from .ml_model import NFLTouchdownModel, predict_week, train_model
from .utils import get_sleeper_injury_status_map

logger = logging.getLogger(__name__)

DISPLAY_COLUMNS = [
    "player_name", "team", "position", "against", "probability",
    "played", "touchdown", "injury_status", "season", "week", "player_id",
]
FEATURE_DISPLAY_COLUMNS = [
    "player_id", "player_name", "team", "position", "against", "injury_status", "season", "week",
]


# ------------------------------------------------------------------ helpers


def api_view(func):
    """Return JSON for every outcome, including unexpected failures."""

    @wraps(func)
    def wrapper(request, *args, **kwargs):
        try:
            return func(request, *args, **kwargs)
        except ValueError as error:
            return JsonResponse({"success": False, "message": str(error)}, status=400)
        except Exception as error:
            logger.exception("Unhandled error in %s", func.__name__)
            return JsonResponse(
                {
                    "success": False,
                    "message": f"Error: {error}",
                    "traceback": traceback.format_exc() if request.GET.get("debug") else None,
                },
                status=500,
            )

    return wrapper


def read_request(request) -> dict:
    """Parse and validate the season and week from a JSON request body."""
    if not request.body:
        raise ValueError("Request body is empty.")
    payload = json.loads(request.body.decode("utf-8"))

    try:
        season = int(payload["season"])
        week = int(payload["week"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Please provide both season and week as numbers.")

    if not 1 <= week <= MAX_WEEK:
        raise ValueError(f"Week must be between 1 and {MAX_WEEK}.")
    return {"season": season, "week": week, "payload": payload}


def store_frame(request, key: str, df: pd.DataFrame) -> None:
    """Round-trip a frame through the session as JSON."""
    request.session[key] = df.to_json(orient="split")


def load_frame(request, key: str) -> Optional[pd.DataFrame]:
    raw = request.session.get(key)
    if raw is None:
        return None
    return pd.read_json(io.StringIO(raw), orient="split")


def to_records(df: pd.DataFrame) -> list:
    """Convert to JSON-safe records, turning NaN into null.

    ``DataFrame.to_dict`` leaves NaN in place, which is not valid JSON, so every
    value is normalized here rather than at each call site.
    """
    return [
        {
            key: (None if value is None or (isinstance(value, float) and pd.isna(value)) else value)
            for key, value in record.items()
        }
        for record in df.replace({np.nan: None}).to_dict("records")
    ]


def add_injury_status(df: pd.DataFrame) -> pd.DataFrame:
    """Attach live injury status, falling back to the weekly report status."""
    if df.empty:
        df["injury_status"] = pd.Series(dtype=str)
        return df

    statuses = get_sleeper_injury_status_map()
    df = df.copy()
    df["injury_status"] = (
        df["player_id"].astype(str).str.upper().map(statuses).fillna("Unknown")
    )

    if "report_status" in df.columns:
        df["injury_status"] = df["injury_status"].where(
            df["injury_status"] != "Unknown", df["report_status"]
        )
    return df


def drop_injured(df: pd.DataFrame) -> pd.DataFrame:
    """Remove players ruled out.

    Injury status was previously displayed but never acted on, so a player
    declared out could still head the shortlist.
    """
    if not MODEL_PARAMS.get("exclude_injured", True) or "injury_status" not in df.columns:
        return df

    excluded = df["injury_status"].isin(EXCLUDED_INJURY_STATUSES)
    if excluded.any():
        logger.info("Excluded %s players listed as unavailable", int(excluded.sum()))
    return df[~excluded].copy()


def prepare_current_week(request, season: int, week: int) -> tuple:
    """Load the week to be scored and apply the shared eligibility filters."""
    current_week = load_frame(request, "current_week_data")
    if current_week is None:
        raise ValueError("Please load data first.")

    history = load_frame(request, "training_data")
    current_week, report = apply_prediction_filters(current_week, history, season, week)
    current_week = drop_injured(add_injury_status(current_week))

    if current_week.empty:
        raise ValueError(
            f"No players remain after filtering ({report.describe()}). "
            "Load the week's data first, or relax min_touches_ewma in config.py."
        )
    return current_week, history, report


# -------------------------------------------------------------------- views


@ensure_csrf_cookie
def index(request):
    return render(request, "predictions/index.html")


@api_view
@require_http_methods(["POST"])
def load_data(request):
    """Load and process NFL data for a season and week."""
    request_data = read_request(request)
    season, week = request_data["season"], request_data["week"]

    manager = NFLDataManager()
    result = manager.load_and_process_data(
        season, week, force_reload=request_data["payload"].get("force_reload", False)
    )

    store_frame(request, "training_data", result["df"])
    store_frame(request, "current_week_data", result["current_week"])
    request.session["season"] = season
    request.session["week"] = week

    return JsonResponse(
        {
            "success": True,
            "message": (
                f"Loaded {len(result['df'])} training records and "
                f"{len(result['current_week'])} players for week {week}."
            ),
            "training_records": len(result["df"]),
            "current_week_players": len(result["current_week"]),
        }
    )


@api_view
@require_http_methods(["POST"])
def train_model_view(request):
    """Train a model, refusing to overwrite an existing one."""
    return _train(request, overwrite=False)


@api_view
@require_http_methods(["POST"])
def retrain_model_view(request):
    """Retrain, replacing any existing model for the week."""
    return _train(request, overwrite=True)


def _train(request, overwrite: bool):
    request_data = read_request(request)
    season, week = request_data["season"], request_data["week"]

    df = load_frame(request, "training_data")
    if df is None:
        raise ValueError("Please load data first before training the model.")

    if not overwrite and NFLTouchdownModel(season, week).model_exists():
        return JsonResponse(
            {
                "success": False,
                "message": "A model already exists for this week. Use retrain to replace it.",
                "model_exists": True,
            }
        )

    success, message = train_model(df, season, week, FEATURES, NUMERIC_FEATURES)
    return JsonResponse({"success": success, "message": message}, status=200 if success else 400)


@api_view
@require_http_methods(["POST"])
def predict_week_view(request):
    """Rank the week's players by touchdown probability."""
    request_data = read_request(request)
    season, week = request_data["season"], request_data["week"]

    current_week, history, report = prepare_current_week(request, season, week)

    predictions, shap_values = predict_week(
        season, week, current_week, FEATURES, NUMERIC_FEATURES
    )
    predictions["probability"] = predictions["probability"].round(4)

    _export_debug_frame(history, current_week)

    display = predictions[[c for c in DISPLAY_COLUMNS if c in predictions.columns]]
    records = to_records(display)
    for record in records:
        record["feature_explanations"] = (shap_values or {}).get(str(record.get("player_id")))

    store_frame(request, "predictions", predictions)

    model = NFLTouchdownModel(season, week)
    model.load()

    return JsonResponse(
        {
            "success": True,
            "message": f"Predictions generated for {len(predictions)} players.",
            "data": records,
            "columns": list(display.columns),
            "filters_applied": report.describe(),
            "model_metrics": model.metrics,
        }
    )


@api_view
@require_http_methods(["POST"])
def get_feature_data(request):
    """Return the exact feature matrix behind the current week's predictions."""
    request_data = read_request(request)
    current_week, _, _ = prepare_current_week(
        request, request_data["season"], request_data["week"]
    )

    available = [f for f in FEATURES if f in current_week.columns]
    columns = [
        c for c in FEATURE_DISPLAY_COLUMNS + available if c in current_week.columns
    ]
    frame = current_week[columns].sort_values("player_name").reset_index(drop=True)

    return JsonResponse(
        {
            "success": True,
            "data": to_records(frame),
            "columns": columns,
            "feature_columns": available,
            "display_columns": FEATURE_DISPLAY_COLUMNS,
            "missing_features": [f for f in FEATURES if f not in current_week.columns],
            "season": request_data["season"],
            "week": request_data["week"],
            "row_count": len(frame),
            "feature_count": len(available),
            "total_features_requested": len(FEATURES),
        }
    )


@api_view
@require_http_methods(["POST"])
def export_predictions(request):
    """Download the most recent predictions as CSV."""
    predictions = load_frame(request, "predictions")
    if predictions is None:
        raise ValueError("No predictions available to export. Run predictions first.")

    season = request.session.get("season", "")
    week = request.session.get("week", "")

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = (
        f'attachment; filename="nfl_touchdown_predictions_s{season}_w{week}.csv"'
    )
    predictions.to_csv(response, index=False)
    return response


@api_view
@require_http_methods(["GET"])
def check_model_exists(request):
    try:
        season = int(request.GET["season"])
        week = int(request.GET["week"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid season or week.")

    model = NFLTouchdownModel(season, week)
    exists = model.model_exists()
    return JsonResponse(
        {"exists": exists, "metrics": model.metrics if exists and model.load() else None}
    )


def _export_debug_frame(history: Optional[pd.DataFrame], current_week: pd.DataFrame) -> None:
    """Write the combined frame to df.csv for offline inspection."""
    try:
        frames = [f for f in (history, current_week) if f is not None and not f.empty]
        if not frames:
            return
        path = os.path.join(str(settings.BASE_DIR), "df.csv")
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    except Exception:
        logger.warning("Could not write df.csv debug export", exc_info=True)
