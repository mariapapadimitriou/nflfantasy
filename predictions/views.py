"""
Django views for NFL Touchdown Predictions
"""
import json
import os
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import ensure_csrf_cookie
from django.contrib import messages
from django.conf import settings
from functools import wraps
import pandas as pd
import numpy as np
import traceback

from .config import FEATURES, NUMERIC_FEATURES
from .data_manager import NFLDataManager
from .ml_model import (
    train_model,
    predict_week,
    retrain_model,
    NFLTouchdownModel,
    compare_breakout_feature_sets,
)
from .utils import get_sleeper_injury_status_map


def json_response_view(func):
    """Decorator to ensure all responses are JSON, even on errors"""
    @wraps(func)
    def wrapper(request, *args, **kwargs):
        try:
            return func(request, *args, **kwargs)
        except json.JSONDecodeError as e:
            return JsonResponse({
                'success': False,
                'message': f'Invalid JSON in request: {str(e)}'
            }, status=400)
        except Exception as e:
            # Log the full traceback for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}")
            return JsonResponse({
                'success': False,
                'message': f'Error: {str(e)}',
                'traceback': traceback.format_exc() if request.GET.get('debug') else None
            }, status=500)
    return wrapper


@ensure_csrf_cookie
def index(request):
    """Main homepage view"""
    return render(request, 'predictions/index.html')


@json_response_view
@require_http_methods(["POST"])
def load_data(request):
    """Load NFL data for a given season and week"""
    try:
        # Handle empty body
        if not request.body:
            return JsonResponse({
                'success': False,
                'message': 'Request body is empty.'
            }, status=400)
        
        data = json.loads(request.body.decode('utf-8'))
        season = int(data.get('season'))
        week = int(data.get('week'))
        force_reload = data.get('force_reload', False)
        
        # Validate inputs
        if not season or not week:
            return JsonResponse({
                'success': False,
                'message': 'Please provide both season and week.'
            }, status=400)
        
        # Validate week range
        if week < 1 or week > 20:
            return JsonResponse({
                'success': False,
                'message': 'Week must be between 1 and 20.'
            }, status=400)
        
        # Load data
        data_manager = NFLDataManager()
        result = data_manager.load_and_process_data(season, week, force_reload=force_reload)
        
        # Store data in session (handle NaN values in JSON)
        # Convert to dict, clean NaN values, then back to DataFrame for JSON storage
        def clean_nan_for_json(df):
            """Convert DataFrame to dict, replace NaN with None, then back to DataFrame"""
            df_dict = df.to_dict('records')
            for record in df_dict:
                for key, value in record.items():
                    try:
                        if pd.isna(value):
                            record[key] = None
                    except (TypeError, ValueError):
                        pass
            return pd.DataFrame(df_dict)
        
        df_clean = clean_nan_for_json(result['df'])
        current_week_clean = clean_nan_for_json(result['current_week'])
        
        request.session['training_data'] = df_clean.to_json(orient='split')
        request.session['current_week_data'] = current_week_clean.to_json(orient='split')
        request.session['season'] = season
        request.session['week'] = week
        
        return JsonResponse({
            'success': True,
            'message': f'Data loaded successfully! Training data: {len(result["df"])} records, Current week: {len(result["current_week"])} players.',
            'training_records': len(result['df']),
            'current_week_players': len(result['current_week'])
        })
        
    except ValueError as ve:
        return JsonResponse({
            'success': False,
            'message': str(ve)
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Error loading data: {str(e)}',
            'traceback': traceback.format_exc()
        }, status=500)


@json_response_view
@require_http_methods(["POST"])
def train_model_view(request):
    """Train the model for a given season and week"""
    try:
        if not request.body:
            return JsonResponse({
                'success': False,
                'message': 'Request body is empty.'
            }, status=400)
        
        data = json.loads(request.body.decode('utf-8'))
        season = int(data.get('season'))
        week = int(data.get('week'))
        
        # Check if training data exists in session
        if 'training_data' not in request.session:
            return JsonResponse({
                'success': False,
                'message': 'Please load data first before training the model.'
            }, status=400)
        
        # Get training data from session
        df = pd.read_json(request.session['training_data'], orient='split')
        
        # Check if model already exists
        model = NFLTouchdownModel(season, week)
        if model.model_exists():
            return JsonResponse({
                'success': False,
                'message': 'Model already exists for this season and week. Use retrain to overwrite.',
                'model_exists': True
            })
        
        # Compare breakout feature configurations before final training
        compare_breakout_feature_sets(df.copy(), season, week, FEATURES, NUMERIC_FEATURES)
        
        # Train model
        success, msg = train_model(df, season, week, FEATURES, NUMERIC_FEATURES)
        
        if success:
            return JsonResponse({
                'success': True,
                'message': msg
            })
        else:
            return JsonResponse({
                'success': False,
                'message': msg
            }, status=400)
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Error training model: {str(e)}',
            'traceback': traceback.format_exc()
        }, status=500)


@json_response_view
@require_http_methods(["POST"])
def retrain_model_view(request):
    """Retrain/overwrite existing model"""
    try:
        if not request.body:
            return JsonResponse({
                'success': False,
                'message': 'Request body is empty.'
            }, status=400)
        
        data = json.loads(request.body.decode('utf-8'))
        season = int(data.get('season'))
        week = int(data.get('week'))
        
        # Check if training data exists in session
        if 'training_data' not in request.session:
            return JsonResponse({
                'success': False,
                'message': 'Please load data first before training the model.'
            }, status=400)
        
        # Get training data from session
        df = pd.read_json(request.session['training_data'], orient='split')
        
        # Compare breakout feature configurations before retraining
        compare_breakout_feature_sets(df.copy(), season, week, FEATURES, NUMERIC_FEATURES)
        
        # Train model (this will overwrite existing)
        success, msg = train_model(df, season, week, FEATURES, NUMERIC_FEATURES)
        
        if success:
            return JsonResponse({
                'success': True,
                'message': msg
            })
        else:
            return JsonResponse({
                'success': False,
                'message': msg
            }, status=400)
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Error retraining model: {str(e)}',
            'traceback': traceback.format_exc()
        }, status=500)


@json_response_view
@require_http_methods(["POST"])
def predict_week_view(request):
    """Make predictions for a given week"""
    try:
        if not request.body:
            return JsonResponse({
                'success': False,
                'message': 'Request body is empty.'
            }, status=400)
        
        data = json.loads(request.body.decode('utf-8'))
        season = int(data.get('season'))
        week = int(data.get('week'))
        
        # Check if current week data exists in session
        if 'current_week_data' not in request.session:
            return JsonResponse({
                'success': False,
                'message': 'Please load data first before making predictions.'
            }, status=400)
        
        # Get current week data from session (NaN values will be preserved)
        current_week = pd.read_json(request.session['current_week_data'], orient='split')
        
        # Filter players with fewer than EWMA_WEEKS records (minimum history requirement)
        # NOTE: For weeks 1-3, count all games (historical + current season) since not enough current season games exist
        #       For weeks 4+, count only current season games to ensure players have recent activity
        from .config import EWMA_WEEKS
        if 'training_data' in request.session:
            training_data = pd.read_json(request.session['training_data'], orient='split')
            if week <= 3:
                # Early weeks (1-3): Count all games across all seasons (historical + current)
                all_data = pd.concat([training_data[["player_id"]], current_week[["player_id"]]], ignore_index=True)
                player_record_counts = all_data.groupby("player_id").size()
                filter_description = f"total games (all seasons)"
            else:
                # Weeks 4+: Count only current season games
                if 'season' in training_data.columns:
                    current_season_data = training_data[training_data["season"] == season][["player_id"]].copy()
                else:
                    # If season column doesn't exist, assume all training_data is from current season
                    current_season_data = training_data[["player_id"]].copy()
                all_current_season_data = pd.concat([current_season_data, current_week[["player_id"]]], ignore_index=True)
                player_record_counts = all_current_season_data.groupby("player_id").size()
                filter_description = f"games in current season ({season})"
            
            players_with_enough_history = player_record_counts[player_record_counts >= EWMA_WEEKS].index
            before_count = len(current_week)
            current_week = current_week[current_week["player_id"].isin(players_with_enough_history)].copy()
            after_count = len(current_week)
        else:
            # If no training data, count only current week records (should be 0, but handle edge case)
            player_record_counts = current_week.groupby("player_id").size()
            players_with_enough_history = player_record_counts[player_record_counts >= EWMA_WEEKS].index
            before_count = len(current_week)
            current_week = current_week[current_week["player_id"].isin(players_with_enough_history)].copy()
            after_count = len(current_week)
        
        # Filter out players who haven't played (no stats) - they shouldn't be predicted
        # NOTE: For future weeks (games haven't happened yet), skip this filter since played=0 for everyone
        # Instead, rely on EWMA features filter which uses historical data
        if 'played' in current_week.columns:
            # Check if this is a future week (all players have played=0)
            all_not_played = (current_week['played'] == 0).all() if len(current_week) > 0 else False
            if not all_not_played:
                # This is a past week - filter out players who didn't play
                before_count = len(current_week)
                current_week = current_week[current_week['played'] == 1].copy()
                after_count = len(current_week)
        
        # Apply minimum usage filter to improve precision (filter out inactive players)
        # Position-aware: QBs use different criteria than skill position players
        from .config import MODEL_PARAMS
        if MODEL_PARAMS.get("min_usage_filter", True):
            min_touches = MODEL_PARAMS.get("min_touches_ewma", 3.0)
            # Use touches_ewma for filtering (still use raw touches for threshold, not normalized)
            if "touches_ewma" in current_week.columns and "position" in current_week.columns:
                before_count = len(current_week)
                
                # Separate filters for QBs vs other positions
                is_qb = current_week["position"] == "QB"
                is_skill_position = ~is_qb
                
                # For skill positions (WR/TE/RB): use touches_ewma or red_zone_touches_ewma
                has_min_touches = current_week["touches_ewma"].notna() & (current_week["touches_ewma"] >= min_touches)
                has_red_zone = False
                if "red_zone_touches_ewma" in current_week.columns:
                    has_red_zone = current_week["red_zone_touches_ewma"].notna() & (current_week["red_zone_touches_ewma"] > 0)
                
                # For QBs: must have rushing attempts (carries_ewma) since they score rushing TDs
                # QBs with low rushing attempts shouldn't be predicted
                qb_has_rushing = False
                if "carries_ewma" in current_week.columns:
                    # QBs need at least some rushing attempts to be legitimate TD threats
                    qb_min_carries = 2.0  # Lower threshold for QBs (they don't rush as much)
                    qb_has_rushing = current_week["carries_ewma"].notna() & (current_week["carries_ewma"] >= qb_min_carries)
                qb_has_red_zone = False
                if "red_zone_touches_ewma" in current_week.columns:
                    # QBs can also pass if they have red zone rushing attempts
                    qb_has_red_zone = current_week["red_zone_touches_ewma"].notna() & (current_week["red_zone_touches_ewma"] > 0)
                
                # Combine: skill positions use touches/red_zone, QBs use carries/red_zone
                skill_position_mask = is_skill_position & (has_min_touches | has_red_zone)
                qb_mask = is_qb & (qb_has_rushing | qb_has_red_zone)
                keep_mask = skill_position_mask | qb_mask
                
                current_week = current_week[keep_mask].copy()
                after_count = len(current_week)
        
        # Additional filter: Remove players with no meaningful stats (all key features are NaN or 0)
        # Even if played=1, they might have no actual usage stats
        # This prevents players with no recent activity from getting high probabilities
        # BUT: Be less aggressive - only filter if we have enough players to spare
        # NOTE: EWMA features may be NaN for early weeks or players with no history, so we need to be careful
        if len(current_week) > 0:
            before_count = len(current_week)
            
            # Key usage features that indicate actual player activity
            # These are the most important indicators of player usage
            key_usage_features = ['targets_ewma', 'receptions_ewma', 'carries_ewma', 'touches_ewma']
            
            # Check if player has at least one meaningful stat (not NaN and > 0)
            has_stats = pd.Series(False, index=current_week.index)
            
            # Count how many key features each player has
            for feature in key_usage_features:
                if feature in current_week.columns:
                    # Player has this stat if it's not NaN and > 0
                    has_feature = current_week[feature].notna() & (current_week[feature] > 0)
                    has_stats = has_stats | has_feature
            
            # Position-specific stats are already covered by red_zone_touches_ewma
            # (which includes both targets and carries in the red zone)
            
            # Note: designed_rush_attempts_ewma has been removed from features
            
            # For red zone activity
            if 'red_zone_touches_ewma' in current_week.columns:
                rz_stats = current_week['red_zone_touches_ewma'].notna() & (current_week['red_zone_touches_ewma'] > 0)
                has_stats = has_stats | rz_stats
            
            # Only filter if we'll still have a reasonable number of players left after filtering
            # Require at least 10 players or 50% of original, whichever is smaller
            players_with_stats = has_stats.sum()
            min_players_needed = min(10, max(1, before_count // 2))
            
            if players_with_stats >= min_players_needed:
                # Keep only players with at least one meaningful stat
                current_week = current_week[has_stats].copy()
                after_count = len(current_week)
                
                pass
        
        # Fetch injury status from Sleeper API and map to current players
        injury_status_map = get_sleeper_injury_status_map()
        if 'player_id' in current_week.columns:
            def _lookup_injury_status(pid):
                if pd.isna(pid):
                    return 'Unknown'
                return injury_status_map.get(str(pid).upper(), 'Unknown')
            current_week['injury_status'] = current_week['player_id'].apply(_lookup_injury_status)
        else:
            current_week['injury_status'] = 'Unknown'
        
        if 'report_status' in current_week.columns:
            current_week['injury_status'] = np.where(
                current_week['injury_status'].isin([None, 'Unknown']),
                current_week['report_status'],
                current_week['injury_status']
            )
        
        # Get full training data (all historical weeks) from session
        if 'training_data' in request.session:
            training_data = pd.read_json(request.session['training_data'], orient='split')
            # Combine training data (historical) with current week data
            full_df = pd.concat([training_data, current_week], ignore_index=True)
        else:
            # If no training data, just use current week
            full_df = current_week
        
        # Export full dataframe (all weeks + current week) before predictions to CSV
        base_dir = settings.BASE_DIR if hasattr(settings, 'BASE_DIR') else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, 'df.csv')
        full_df.to_csv(csv_path, index=False)
        
        # Check if we have any players to predict
        if len(current_week) == 0:
            # Provide detailed error message with filter information
            error_msg = 'No players available for prediction after filtering. All players were filtered out.'
            error_details = []
            
            if 'training_data' in request.session:
                training_data = pd.read_json(request.session['training_data'], orient='split')
                error_details.append(f"Training data has {len(training_data)} records")
            
            from .config import EWMA_WEEKS, MODEL_PARAMS
            error_details.append(f"EWMA_WEEKS filter: {EWMA_WEEKS} minimum records")
            if MODEL_PARAMS.get("min_usage_filter", True):
                error_details.append(f"min_touches_ewma filter: {MODEL_PARAMS.get('min_touches_ewma', 3.0)} minimum touches")
            
            error_msg += f" Filters applied: {', '.join(error_details)}"
            error_msg += " Consider: 1) Loading data for the week first, 2) Lowering min_touches_ewma in config, or 3) Disabling min_usage_filter"
            
            return JsonResponse({
                'success': False,
                'message': error_msg
            }, status=400)
        
        # Make predictions (returns tuple: predictions_df, shap_values)
        df_predicted, shap_values = predict_week(season, week, current_week, FEATURES, NUMERIC_FEATURES)
        
        # Round probability for display
        df_predicted['probability'] = df_predicted['probability'].round(4)
        
        # Prepare data for frontend
        display_cols = ['player_name', 'team', 'position', 'against', 'probability', 
                       'played', 'touchdown', 'injury_status', 'season', 'week', 'player_id']
        df_display = df_predicted[[col for col in display_cols if col in df_predicted.columns]].copy()
        
        # Convert to dict first, then replace NaN with None for JSON compatibility
        records = df_display.to_dict('records')
        
        # Add SHAP values to each record
        for record in records:
            player_id = str(record.get('player_id', ''))
            if shap_values and player_id in shap_values:
                record['feature_explanations'] = shap_values[player_id]
            else:
                record['feature_explanations'] = None
        
        # Clean NaN values - replace with None (which becomes null in JSON)
        for record in records:
            for key, value in record.items():
                # Check if value is NaN (works for all types)
                try:
                    if pd.isna(value):
                        record[key] = None
                except (TypeError, ValueError):
                    # If pd.isna() fails, value is not NaN, so keep it
                    pass
        
        # Store predictions in session (convert NaN to None for JSON compatibility)
        # First convert to dict and clean NaN, then back to DataFrame for JSON storage
        df_predicted_dict = df_predicted.to_dict('records')
        for record in df_predicted_dict:
            for key, value in record.items():
                try:
                    if pd.isna(value):
                        record[key] = None
                except (TypeError, ValueError):
                    pass
        df_predicted_clean = pd.DataFrame(df_predicted_dict)
        request.session['predictions'] = df_predicted_clean.to_json(orient='split')
        
        return JsonResponse({
            'success': True,
            'message': f'Predictions generated for {len(df_predicted)} players.',
            'data': records,
            'columns': list(df_display.columns)
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Error making predictions: {str(e)}',
            'traceback': traceback.format_exc()
        }, status=500)


@json_response_view
@require_http_methods(["POST"])
def get_feature_data(request):
    """Get feature data (X table) for current week predictions"""
    try:
        if not request.body:
            return JsonResponse({
                'success': False,
                'message': 'Request body is empty.'
            }, status=400)
        
        data = json.loads(request.body.decode('utf-8'))
        season = int(data.get('season'))
        week = int(data.get('week'))
        
        # Check if current week data exists in session
        if 'current_week_data' not in request.session:
            return JsonResponse({
                'success': False,
                'message': 'Please load data first before viewing features.'
            }, status=400)
        
        # Get current week data from session
        current_week = pd.read_json(request.session['current_week_data'], orient='split')
        
        # Apply same filters as predict_week_view
        from .config import EWMA_WEEKS, FEATURES
        
        # Filter players with fewer than EWMA_WEEKS records
        # NOTE: Same logic as predict_week_view - for weeks 1-3, count all games; for weeks 4+, count only current season
        if 'training_data' in request.session:
            training_data = pd.read_json(request.session['training_data'], orient='split')
            if week <= 3:
                # Early weeks (1-3): Count all games across all seasons (historical + current)
                all_data = pd.concat([training_data[["player_id"]], current_week[["player_id"]]], ignore_index=True)
                player_record_counts = all_data.groupby("player_id").size()
            else:
                # Weeks 4+: Count only current season games
                if 'season' in training_data.columns:
                    current_season_data = training_data[training_data["season"] == season][["player_id"]].copy()
                else:
                    # If season column doesn't exist, assume all training_data is from current season
                    current_season_data = training_data[["player_id"]].copy()
                all_current_season_data = pd.concat([current_season_data, current_week[["player_id"]]], ignore_index=True)
                player_record_counts = all_current_season_data.groupby("player_id").size()
            players_with_enough_history = player_record_counts[player_record_counts >= EWMA_WEEKS].index
            current_week = current_week[current_week["player_id"].isin(players_with_enough_history)].copy()
        
        # Filter out players who haven't played
        if 'played' in current_week.columns:
            current_week = current_week[current_week['played'] == 1].copy()
        
        # Additional filter: Remove players with no meaningful stats
        if len(current_week) > 0:
            key_usage_features = ['targets_ewma', 'receptions_ewma', 'carries_ewma', 'touches_ewma']
            has_stats = pd.Series(False, index=current_week.index)
            
            for feature in key_usage_features:
                if feature in current_week.columns:
                    has_feature = current_week[feature].notna() & (current_week[feature] > 0)
                    has_stats = has_stats | has_feature
            
            if 'red_zone_touches_ewma' in current_week.columns:
                rz_stats = current_week['red_zone_touches_ewma'].notna() & (current_week['red_zone_touches_ewma'] > 0)
                has_stats = has_stats | rz_stats
            
            # Note: designed_rush_attempts_ewma has been removed from features
            
            players_with_stats = has_stats.sum()
            min_players_needed = min(10, max(1, len(current_week) // 2))
            
            if players_with_stats >= min_players_needed:
                current_week = current_week[has_stats].copy()
        
        # Fetch injury status to display in feature table
        injury_status_map = get_sleeper_injury_status_map()
        if 'player_id' in current_week.columns:
            def _lookup_injury_status(pid):
                if pd.isna(pid):
                    return 'Unknown'
                return injury_status_map.get(str(pid).upper(), 'Unknown')
            current_week['injury_status'] = current_week['player_id'].apply(_lookup_injury_status)
        else:
            current_week['injury_status'] = 'Unknown'
        
        if 'report_status' in current_week.columns:
            current_week['injury_status'] = np.where(
                current_week['injury_status'].isin([None, 'Unknown']),
                current_week['report_status'],
                current_week['injury_status']
            )
        
        # Select only feature columns + player identifying info
        # This matches exactly what goes into the model (X matrix)
        display_cols = ['player_id', 'player_name', 'team', 'position', 'against', 'injury_status', 'season', 'week']
        
        # Get features that exist in current_week (same as what predict_week does)
        available_features = [f for f in FEATURES if f in current_week.columns]
        missing_features = [f for f in FEATURES if f not in current_week.columns]
        
        # Combine display columns and feature columns
        # Order: display_cols first, then features (same order as FEATURES)
        all_cols = display_cols + available_features
        
        # Filter to only columns that exist
        available_cols = [col for col in all_cols if col in current_week.columns]
        feature_df = current_week[available_cols].copy()
        
        # Sort by player_name for easier viewing
        if 'player_name' in feature_df.columns:
            feature_df = feature_df.sort_values('player_name').reset_index(drop=True)
        
        # Convert to records for JSON serialization
        records = feature_df.to_dict('records')
        
        # Clean NaN values
        for record in records:
            for key, value in record.items():
                try:
                    if pd.isna(value):
                        record[key] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        record[key] = float(value) if pd.notna(value) else None
                    elif isinstance(value, np.bool_):
                        record[key] = bool(value)
                except (TypeError, ValueError):
                    pass
        
        return JsonResponse({
            'success': True,
            'data': records,
            'columns': available_cols,
            'feature_columns': available_features,  # Use available_features instead of feature_cols
            'display_columns': display_cols,
            'missing_features': missing_features,
            'season': season,
            'week': week,
            'row_count': len(records),
            'feature_count': len(available_features),
            'total_features_requested': len(FEATURES)
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Error loading feature data: {str(e)}',
            'traceback': traceback.format_exc()
        }, status=500)


@json_response_view
@require_http_methods(["POST"])
def export_predictions(request):
    """Export predictions to CSV"""
    try:
        if 'predictions' not in request.session:
            return JsonResponse({
                'success': False,
                'message': 'No predictions available to export. Please run predictions first.'
            }, status=400)
        
        df_predicted = pd.read_json(request.session['predictions'], orient='split')
        season = request.session.get('season', '')
        week = request.session.get('week', '')
        
        # Create CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="nfl_touchdown_predictions_s{season}_w{week}.csv"'
        
        df_predicted.to_csv(response, index=False)
        return response
        
    except Exception as e:
        # If it's already a JSON response attempt, return JSON
        # Otherwise return error response
        import json
        data = json.loads(request.body) if hasattr(request, 'body') and request.body else {}
        return JsonResponse({
            'success': False,
            'message': f'Error exporting data: {str(e)}'
        }, status=500)


@json_response_view
@require_http_methods(["GET"])
def check_model_exists(request):
    """Check if a model exists for given season/week"""
    try:
        season = int(request.GET.get('season'))
        week = int(request.GET.get('week'))
        
        model = NFLTouchdownModel(season, week)
        exists = model.model_exists()
        
        return JsonResponse({
            'exists': exists
        })
        
    except (ValueError, TypeError) as e:
        return JsonResponse({
            'success': False,
            'message': f'Invalid season or week: {str(e)}'
        }, status=400)
