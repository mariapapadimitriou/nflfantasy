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
import traceback

from .config import FEATURES, NUMERIC_FEATURES
from .data_manager import NFLDataManager
from .ml_model import train_model, predict_week, retrain_model, NFLTouchdownModel


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
        
        # Filter out players who haven't played (no stats) - they shouldn't be predicted
        # Only filter if 'played' column exists
        if 'played' in current_week.columns:
            before_count = len(current_week)
            current_week = current_week[current_week['played'] == 1].copy()
            after_count = len(current_week)
            if before_count > after_count:
                print(f"[Filter] Removed {before_count - after_count} players with no stats (played=0) from predictions")
        
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
            
            # Also check for position-specific stats that might indicate activity
            # For WR/TE: end_zone_targets_ewma
            if 'end_zone_targets_ewma' in current_week.columns:
                wr_te_stats = current_week['end_zone_targets_ewma'].notna() & (current_week['end_zone_targets_ewma'] > 0)
                has_stats = has_stats | wr_te_stats
            
            # For RB/QB: designed_rush_attempts_ewma
            if 'designed_rush_attempts_ewma' in current_week.columns:
                qb_stats = current_week['designed_rush_attempts_ewma'].notna() & (current_week['designed_rush_attempts_ewma'] > 0)
                has_stats = has_stats | qb_stats
            
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
                
                if before_count > after_count:
                    print(f"[Filter] Removed {before_count - after_count} players with no meaningful stats (all key features NaN/0) from predictions")
                    print(f"[Filter] Remaining players: {after_count}")
            else:
                # If filter would remove too many players, keep them all but log a warning
                print(f"[Warning] Filter would leave only {players_with_stats} players (need at least {min_players_needed})")
                print(f"[Warning] Keeping all {before_count} players (may include players with no stats)")
                print(f"[Warning] This might indicate early season (EWMA features may be NaN) or data quality issue")
        
        # Get full training data (all historical weeks) from session
        if 'training_data' in request.session:
            training_data = pd.read_json(request.session['training_data'], orient='split')
            # Combine training data (historical) with current week data
            full_df = pd.concat([training_data, current_week], ignore_index=True)
            print(f"[Export] Combined dataframe: {len(training_data)} historical rows + {len(current_week)} current week rows = {len(full_df)} total rows")
        else:
            # If no training data, just use current week
            full_df = current_week
            print(f"[Export] No training data found, using only current week: {len(full_df)} rows")
        
        # Export full dataframe (all weeks + current week) before predictions to CSV
        base_dir = settings.BASE_DIR if hasattr(settings, 'BASE_DIR') else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, 'df.csv')
        full_df.to_csv(csv_path, index=False)
        print(f"[Export] Saved full dataframe to {csv_path} with {len(full_df)} rows and {len(full_df.columns)} columns")
        
        # Check if we have any players to predict
        if len(current_week) == 0:
            return JsonResponse({
                'success': False,
                'message': 'No players available for prediction after filtering. All players were filtered out (no stats or no meaningful features).'
            }, status=400)
        
        # Make predictions (returns tuple: predictions_df, shap_values)
        df_predicted, shap_values = predict_week(season, week, current_week, FEATURES, NUMERIC_FEATURES)
        
        # Round probability for display
        df_predicted['probability'] = df_predicted['probability'].round(4)
        
        # Prepare data for frontend
        display_cols = ['player_name', 'team', 'position', 'against', 'probability', 
                       'played', 'touchdown', 'report_status', 'season', 'week', 'player_id']
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
