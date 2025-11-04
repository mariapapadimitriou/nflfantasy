"""
Django views for NFL Touchdown Predictions
"""
import json
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import ensure_csrf_cookie
from django.contrib import messages
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
