"""
NFL Touchdown Prediction Web Application
Clean, maintainable architecture with intelligent caching
"""
import dash
from dash import dcc, html, dash_table, Input, Output, State
import pandas as pd

from config import FEATURES, NUMERIC_FEATURES
from data_manager import NFLDataManager
from model import train_model, predict_week, retrain_model

# Initialize Dash app
app = dash.Dash(
    __name__, 
    title="NFL Touchdown Predictions",
    external_stylesheets=['https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap']
)

# App Layout
app.layout = html.Div([
    # Header
    html.H1(
        "NFL Touchdown Predictions", 
        style={
            'fontFamily': 'Roboto', 
            'color': '#43dae5', 
            'textAlign': 'center',
            'marginBottom': '20px'
        }
    ),
    
    # Input Section
    html.Div([
        dcc.Input(
            id='input-season', 
            type='number', 
            placeholder='Enter season (e.g. 2025)', 
            min=2020, 
            max=2050,
            style={
                'backgroundColor': '#223436', 
                'color': '#43dae5', 
                'border': '1px solid #43dae5', 
                'fontFamily': 'Roboto', 
                'padding': '10px', 
                'marginRight': '10px',
                'borderRadius': '5px'
            }
        ),
        dcc.Input(
            id='input-week', 
            type='number', 
            placeholder='Enter week (1-20)', 
            min=1, 
            max=20,
            style={
                'backgroundColor': '#223436', 
                'color': '#43dae5', 
                'border': '1px solid #43dae5', 
                'fontFamily': 'Roboto', 
                'padding': '10px',
                'borderRadius': '5px'
            }
        ),
        dcc.Checklist(
            id='force-reload-checkbox',
            options=[{'label': ' Force Reload Data', 'value': 'force'}],
            value=[],
            style={
                'color': '#43dae5',
                'fontFamily': 'Roboto',
                'marginLeft': '20px',
                'display': 'inline-block'
            }
        )
    ], style={
        'display': 'flex', 
        'justifyContent': 'center', 
        'alignItems': 'center',
        'marginBottom': '20px'
    }),
    
    # Button Section
    html.Div([
        html.Button(
            'Load Data', 
            id='load-data-btn', 
            style={
                'backgroundColor': '#43dae5', 
                'color': '#121212', 
                'fontFamily': 'Roboto', 
                'border': 'none', 
                'padding': '12px 24px', 
                'marginRight': '10px',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontWeight': '500'
            }
        ),
        html.Button(
            'Train Model', 
            id='train-model-btn', 
            style={
                'backgroundColor': '#43dae5', 
                'color': '#121212', 
                'fontFamily': 'Roboto', 
                'border': 'none', 
                'padding': '12px 24px', 
                'marginRight': '10px',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontWeight': '500'
            }
        ),
        html.Button(
            'Predict Week', 
            id='predict-week-btn', 
            style={
                'backgroundColor': '#43dae5', 
                'color': '#121212', 
                'fontFamily': 'Roboto', 
                'border': 'none', 
                'padding': '12px 24px', 
                'marginRight': '10px',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontWeight': '500'
            }
        ),
        html.Button(
            'Export CSV', 
            id='export-btn', 
            style={
                'backgroundColor': 'transparent', 
                'border': '2px solid #43dae5', 
                'color': '#43dae5', 
                'fontFamily': 'Roboto', 
                'padding': '12px 24px',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontWeight': '500'
            }
        ),
    ], style={
        'display': 'flex', 
        'justifyContent': 'center', 
        'marginBottom': '20px'
    }),
    
    # Confirmation Dialog
    dcc.ConfirmDialog(
        id='confirm-retrain-dialog',
        message='A model already exists for this season and week. Are you sure you want to overwrite it?',
    ),
    
    # Status Message
    html.Div(
        id='output-message', 
        style={
            'color': '#43dae5', 
            'fontFamily': 'Roboto', 
            'textAlign': 'center', 
            'marginBottom': '20px',
            'fontSize': '16px'
        }
    ),
    
    # Data Table with Loading
    dcc.Loading(
        id='loading', 
        type='default',
        children=[
            dash_table.DataTable(
                id='output-table',
                columns=[],
                data=[],
                style_header={
                    'backgroundColor': '#223436', 
                    'color': '#43dae5', 
                    'fontWeight': 'bold', 
                    'border': '1px solid #43dae5'
                },
                style_data={
                    'backgroundColor': '#121212', 
                    'color': '#e0e0e0', 
                    'border': '1px solid #43dae5'
                },
                style_cell={
                    'fontFamily': 'Roboto', 
                    'fontSize': '14px', 
                    'padding': '10px', 
                    'textAlign': 'left', 
                    'border': '1px solid #43dae5'
                },
                style_data_conditional=[
                    {
                        'if': {'state': 'active'}, 
                        'backgroundColor': '#223436', 
                        'border': '1px solid #43dae5'
                    },
                    {
                        'if': {'state': 'selected'}, 
                        'backgroundColor': '#223436', 
                        'border': '1px solid #43dae5'
                    }
                ],
                style_table={'overflowX': 'auto'},
                page_size=20,
                sort_action='native',
                filter_action='native'
            )
        ]
    ),
    
    # Download Component
    dcc.Download(id="download-dataframe-csv"),
    
    # Hidden Stores for Data Persistence
    dcc.Store(id='stored-df', storage_type='memory'),
    dcc.Store(id='stored-current-week', storage_type='memory'),
    dcc.Store(id='stored-predicted', storage_type='memory'),
    
], style={
    'backgroundColor': '#121212', 
    'minHeight': '100vh', 
    'padding': '20px'
})


@app.callback(
    [
        Output('output-message', 'children'),
        Output('output-table', 'columns'),
        Output('output-table', 'data'),
        Output('confirm-retrain-dialog', 'displayed'),
        Output('download-dataframe-csv', 'data'),
        Output('stored-df', 'data'),
        Output('stored-current-week', 'data'),
        Output('stored-predicted', 'data')
    ],
    [
        Input('load-data-btn', 'n_clicks'),
        Input('train-model-btn', 'n_clicks'),
        Input('predict-week-btn', 'n_clicks'),
        Input('confirm-retrain-dialog', 'submit_n_clicks'),
        Input('export-btn', 'n_clicks')
    ],
    [
        State('input-season', 'value'),
        State('input-week', 'value'),
        State('force-reload-checkbox', 'value'),
        State('stored-df', 'data'),
        State('stored-current-week', 'data'),
        State('stored-predicted', 'data')
    ]
)
def handle_callbacks(load_clicks, train_clicks, predict_clicks, confirm_clicks, export_clicks,
                    season, week, force_reload, stored_df, stored_current_week, stored_predicted):
    """Handle all button callbacks"""
    
    import json
    ctx = dash.callback_context
    
    # Default returns
    message = ""
    columns, data = [], []
    show_dialog = False
    export_data = None
    
    # No trigger
    if not ctx.triggered:
        return "", [], [], False, None, None, None, None
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Validate inputs
    if button_id != 'export-btn' and (not season or not week):
        return "⚠️ Please enter both season and week.", [], [], False, None, stored_df, stored_current_week, stored_predicted
    
    # Initialize data manager
    data_manager = NFLDataManager()
    
    # LOAD DATA
# In app.py, update the LOAD DATA section:

# LOAD DATA
    if button_id == 'load-data-btn':
        try:
            print(f"[App] Loading data for Season {season}, Week {week}")
            force = 'force' in force_reload
            
            # Check if week is valid for the season
            from datetime import datetime
            current_year = datetime.now().year
            current_month = datetime.now().month
            
            # NFL season typically starts in September (month 9)
            if season > current_year or (season == current_year and current_month < 9):
                return f"⚠️ Season {season} hasn't started yet.", [], [], False, None, None, None, None
            
            result = data_manager.load_and_process_data(season, week, force_reload=force)
            df = result['df']
            current_week = result['current_week']
            
            message = f"✅ Data loaded for Season {season}, Week {week}. Training data: {len(df)} records, Current week: {len(current_week)} players."
            
            return (
                message, [], [], False, None, 
                df.to_json(date_format='iso', orient='split'),
                current_week.to_json(date_format='iso', orient='split'),
                None
            )
        except ValueError as ve:
            return f"⚠️ {str(ve)}", [], [], False, None, None, None, None
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[ERROR] Full traceback:\n{error_details}")
            return f"❌ Error loading data: {str(e)}", [], [], False, None, None, None, None
    
    # TRAIN MODEL
    elif button_id == 'train-model-btn':
        if stored_df is None:
            return "⚠️ Please load data first before training the model.", [], [], False, None, stored_df, stored_current_week, stored_predicted
        
        try:
            df = pd.read_json(stored_df, orient='split')
            success, msg = retrain_model(season, week, df, FEATURES, NUMERIC_FEATURES)
            
            if not success:
                # Model exists, show confirmation dialog
                return "", [], [], True, None, stored_df, stored_current_week, stored_predicted
            
            return f"✅ {msg}", [], [], False, None, stored_df, stored_current_week, stored_predicted
            
        except Exception as e:
            return f"❌ Error training model: {str(e)}", [], [], False, None, stored_df, stored_current_week, stored_predicted
    
    # CONFIRM RETRAIN
    elif button_id == 'confirm-retrain-dialog' and confirm_clicks:
        if stored_df is None:
            return "⚠️ No data available.", [], [], False, None, stored_df, stored_current_week, stored_predicted
        
        try:
            df = pd.read_json(stored_df, orient='split')
            success, msg = train_model(df, season, week, FEATURES, NUMERIC_FEATURES)
            
            if success:
                return f"✅ {msg}", [], [], False, None, stored_df, stored_current_week, stored_predicted
            else:
                return f"❌ {msg}", [], [], False, None, stored_df, stored_current_week, stored_predicted
                
        except Exception as e:
            return f"❌ Error retraining model: {str(e)}", [], [], False, None, stored_df, stored_current_week, stored_predicted
    
    # PREDICT WEEK
    elif button_id == 'predict-week-btn':
        if stored_current_week is None:
            return "⚠️ Please load data first before making predictions.", [], [], False, None, stored_df, stored_current_week, stored_predicted
        
        try:
            current_week = pd.read_json(stored_current_week, orient='split')
            df_predicted = predict_week(season, week, current_week, FEATURES, NUMERIC_FEATURES)
            
            # Round probability for display
            df_predicted['probability'] = df_predicted['probability'].round(4)
            
            # Reorder columns for better display
            display_cols = ['player_name', 'team', 'position', 'against', 'probability', 
                          'played', 'touchdown', 'report_status', 'season', 'week']
            df_display = df_predicted[[col for col in display_cols if col in df_predicted.columns]]
            
            columns = [{'name': i, 'id': i} for i in df_display.columns]
            data = df_display.to_dict('records')
            
            message = f"✅ Predictions generated for {len(df_predicted)} players in Season {season}, Week {week}."
            
            return (
                message, columns, data, False, None,
                stored_df, stored_current_week,
                df_predicted.to_json(date_format='iso', orient='split')
            )
            
        except Exception as e:
            return f"❌ Error making predictions: {str(e)}", [], [], False, None, stored_df, stored_current_week, stored_predicted
    
    # EXPORT DATA
    elif button_id == 'export-btn':
        if stored_predicted is None:
            return "⚠️ No predictions available to export. Please run predictions first.", [], [], False, None, stored_df, stored_current_week, stored_predicted
        
        try:
            df_predicted = pd.read_json(stored_predicted, orient='split')
            export_data = dcc.send_data_frame(
                df_predicted.to_csv, 
                f"nfl_touchdown_predictions_s{season}_w{week}.csv",
                index=False
            )
            return "✅ Data exported successfully!", [], [], False, export_data, stored_df, stored_current_week, stored_predicted
            
        except Exception as e:
            return f"❌ Error exporting data: {str(e)}", [], [], False, None, stored_df, stored_current_week, stored_predicted
    
    return "", [], [], False, None, stored_df, stored_current_week, stored_predicted


# Custom HTML for favicon
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>NFL Touchdown Predictions</title>
    <link rel="icon" type="image/png" href="/assets/fball.svg">
    {%css%}
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)