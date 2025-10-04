import dash
from dash import dcc, html, dash_table, Input, Output, State
import os
import pandas as pd

from data import load_data
from model import train_model, predict_week, retrain_model
from utils import american_odds_to_probability, match_teams

features = ['home','prev_receiving_touchdowns', 'prev_receiving_yards', 
            'prev_rushing_touchdowns', 'prev_rushing_yards', 'report_status', 
            'rolling_receiving_touchdowns', 'rolling_receiving_yards', 'rolling_rushing_touchdowns',
            'rolling_rushing_yards', 'rookie', 'wp', 'rolling_red_zone', 'rolling_yapg', 
            "prev_red_zone", "qb_rolling_passing_tds", "qb_rolling_passing_yards"]

numeric_features = ['prev_receiving_touchdowns', 'prev_receiving_yards', 
                    "prev_rushing_touchdowns", 'prev_rushing_yards', 
                    "rolling_receiving_touchdowns", "rolling_receiving_yards", "rolling_rushing_touchdowns", 
                    "rolling_rushing_yards", "wp", "rolling_red_zone", "rolling_yapg", 
                    "prev_red_zone", "qb_rolling_passing_tds", "qb_rolling_passing_yards"]

app = dash.Dash(__name__, title="NFL Touchdown Predictions", 
                external_stylesheets=['https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap'])
app.layout = html.Div([
    html.H1("NFL Touchdown Predictions", style={'fontFamily': 'Roboto', 'color': '#43dae5', 'textAlign': 'center'}),
    html.Div([
        dcc.Input(id='input-season', type='number', placeholder='Enter season (e.g. 2023)', min=2020, max=2050,
                  style={'backgroundColor': '#223436', 'color': '#43dae5', 'border': '1px solid #43dae5', 
                         'fontFamily': 'Roboto', 'padding': '10px', 'marginRight': '10px'}),
        dcc.Input(id='input-week', type='number', placeholder='Enter week (1-20)', min=1, max=20,
                  style={'backgroundColor': '#223436', 'color': '#43dae5', 'border': '1px solid #43dae5', 
                         'fontFamily': 'Roboto', 'padding': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
    html.Div([
        html.Button('Load Data', id='load-data-btn', style={'backgroundColor': '#43dae5', 'color': '#121212', 
                                                            'fontFamily': 'Roboto', 'border': 'none', 'padding': '10px', 
                                                            'marginRight': '10px'}),
        html.Button('Predict Week', id='predict-week-btn', style={'backgroundColor': '#43dae5', 'color': '#121212', 
                                                                 'fontFamily': 'Roboto', 'border': 'none', 'padding': '10px', 
                                                                 'marginRight': '10px'}),
        html.Button('Retrain Model', id='retrain-model-btn', style={'backgroundColor': 'transparent', 
                                                                   'border': '2px solid #43dae5', 'color': '#43dae5', 
                                                                   'fontFamily': 'Roboto', 'padding': '10px', 
                                                                   'marginRight': '10px'}),
        html.Button('Export Data', id='export-btn', style={'backgroundColor': '#43dae5', 'color': '#121212', 
                                                          'fontFamily': 'Roboto', 'border': 'none', 'padding': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
    dcc.ConfirmDialog(
        id='confirm-retrain-dialog',
        message='A model has already been trained on data previous to this season and week. Are you sure you want to overwrite it?',
    ),
    html.Div(id='output-message', style={'color': '#43dae5', 'fontFamily': 'Roboto', 'textAlign': 'center', 'marginBottom': '20px'}),
    dcc.Loading(
        id='loading', type='default', children=[
            dash_table.DataTable(
                id='output-table', columns=[], data=[],
                style_header={'backgroundColor': '#223436', 'color': '#43dae5', 'fontWeight': 'bold', 'border': '1px solid #43dae5'},
                style_data={'backgroundColor': '#121212', 'color': '#e0e0e0', 'border': '1px solid #43dae5'},
                style_cell={'fontFamily': 'Roboto', 'fontSize': '14px', 'padding': '10px', 'textAlign': 'left', 'border': '1px solid #43dae5'},
                style_data_conditional=[
                    {'if': {'state': 'active'}, 'backgroundColor': '#223436', 'border': '1px solid #43dae5'},
                    {'if': {'state': 'selected'}, 'backgroundColor': '#223436', 'border': '1px solid #43dae5'}
                ],
                style_table={'overflowX': 'auto'},
            )
        ]
    ),
    dcc.Download(id="download-dataframe-csv"),
    # Hidden stores for persistency:
    dcc.Store(id='stored-df', storage_type='memory'),
    dcc.Store(id='stored-current-week', storage_type='memory'),
    dcc.Store(id='stored-predicted', storage_type='memory'),
], style={'backgroundColor': '#121212', 'height': '100vh', 'padding': '20px'})

@app.callback(
    [Output('output-message', 'children'),
     Output('output-table', 'columns'),
     Output('output-table', 'data'),
     Output('confirm-retrain-dialog', 'displayed'),
     Output("download-dataframe-csv", "data"),
     Output('stored-df', 'data'),
     Output('stored-current-week', 'data'),
     Output('stored-predicted', 'data')],
    [Input('load-data-btn', 'n_clicks'),
     Input('predict-week-btn', 'n_clicks'),
     Input('retrain-model-btn', 'n_clicks'),
     Input('confirm-retrain-dialog', 'submit_n_clicks'),
     Input('export-btn', 'n_clicks')],
    [State('input-season', 'value'),
     State('input-week', 'value'),
     State('stored-df', 'data'),
     State('stored-current-week', 'data'),
     State('stored-predicted', 'data')]
)
def handle_buttons(load_clicks, predict_clicks, retrain_clicks, confirm_retrain_clicks, export_clicks,
                   season, week, stored_df, stored_current_week, stored_predicted):
    import json
    ctx = dash.callback_context
    export_data = None
    message = ""
    columns, data = [], []
    show_dialog = False

    if not ctx.triggered:
        return "", [], [], False, None, None, None, None
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # General validation for season and week inputs
    if not season or not week:
        return "Please enter both season and week to proceed.", [], [], False, None, stored_df, stored_current_week, stored_predicted

    # Load Data
    if button_id == 'load-data-btn':
        # You may want to check for CSV and load from there, but here's the basic pattern:
        loaded = load_data(season, week)
        df = loaded.get('df') if 'df' in loaded else None
        current_week = loaded.get('current_week') if 'current_week' in loaded else None
        if df is None or current_week is None:
            return "Error: Data not loaded correctly.", [], [], False, None, None, None, None
        # Convert DataFrames to JSON for storage
        return f"Data for season {season}, week {week} has been loaded.", [], [], False, None, df.to_json(date_format='iso', orient='split'), current_week.to_json(date_format='iso', orient='split'), None

    # Predict Week
    elif button_id == 'predict-week-btn':
        if stored_current_week:
            current_week = pd.read_json(stored_current_week, orient='split')
            df_predicted = predict_week(season, week, current_week, features, numeric_features)
            columns = [{'name': i, 'id': i} for i in df_predicted.columns]
            return "", columns, df_predicted.to_dict('records'), False, None, stored_df, stored_current_week, df_predicted.to_json(date_format='iso', orient='split')
        else:
            return "Data not loaded. Please load the data first before predicting.", [], [], False, None, stored_df, stored_current_week, stored_predicted

    # Retrain Model
    elif button_id == 'retrain-model-btn':
        model_path = f"Model/xgboost_model-{season}-{week}.json"
        if os.path.exists(model_path):
            return "", [], [], True, None, stored_df, stored_current_week, stored_predicted
        elif stored_df is not None:
            df = pd.read_json(stored_df, orient='split')
            success, message = retrain_model(season, week, df, features, numeric_features)
            return message, [], [], False, None, stored_df, stored_current_week, stored_predicted
        else:
            return "Data not loaded. Please load the data first before retraining the model.", [], [], False, None, stored_df, stored_current_week, stored_predicted

    # Confirm Retrain Dialog
    elif button_id == 'confirm-retrain-dialog' and confirm_retrain_clicks:
        if stored_df is not None:
            df = pd.read_json(stored_df, orient='split')
            success, message = retrain_model(season, week, df, features, numeric_features)
            return message, [], [], False, None, stored_df, stored_current_week, stored_predicted

    # Export Button
    elif button_id == 'export-btn' and stored_predicted is not None:
        df_predicted = pd.read_json(stored_predicted, orient='split')
        export_data = dcc.send_data_frame(df_predicted.to_csv, "predicted_week_data.csv")
        return "", [], [], False, export_data, stored_df, stored_current_week, stored_predicted

    # Default return for other buttons
    return "", columns, data, show_dialog, export_data, stored_df, stored_current_week, stored_predicted

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
    app.run_server(debug=True)