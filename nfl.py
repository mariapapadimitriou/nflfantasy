import nfl_data_py as nfl
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.impute import SimpleImputer
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import joblib
import datetime
import requests
from fuzzywuzzy import process
import pandas as pd
from datetime import timedelta
import dash
from dash import dcc, html, dash_table, Input, Output, State
import os
import json
import pandas as pd


def american_odds_to_probability(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)
    
def map_weather_to_scale(condition):
    if pd.isna(condition) or 'Indoors' in condition or 'Controlled Climate' in condition or 'N/A' in condition:
        return 1
    elif any(word in condition.lower() for word in ['sunny', 'clear', 'fair', 'partly cloudy']):
        return 1
    elif any(word in condition.lower() for word in ['mostly cloudy', 'cloudy', 'breezy', 'mild', 'clouds']):
        return 2
    elif any(word in condition.lower() for word in ['overcast', 'cool', 'wind', 'hazy', 'drizzle']):
        return 3
    elif any(word in condition.lower() for word in ['rain', 'showers', 'snow', 'fog']):
        return 4
    elif any(word in condition.lower() for word in ['storm', 'cold', 'frigid', 'thunderstorm', 'miserable']):
        return 5

    return 1  # For anything unhandled

def match_teams(team, choices, scorer=process.extractOne):
    match = scorer(team, choices)
    return match[0] if match else None

def pull_data_for_season_date(date):
    url = "https://www.teamrankings.com/nfl/stat/opponent-red-zone-scoring-pct?date={date}".format(date=date.strftime('%Y-%m-%d'))

    try:
        tables = pd.read_html(url)
        df = tables[0]

        return df

    except Exception as e:
        print(f"Failed to pull data for {date}: {e}")
        return None
    
def calculate_rolling_avg(group):
    group['rolling_rushing_yards'] = group['rushing_yards'].rolling(window=4, min_periods=1).mean().shift(1)
    group['rolling_receiving_yards'] = group['receiving_yards'].rolling(window=4, min_periods=1).mean().shift(1)
    group['rolling_rushing_touchdowns'] = group['rushing_tds'].rolling(window=4, min_periods=1).mean().shift(1)
    group['rolling_receiving_touchdowns'] = group['receiving_tds'].rolling(window=4, min_periods=1).mean().shift(1)
    return group

def calculate_prev(group):
    group['prev_rushing_yards'] = group['rushing_yards'].shift(1)
    group['prev_receiving_yards'] = group['receiving_yards'].shift(1)
    group['prev_rushing_touchdowns'] = group['rushing_tds'].shift(1)
    group['prev_receiving_touchdowns'] = group['receiving_tds'].shift(1)

    return group

def get_rolling_yards(group):
    group['rolling_yapg'] = group['yards_gained'].shift(1).rolling(window=4, min_periods=1).mean()
    return group

def get_qb_rolling_stats(group):
    group['qb_rolling_passing_tds'] = group['passing_tds'].rolling(window=4, min_periods=1).mean().shift(1)
    group['qb_rolling_passing_yards'] = group['passing_yards'].rolling(window=4, min_periods=1).mean().shift(1)
    return group

def trainModel(df, season, week):
    
    features = ['home', 'prev_receiving_touchdowns', 'prev_receiving_yards', 'prev_rushing_touchdowns', 'prev_rushing_yards', 'report_status', 'rolling_receiving_touchdowns', 'rolling_receiving_yards', 'rolling_rushing_touchdowns', 'rolling_rushing_yards', 'rookie', 'wp', 'rolling_red_zone', 'rolling_yapg', "prev_red_zone", "qb_rolling_passing_tds", "qb_rolling_passing_yards"]

    X = df[["player_id"] + features]

    y = df['touchdown']
    
    player_ids = X['player_id'].values
    
    X = X.drop('player_id', axis=1)
    
    ordinal_features = ["report_status"]
    numeric_features = ['prev_receiving_touchdowns', 'prev_receiving_yards', "prev_rushing_touchdowns", 'prev_rushing_yards', "rolling_receiving_touchdowns", "rolling_receiving_yards", "rolling_rushing_touchdowns",  "rolling_rushing_yards", "wp", "rolling_red_zone", "rolling_yapg", "prev_red_zone", "qb_rolling_passing_tds", "qb_rolling_passing_yards"]
    #categorical_features = ["qb_id"]

    report_status_order = ["Healthy", "Minor", 'Questionable', 'Doubtful', 'Out']
    
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    X[numeric_features] = imputer.fit_transform(X[numeric_features])
    X[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    encoder = ColumnTransformer(
        transformers=[
            ('ordinal', OrdinalEncoder(categories=[report_status_order]), ordinal_features),
            #('onehot', OneHotEncoder(), categorical_features)
        ],
        remainder='passthrough'
    )
    
    X_encoded = encoder.fit_transform(X)
    
    X_train, X_test, y_train, y_test, player_ids_train, player_ids_test = train_test_split(X_encoded, y, player_ids, test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    threshold = 0.5
    
    def objective(trial):
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'eta': trial.suggest_float('eta', 0.01, 0.9),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'scale_pos_weight': len(y_train_resampled[y_train_resampled == 0]) / len(y_train_resampled[y_train_resampled == 1]),
            'lambda': trial.suggest_float('lambda', 1e-8, 10.0),
            'alpha': trial.suggest_float('alpha', 1e-8, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
        }
    
        dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled)
        dtest = xgb.DMatrix(X_test, label=y_test)
    
        model = xgb.train(param, dtrain, num_boost_round=100)
    
        y_pred_prob = model.predict(dtest)
    
        y_pred_binary = [1 if prob > threshold else 0 for prob in y_pred_prob]
    
        loss = log_loss(y_test, y_pred_prob)
        return loss
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, timeout=600)
    
    best_params = study.best_trial.params
    
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': len(y_train_resampled[y_train_resampled == 0]) / len(y_train_resampled[y_train_resampled == 1]),
    })
    
    best_model = xgb.train(best_params, dtrain, num_boost_round=100)
    
    best_model.save_model(f"Model/xgboost_model-{season}-{week}.json")
    y_pred_prob = best_model.predict(dtest)
    
    joblib.dump(imputer, f"Model/imputer-{season}-{week}.pkl")
    joblib.dump(scaler, f"Model/scaler-{season}-{week}.pkl")
    joblib.dump(encoder, f"Model/encoder-{season}-{week}.pkl")
    
    y_pred_binary = [1 if prob > threshold else 0 for prob in y_pred_prob]

    print(classification_report(y_test, y_pred_binary))
    
    return classification_report(y_test, y_pred_binary)

# Function to simulate loading data
def load_data(season, week):
    years = list(range(season-2, season+1))

    player_stats_load = nfl.import_seasonal_data(years)
    pbp_load = nfl.import_pbp_data([2021] + years)
    roster_load = nfl.import_seasonal_rosters(years)
    injuries_load = nfl.import_injuries(years)
    game_data_load = nfl.import_schedules(years)
    weekly_stats_load = nfl.import_weekly_data([2021] + years, columns=["player_id", "season", "week", "rushing_yards", "rushing_tds", "receiving_yards", "receiving_tds", "passing_tds", "passing_yards"])
    season_stats_load = nfl.import_seasonal_data([2021] + years)
    played_load = nfl.import_weekly_data(years)
    team_names = nfl.import_team_desc()

    season_start_dates = {
        2021: datetime.datetime(2021, 9, 9),  # Example start date: September 9, 2021
        2022: datetime.datetime(2022, 9, 8),  # September 8, 2022
        2023: datetime.datetime(2023, 9, 7),  # September 7, 2023
        2024: datetime.datetime(2024, 9, 5)   # September 5, 2024
    }

    weeks_in_season = 18

    all_data = []

    for year, start_date in season_start_dates.items():
        for w in range(weeks_in_season):

            game_date = start_date + timedelta(weeks=week)

            data = pull_data_for_season_date(game_date)

            data["week"] = w + 1

            data["season"] = year

            data = data[["Team", "Last 3", "season", "week"]]

            data = data.rename(columns={"Last 3": "rolling_red_zone"})

            try:
              data["rolling_red_zone"] = data["rolling_red_zone"].str.rstrip('%').astype(float) / 100
            except:
              data["rolling_red_zone"] = pd.NA

              data = data.dropna()

            if data is not None:
                all_data.append(data)

    if all_data:
        combined_df_load = pd.concat(all_data, ignore_index=True)

    red_zone = {
        2020: 'https://www.teamrankings.com/nfl/stat/opponent-red-zone-scoring-pct?date=2021-03-01',
        2021: 'https://www.teamrankings.com/nfl/stat/opponent-red-zone-scoring-pct?date=2022-03-01',
        2022: 'https://www.teamrankings.com/nfl/stat/opponent-red-zone-scoring-pct?date=2023-03-01',
        2023: 'https://www.teamrankings.com/nfl/stat/opponent-red-zone-scoring-pct?date=2024-03-01',
        }

    red_zone_data_load = pd.DataFrame()

    for y in red_zone.keys():

      url = red_zone[y]
      new_red = pd.read_html(url)[0]
      new_red['year'] = y
      new_red['next_year'] = y + 1
      r = new_red.rename(columns={str(y): 'prev_red_zone'})

      r["prev_red_zone"] = r["prev_red_zone"].str.rstrip('%').astype(float) / 100

      r = r[['Team', 'year', 'prev_red_zone', 'next_year']]

      red_zone_data_load = pd.concat([red_zone_data_load, r])
      
    player_stats = player_stats_load.copy()
    pbp = pbp_load.copy()
    roster = roster_load.copy()
    injuries = injuries_load.copy()
    game_data = game_data_load.copy()
    played = played_load.copy()
    weekly_stats = weekly_stats_load.copy()
    combined_df = combined_df_load.copy()
    red_zone_data = red_zone_data_load.copy()

    touchdowns = pbp[pbp["touchdown"] == 1][["td_player_id", "game_id"]].drop_duplicates().reset_index(drop=True)
    roster_summary = roster[roster.position.isin(['WR', 'QB', 'RB', 'TE'])][["player_id", "position", "season", "team", "rookie_year", "player_name", "status"]].drop_duplicates().reset_index(drop=True)
    roster_summary["rookie"] = np.where(roster_summary.rookie_year == roster_summary.season, 1, 0)
    injuries_summary = injuries[["season", "week", "gsis_id", "report_status"]]
    injuries_summary["report_status"] = np.where(injuries_summary["report_status"].isna(), "Minor", injuries_summary["report_status"])

    games = game_data[["season", "week", "home_moneyline", "game_id", "home_team", "away_team", "home_qb_id", "away_qb_id"]]
    games["home_wp"] = games["home_moneyline"].apply(american_odds_to_probability)

    home_games = pd.merge(roster_summary, games, how='left', left_on=["season", "team"], right_on=["season", "home_team"]).dropna()
    home_games["home"] = 1
    home_games["qb_id"] = home_games["home_qb_id"]
    away_games = pd.merge(roster_summary, games, how='left', left_on=["season", "team"], right_on=["season", "away_team"]).dropna()
    away_games["home"] = 0
    away_games["qb_id"] = away_games["away_qb_id"]

    games = pd.concat([away_games, home_games])

    games = games[games.status == 'ACT']

    df = games[["player_id", "player_name", "game_id", "team", "season", "week", "rookie", "home_team", "away_team", "home_wp", "home", "qb_id"]]

    mdf = pd.merge(df, touchdowns, how='left', left_on=['player_id', 'game_id'], right_on=['td_player_id', 'game_id'], indicator=True)

    mdf['touchdown'] = mdf['_merge'].apply(lambda x: 1 if x == 'both' else 0)

    df = mdf.drop(["_merge", "td_player_id"], axis=1)

    df["against"] = np.where(df["team"] == df["home_team"], df["away_team"], df["home_team"])

    df["home"] = np.where(df["team"] == df["home_team"], 1, 0)

    df["wp"] = np.where(df["team"] == df["home_team"], df["home_wp"], 1-df["home_wp"])

    df = df.drop(["home_team", "away_team", "home_wp"], axis=1)

    df = pd.merge(df, injuries_summary, how='left', left_on=["player_id", "season", "week"], right_on=["gsis_id", "season", "week"])
    df["report_status"] = np.where(df["report_status"].isna(), 'Healthy', df["report_status"])

    yards_defense = pbp[["defteam", "game_id", "yards_gained", 'season', 'week']].groupby(["game_id", "defteam"]).agg({"yards_gained": "sum", 'season': 'first', 'week': 'first'}).reset_index()

    yards_defense = yards_defense.drop(['season', "week"], axis=1)

    df = pd.merge(df, yards_defense, how='left', left_on=['game_id', 'against'], right_on=['game_id', 'defteam'])

    df = df.sort_values(by=["against", "season", "week"])

    df = df.groupby('against').apply(get_rolling_yards)

    df = df.drop(["gsis_id", "game_id", "defteam"], axis=1)

    df = pd.merge(df, weekly_stats, on=['player_id', 'season', 'week'], how='left')

    df = df.sort_values(by=['player_id', 'season', 'week'])

    df = df.groupby(['player_id']).apply(calculate_rolling_avg)

    df = df.drop(["rushing_yards", "rushing_tds",	"receiving_yards",	"receiving_tds",	"passing_tds", "passing_yards"], axis=1)

    qbs = df["qb_id"].unique()

    qb_weekly_stats = weekly_stats[weekly_stats['player_id'].isin(qbs)][["player_id", "season", "week", "passing_tds", "passing_yards"]]

    df = pd.merge(df, qb_weekly_stats, left_on=['qb_id', 'season', 'week'], right_on=['player_id', 'season', 'week'], how='left', suffixes=('', '_drop'))

    df = df.sort_values(by=['qb_id', 'season', 'week'])

    df = df.groupby(['qb_id']).apply(get_qb_rolling_stats)

    df = pd.merge(df, season_stats_load[["player_id", "season", "rushing_yards", "rushing_tds", "receiving_yards", "receiving_tds"]], how='left', on=["player_id", "season"])

    df = df.sort_values(by=['player_id', 'season'])

    df = df.groupby(['player_id']).apply(calculate_prev)

    df = df.drop(["rushing_yards", "rushing_tds",	"receiving_yards",	"receiving_tds"], axis=1)

    # test derrick henry #df[df.player_id == '00-0032764'].sort_values(by=['season', 'week'])

    df = df.drop_duplicates()

    df.loc[~df['report_status'].isin(["Healthy", "Minor", 'Questionable', 'Doubtful', 'Out']), 'report_status'] = 'Minor'

    df = df[~(df.report_status == 'Out')]

    red_zone_data['team_name'] = red_zone_data['Team'].apply(lambda x: match_teams(x, team_names['team_name']))

    red_zone_df = pd.merge(red_zone_data, team_names[["team_name", "team_abbr"]], left_on='team_name', right_on='team_name', how='left')

    red_zone_df = red_zone_df[["team_abbr", "year", "prev_red_zone", "next_year"]]

    df = pd.merge(df, red_zone_df,  how='left', left_on=['against', 'season'], right_on=['team_abbr', 'next_year'])

    df = df.drop(["year", "next_year", "team_abbr"], axis=1)

    combined_df['Team'] = combined_df['Team'].apply(lambda x: match_teams(x, team_names['team_name']))

    combined_df = pd.merge(combined_df, team_names[["team_name", "team_abbr"]], left_on='Team', right_on='team_name', how='left')

    df = pd.merge(df, combined_df,  how='left', left_on=['against', 'season', "week"], right_on=['team_abbr', 'season', "week"])

    df = df.drop(["Team", "team_name", "team_abbr"], axis=1)

    # Below is for runing the model in real time

    current_week = df[(df.season == season) & (df.week == week)]

    if week == 1:
      ref_week = 20
    else:
      ref_week = week - 1

    df = df[(df['season'] < season) | ((df['season'] == season) & (df['week'] <= ref_week))]

    # Below removes all the records where he player didn't play the game

    played = weekly_stats[["player_id", "season", "week"]].drop_duplicates()
    played["played"] = 1
    df = pd.merge(df, played, how='left', on=["player_id", "season", "week"])
    df = df[df.played == 1]
    
    current_week = pd.merge(current_week, played, how='left', on=["player_id", "season", "week"])

    date_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    df.to_csv(f'Historical/df_{season}_{week}_{date_str}.csv', index=False)
    current_week.to_csv(f'Historical/current_week_{season}_{week}_{date_str}.csv', index=False)
    
    return f"Data for season {season}, week {week} has been loaded.", df, current_week

def load_data_from_csv(season, week):
    date_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    df_path = f'Historical/df_{season}_{week}_{date_str}.csv'
    current_week_path = f'Historical/current_week_{season}_{week}_{date_str}.csv'
    
    if os.path.exists(df_path) and os.path.exists(current_week_path):
        df = pd.read_csv(df_path)
        current_week = pd.read_csv(current_week_path)
        return f"Data loaded from CSV for season {season}, week {week}.", df, current_week
    else:
        return None, None, None  # Indicates no data available


# Function to simulate prediction (can return a DataFrame)
def predict_week(season, week, current_week):
    
    features = ['home', 'prev_receiving_touchdowns', 'prev_receiving_yards', 'prev_rushing_touchdowns', 'prev_rushing_yards', 'report_status', 'rolling_receiving_touchdowns', 'rolling_receiving_yards', 'rolling_rushing_touchdowns', 'rolling_rushing_yards', 'rookie', 'wp', 'rolling_red_zone', 'rolling_yapg', "prev_red_zone", "qb_rolling_passing_tds", "qb_rolling_passing_yards"]
    
    new_data = current_week

    X = new_data[["player_id"] + features]

    player_ids = X['player_id'].values

    X = X.drop('player_id', axis=1)

    # Load the fitted objects
    imputer = joblib.load(f"Model/imputer-{season}-{week}.pkl")
    scaler = joblib.load(f"Model/scaler-{season}-{week}.pkl")
    encoder = joblib.load(f"Model/encoder-{season}-{week}.pkl")

    numeric_features = ['prev_receiving_touchdowns', 'prev_receiving_yards', "prev_rushing_touchdowns", 'prev_rushing_yards', "rolling_receiving_touchdowns", "rolling_receiving_yards", "rolling_rushing_touchdowns",  "rolling_rushing_yards", "wp", "rolling_red_zone", "rolling_yapg", "prev_red_zone", "qb_rolling_passing_tds", "qb_rolling_passing_yards"]
    
    X[numeric_features] = imputer.transform(X[numeric_features])
    X[numeric_features] = scaler.transform(X[numeric_features])

    X_encoded = encoder.transform(X)

    dnew = xgb.DMatrix(X_encoded)

    best_model = xgb.Booster()
    best_model.load_model(f"Model/xgboost_model-{season}-{week}.json")

    y_pred_new = best_model.predict(dnew)

    # Map predictions to player_ids and create a DataFrame
    predictions_df = pd.DataFrame({
        'player_id': player_ids,
        'probability': y_pred_new
    })

    # Sort the DataFrame by predicted touchdown probability (highest to lowest)
    predictions_df_sorted = predictions_df.sort_values(by='probability', ascending=False)

    output = pd.merge(predictions_df_sorted, new_data[["season", "week", "player_id", "player_name", "team", "against", "report_status", "played"]], how='inner', on='player_id')

    output = pd.merge(output, current_week[["player_id", "touchdown"]], how='left', on='player_id')

    output["touchdown"] = np.where(output["played"] == 1, output["touchdown"], np.nan)

    output["played"] = np.where(output["played"] == 1, "Y", "N")

    filtered_list_with_na = [x for x in output["touchdown"].values if pd.notna(x)]
    filtered_list_all_values = [output["probability"].values[i] for i in range(len(output["touchdown"].values)) if pd.notna(output["touchdown"].values[i])]

    final_df = pd.DataFrame({
        'touchdown': filtered_list_with_na,
        'probability': filtered_list_all_values
    })

    return output

# Function to simulate model retraining
def retrain_model(season, week, df):
    
    model_filename = f"model-{season}-{week}.json"
    if os.path.exists(model_filename):
        return False, f"Model for season {season}, week {week} has already been trained."
    else:
        # Simulate saving the model
        trainModel(df, season, week)
        return True, f"Model (using data prior to season {season}, week {week}) has been retrained and saved."


app = dash.Dash(__name__, title="NFL Touchdown Predictions", external_stylesheets=['https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap'])
app.layout = html.Div([
    html.H1("NFL Touchdown Predictions", style={'fontFamily': 'Roboto', 'color': '#43dae5', 'textAlign': 'center'}),

    # Input for season and week
    html.Div([
        dcc.Input(id='input-season', type='number', placeholder='Enter season (e.g. 2023)', min=2020, max=2050,
                  style={'backgroundColor': '#223436', 'color': '#43dae5', 'border': '1px solid #43dae5', 'fontFamily': 'Roboto', 'padding': '10px', 'marginRight': '10px'}),
        dcc.Input(id='input-week', type='number', placeholder='Enter week (1-20)', min=1, max=20,
                  style={'backgroundColor': '#223436', 'color': '#43dae5', 'border': '1px solid #43dae5', 'fontFamily': 'Roboto', 'padding': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),  # Center the inputs

    # Buttons
    html.Div([
        html.Button('Load Data', id='load-data-btn', style={'backgroundColor': '#43dae5', 'color': '#121212', 'fontFamily': 'Roboto', 'border': 'none', 'padding': '10px', 'marginRight': '10px'}),
        html.Button('Predict Week', id='predict-week-btn', style={'backgroundColor': '#43dae5', 'color': '#121212', 'fontFamily': 'Roboto', 'border': 'none', 'padding': '10px', 'marginRight': '10px'}),
        html.Button('Retrain Model', id='retrain-model-btn', 
                    style={'backgroundColor': 'transparent', 'border': '2px solid #43dae5', 'color': '#43dae5', 'fontFamily': 'Roboto', 'padding': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),

    # ConfirmDialog to ask the user for confirmation before retraining the model
    dcc.ConfirmDialog(
        id='confirm-retrain-dialog',
        message='A model has already been trained on data previous to this season and week. Are you sure you want to overwrite it?',
    ),

    # Message Output
    html.Div(id='output-message', style={'color': '#43dae5', 'fontFamily': 'Roboto', 'textAlign': 'center', 'marginBottom': '20px'}),

    # DataTable with consistent dark mode
    dcc.Loading(
        id='loading',
        type='default',
        children=[
             dash_table.DataTable(
                id='output-table',
                columns=[],  # Columns will be set dynamically
                data=[],     # Data will be set dynamically
                style_header={
                    'backgroundColor': '#223436',  # Dark header background
                    'color': '#43dae5',  # Light blue text
                    'fontWeight': 'bold',
                    'border': '1px solid #43dae5'
                },
                style_data={
                    'backgroundColor': '#121212',  # Dark background for cells
                    'color': '#e0e0e0',  # Light text color
                    'border': '1px solid #43dae5'
                },
                style_cell={
                    'fontFamily': 'Roboto',  # Use Roboto font
                    'fontSize': '14px',
                    'padding': '10px',  # Add some padding to cells
                    'textAlign': 'left',
                    'border': '1px solid #43dae5'  # Border for each cell
                },
                style_data_conditional=[
                    {
                        'if': {'state': 'active'},
                        'backgroundColor': '#223436',
                        'border': '1px solid #43dae5',
                    },
                    {
                        'if': {'state': 'selected'},
                        'backgroundColor': '#223436',
                        'border': '1px solid #43dae5',
                    }
                ],
                style_table={'overflowX': 'auto'},  # To handle wide tables
             )
        ]
    )
], style={'backgroundColor': '#121212', 'height': '100vh', 'padding': '20px'})  # Overall dark background

df, current_week = None, None

@app.callback(
    [Output('output-message', 'children'),
     Output('output-table', 'columns'),
     Output('output-table', 'data'),
     Output('confirm-retrain-dialog', 'displayed')],
    [Input('load-data-btn', 'n_clicks'),
     Input('predict-week-btn', 'n_clicks'),
     Input('retrain-model-btn', 'n_clicks'),
     Input('confirm-retrain-dialog', 'submit_n_clicks')],
    [State('input-season', 'value'),
     State('input-week', 'value')]
)
def handle_buttons(load_clicks, predict_clicks, retrain_clicks, confirm_retrain_clicks, season, week):
    global df, current_week
    ctx = dash.callback_context

    # Check if any button is clicked
    if not ctx.triggered:
        return "", [], [], False  # No button clicked yet

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # General validation for season and week inputs
    if not season or not week:
        return "Please enter both season and week to proceed.", [], [], False

    # Handle Load Data Button
    if button_id == 'load-data-btn':
        message, df, current_week = load_data_from_csv(season, week)
        if message is None:
            message, df, current_week = load_data(season, week)
        return message, [], [], False

    # Handle Predict Week Button
    elif button_id == 'predict-week-btn':
        if current_week is not None:
            df_predicted = predict_week(season, week, current_week)
            columns = [{'name': i, 'id': i} for i in df_predicted.columns]
            return "", columns, df_predicted.to_dict('records'), False
        else:
            return "Data not loaded. Please load the data first before predicting.", [], [], False

    # Handle Retrain Model Button
    elif button_id == 'retrain-model-btn':
        model_path = f"Model/xgboost_model-{season}-{week}.json"

        if os.path.exists(model_path):

            return "", [], [], True

        # No existing model found, retrain directly
        elif df is not None:
            success, message = retrain_model(season, week, df)
            return message, [], [], False
        else:
            return "Data not loaded. Please load the data first before retraining the model.", [], [], False

    elif button_id == 'confirm-retrain-dialog' and confirm_retrain_clicks:
        if df is not None:
            success, message = retrain_model(season, week, df)
            return message, [], [], False

    return "", [], [], False 

if __name__ == '__main__':
    app.run_server(debug=True)


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
