import dash
from dash import dash_table, dcc, html
import pandas as pd
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import os
import urllib.request
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, log_loss
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from itertools import combinations
import random
import joblib
import datetime

### HELPER FUNCTIONS

def american_odds_to_probability(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)
    
### LOAD IN DATA

years = [2022, 2023, 2024]

player_stats_load = nfl.import_seasonal_data(years)
pbp_load = nfl.import_pbp_data(years)
roster_load = nfl.import_seasonal_rosters(years)
injuries_load = nfl.import_injuries(years)
game_data_load = nfl.import_schedules(years)
weekly_stats_load = nfl.import_weekly_data([2021] + years, columns=["player_id", "season", "week", "rushing_yards", "rushing_tds", "receiving_yards", "receiving_tds"])
season_stats_load = nfl.import_seasonal_data([2021] + years)
played = nfl.import_weekly_data(years)

season_stats = season_stats_load.sort_values(by=['player_id', 'season'])[["player_id", "season", "rushing_yards", "rushing_tds", "receiving_yards", "receiving_tds"]]

#### INPUTS 
season = 2024
week = 7
date = datetime.date(2024, 10, 17).strftime("%Y-%m-%d")
features = ['home', 'prev_receiving_touchdowns', 'prev_receiving_yards', 'prev_rushing_touchdowns', 'prev_rushing_yards', 'report_status', 'rolling_receiving_touchdowns', 'rolling_receiving_yards', 'rolling_rushing_touchdowns', 'rolling_rushing_yards', 'wp', "against"]
numeric_features = ['prev_receiving_touchdowns', 'prev_receiving_yards', "prev_rushing_touchdowns", 'prev_rushing_yards', "rolling_receiving_touchdowns", "rolling_receiving_yards", "rolling_rushing_touchdowns",  "rolling_rushing_yards", "wp"]
####

roll_time = 4



rolling_start_week = week - roll_time

future_games = game_data_load[game_data_load.gameday >= date]

future_games["home_wp"] = future_games["home_moneyline"].apply(american_odds_to_probability)

future_games = future_games[(future_games.season == season) & (future_games.week == week)]

future_games = future_games[["game_id", "game_type", "season", "week", "home_team", "away_team", "home_wp"]]

current_rosters = roster_load[roster_load.season == season]

current_rosters = current_rosters[["season", "team", "position", "player_id", "status", "player_name", "rookie_year"]]

current_rosters = current_rosters[current_rosters.position.isin(['WR', 'QB', 'RB', 'TE'])]

home_games = pd.merge(current_rosters, future_games, how='left', left_on=["season", "team"], right_on=["season", "home_team"]).dropna()

home_games["home"] = 1

away_games = pd.merge(current_rosters, future_games, how='left', left_on=["season", "team"], right_on=["season", "away_team"]).dropna()

away_games["home"] = 0

games = pd.concat([away_games, home_games])

games["against"] = np.where(games["team"] == games["home_team"], games["away_team"], games["home_team"])

games = games[games.status == 'ACT']

weekly_stats = weekly_stats_load[(weekly_stats_load.season == season) & (weekly_stats_load.week >= rolling_start_week)]

averages = weekly_stats.groupby('player_id').agg({'rushing_yards': 'mean', 'receiving_yards': 'mean', 'rushing_tds': 'mean', 'receiving_tds': 'mean'}).reset_index()

averages.columns = ['player_id', 'rolling_rushing_yards', 'rolling_receiving_yards', 'rolling_rushing_touchdowns', 'rolling_receiving_touchdowns']

games = pd.merge(games, averages, how='left', on='player_id')

prev_stats = season_stats[season_stats.season == season - 1]

prev_stats = prev_stats[["player_id",	"rushing_yards", "rushing_tds", "receiving_yards", "receiving_tds"]]

prev_stats.columns = ['player_id', 'prev_rushing_yards', 'prev_receiving_yards', 'prev_rushing_touchdowns', 'prev_receiving_touchdowns']

games = pd.merge(games, prev_stats, how='left', on='player_id')

games["wp"] = np.where(games["team"] == games["home_team"], games["home_wp"], 1-games["home_wp"])

#games["weather_scale"] = 1

injuries = injuries_load[(injuries_load.season == season) & (injuries_load.week == week)]

injuries["report_status"] = np.where(injuries["report_status"].isna(), "Minor", injuries["report_status"])

games = pd.merge(games, injuries[["gsis_id", "report_status"]], how='left', left_on=["player_id"], right_on=["gsis_id"])

games["report_status"] = np.where(games["report_status"].isna(), 'Healthy', games["report_status"])

games.loc[games["rookie_year"] == season, ['prev_rushing_yards', 'prev_receiving_yards', 'prev_rushing_touchdowns', 'prev_receiving_touchdowns']] = 0

games = games[~(games.report_status == 'Out')]

games = games.drop('gsis_id', axis=1)

#new_data = games[games.player_id == '00-0032764']

new_data = games

X = new_data.drop(['week', 'season', 'rookie_year', "team", "position"], axis=1)[["player_id"] + features]

player_ids = X['player_id'].values

X = X.drop('player_id', axis=1)

# Load the fitted objects
imputer = joblib.load('/Model/imputer.pkl')
scaler = joblib.load('/Model/scaler.pkl')
encoder = joblib.load('/Model/encoder.pkl')

X[numeric_features] = imputer.transform(X[numeric_features])
X[numeric_features] = scaler.transform(X[numeric_features])

X_encoded = encoder.transform(X)

dnew = xgb.DMatrix(X_encoded)

best_model = xgb.Booster()
best_model.load_model('/Model/xgboost_model.json')

y_pred_new = best_model.predict(dnew)

# Map predictions to player_ids and create a DataFrame
predictions_df = pd.DataFrame({
    'player_id': player_ids,
    'probability': y_pred_new
})

# Sort the DataFrame by predicted touchdown probability (highest to lowest)
predictions_df_sorted = predictions_df.sort_values(by='probability', ascending=False)

output = pd.merge(predictions_df_sorted, games[["season", "week", "player_id", "position", "player_name", "team", "against", "report_status"]], how='inner', on='player_id')

output["probability"] = output["probability"]

#output = pd.merge(output, current_week[["player_id", "touchdown"]], how='left', on='player_id')

df = output

# Initialize Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Editable Table Example"),
    
    # Dash DataTable component with editable columns
    dash_table.DataTable(
        id='editable-table',
        columns=[{"name": i, "id": i, "editable": True} for i in df.columns],
        data=df.to_dict('records'),  # Populate table with data from DataFrame
        editable=True  # Enable editing
    ),
    
    # A div to display results from the table (optional)
    html.Div(id='output')
])

# Run the Dash app
if __name__ == '__main__':
    #app.run_server(debug=True)

    print('Here')
