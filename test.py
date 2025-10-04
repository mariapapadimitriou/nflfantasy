from data import load_data
from model import train_model, predict_week

def main():
    season = 2025
    week = 6
    load = False  # Toggle this: True to load from nflreadpy, False to load from csv

    result = load_data(season, week, load=load)
    df = result["df"]
    current_week = result["current_week"]

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

    train_model(df, season, week, features, numeric_features)
    pred = predict_week(season, week, current_week, features, numeric_features)

    pred.to_csv('predictions.csv')

if __name__ == "__main__":

    main()