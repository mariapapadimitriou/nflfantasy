import pandas as pd
import numpy as np

def match_teams(team, abbr_list):
    # Try direct match
    if team in abbr_list.values:
        return team
    # Try lower-case match
    for abbr in abbr_list:
        if team.lower() == abbr.lower():
            return abbr
    # Try partial match
    for abbr in abbr_list:
        if team.lower() in abbr.lower():
            return abbr
    return team

def calculate_rolling_avg(df):
    # Rolling stats for receiving/rushing touchdowns/yards
    df['rolling_receiving_touchdowns'] = df['receiving_tds'].rolling(window=3, min_periods=1).mean().shift(1)
    df['rolling_receiving_yards'] = df['receiving_yards'].rolling(window=3, min_periods=1).mean().shift(1)
    df['rolling_rushing_touchdowns'] = df['rushing_tds'].rolling(window=3, min_periods=1).mean().shift(1)
    df['rolling_rushing_yards'] = df['rushing_yards'].rolling(window=3, min_periods=1).mean().shift(1)
    df['rolling_yapg'] = df['yards_gained'].rolling(window=3, min_periods=1).mean().shift(1)
    return df

def calculate_prev(df):
    # Previous season's stats
    df['prev_receiving_yards'] = df['receiving_yards'].shift(1)
    df['prev_receiving_touchdowns'] = df['receiving_tds'].shift(1)
    df['prev_rushing_yards'] = df['rushing_yards'].shift(1)
    df['prev_rushing_touchdowns'] = df['rushing_tds'].shift(1)
    return df

def get_rolling_yards(df):
    # Defensive rolling feature
    df['rolling_yapg'] = df['yards_gained'].rolling(window=3, min_periods=1).mean().shift(1)

    return df

def get_qb_rolling_stats(df):
    df['qb_rolling_passing_yards'] = df['passing_yards'].rolling(window=3, min_periods=1).mean().shift(1)
    df['qb_rolling_passing_tds'] = df['passing_tds'].rolling(window=3, min_periods=1).mean().shift(1)
    return df

def american_odds_to_probability(odds):
    odds = int(odds)
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def compute_red_zone_stats(pbp: pd.DataFrame) -> tuple:
    pbp['is_red_zone'] = pbp['yardline_100'] <= 20
    red_zone_drives = pbp[pbp['is_red_zone']].groupby(['game_id', 'drive', 'posteam']).size().reset_index().drop(0, axis=1)

    def drive_scored(row):
        mask = (
            (pbp['game_id'] == row['game_id']) &
            (pbp['drive'] == row['drive']) &
            (pbp['posteam'] == row['posteam'])
        )
        return (pbp[mask]['touchdown'] == 1).any()

    red_zone_drives['red_zone_scored_td'] = red_zone_drives.apply(drive_scored, axis=1)

    drive_info = pbp.groupby(['game_id', 'drive'])[['season', 'week']].first().reset_index()
    red_zone_drives = red_zone_drives.merge(drive_info, on=['game_id', 'drive'], how='left')

    rz_team_week = red_zone_drives.groupby(['posteam', 'season', 'week'])['red_zone_scored_td'].agg(['sum', 'count']).reset_index()
    rz_team_week['red_zone_scoring_pct'] = rz_team_week['sum'] / rz_team_week['count']

    rz_team_week = rz_team_week.sort_values(['posteam', 'season', 'week'])
    rz_team_week['rolling_red_zone'] = (
        rz_team_week.groupby('posteam')['red_zone_scoring_pct']
        .rolling(window=3, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
    )

    prev_year = (
        rz_team_week.groupby(['posteam', 'season'])
        .agg(prev_year_td=('sum', 'sum'), prev_year_drives=('count', 'sum'))
        .reset_index()
    )
    prev_year['prev_red_zone'] = prev_year['prev_year_td'] / prev_year['prev_year_drives']
    prev_year['next_year'] = prev_year['season'] + 1

    return rz_team_week, prev_year