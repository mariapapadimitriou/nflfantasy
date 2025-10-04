import pandas as pd
import numpy as np
import os
from utils import (
    match_teams,
    calculate_rolling_avg,
    calculate_prev,
    get_rolling_yards,
    get_qb_rolling_stats,
    american_odds_to_probability,
    compute_red_zone_stats
)
import nflreadpy as nfl

class NFLDataSource:
    def load_player_stats(self, seasons):
        print(f"[INFO] Loading player stats for seasons: {seasons}")

        return nfl.load_player_stats(seasons=seasons).to_pandas()
    def load_pbp(self, seasons):
        print(f"[INFO] Loading play-by-play for seasons: {seasons}")

        return nfl.load_pbp(seasons=seasons).to_pandas()
    def load_rosters(self, seasons):
        print(f"[INFO] Loading rosters for seasons: {seasons}")

        return nfl.load_rosters(seasons=seasons).to_pandas()
    def load_schedules(self, seasons):
        print(f"[INFO] Loading schedules for seasons: {seasons}")

        return nfl.load_schedules(seasons=seasons).to_pandas()
    def load_teams(self):
        print(f"[INFO] Loading teams master list")
        return nfl.load_teams().to_pandas()

def load_or_fetch(csv_filename, fetch_function, load, *args, **kwargs):
    if not load and os.path.exists(csv_filename):
        print(f"[STEP] Loading {csv_filename} from CSV")
        df = pd.read_csv(csv_filename)
    else:
        print(f"[STEP] Fetching {csv_filename} from API")
        df = fetch_function(*args, **kwargs)
        df.to_csv(csv_filename, index=False)
    return df

def load_data(season, week, load=True):
    print(f"[STEP] Starting data load for season {season}, week {week}, load={load}")
    data_source = NFLDataSource()
    years = list(range(season - 3, season + 1))
    if week == 1:
        years = list(range(season - 4, season))

    # Use load_or_fetch for each initial dataframe
    player_stats = load_or_fetch('player_stats.csv', data_source.load_player_stats, load, years)
    pbp = load_or_fetch('pbp.csv', data_source.load_pbp, load, years)
    roster = load_or_fetch('roster.csv', data_source.load_rosters, load, years)
    team_names = load_or_fetch('teams.csv', data_source.load_teams, load)
    schedules = load_or_fetch('games.csv', data_source.load_schedules, load, years)

    # Export after load for inspection
    player_stats.to_csv('step_player_stats.csv', index=False)
    pbp.to_csv('step_pbp.csv', index=False)
    roster.to_csv('step_roster.csv', index=False)
    team_names.to_csv('step_team_names.csv', index=False)
    schedules.to_csv('step_schedules.csv', index=False)

    print(f"[STEP] Computing red zone scoring % from play-by-play")
    #rz_stats_rolling, rz_stats_prev = compute_red_zone_stats(pbp)
    # rz_stats_rolling.to_csv('red_zone_rolling.csv', index=False)
    # rz_stats_prev.to_csv('red_zone_prev.csv', index=False)

    rz_stats_rolling = pd.read_csv('red_zone_rolling.csv')
    rz_stats_prev = pd.read_csv('red_zone_prev.csv')

    schedules = schedules[
        schedules['season'].isin(years) &
        (
            (schedules['season'] < season) |
            ((schedules['season'] == season) & (schedules['week'] <= week))
        )
    ]
    schedules.to_csv('step_schedules_filtered.csv', index=False)

    print(f"[STEP] Building weekly_stats dataframe")
    weekly_stats = player_stats[[
        "player_id", "season", "week",
        "rushing_yards", "rushing_tds",
        "receiving_yards", "receiving_tds",
        "passing_tds", "passing_yards"
    ]].copy()
    weekly_stats.to_csv('step_weekly_stats.csv', index=False)
    season_stats = nfl.load_player_stats(seasons=years, summary_level='reg+post').to_pandas()
    season_stats.to_csv('step_season_stats.csv', index=False)
    played = weekly_stats.copy()
    played.to_csv('step_played.csv', index=False)

    print(f"[STEP] Building roster summary dataframe")
    roster_summary = roster[roster.position.isin(['WR', 'QB', 'RB', 'TE'])][[
        "gsis_id", "position", "season", "team", "rookie_year", "full_name", "status"
    ]].drop_duplicates().reset_index(drop=True)
    roster_summary["rookie"] = np.where(roster_summary.rookie_year == roster_summary.season, 1, 0)
    roster_summary = roster_summary[roster_summary.status == 'ACT']
    roster_summary.to_csv('step_roster_summary.csv', index=False)

    print(f"[STEP] Building games dataframe")
    games = schedules[["season", "week", "home_moneyline", "game_id", "home_team", "away_team", "home_qb_id", "away_qb_id"]].copy()
    games["home_wp"] = games["home_moneyline"].apply(
        lambda x: american_odds_to_probability(x) if pd.notnull(x) else 0.5
    )
    games.to_csv('step_games.csv', index=False)

    print(f"[STEP] Merging home and away games with roster")
    home_games = pd.merge(roster_summary, games, how='left', left_on=["season", "team"], right_on=["season", "home_team"]).dropna()
    home_games["home"] = 1
    home_games["qb_id"] = home_games["home_qb_id"]

    away_games = pd.merge(roster_summary, games, how='left', left_on=["season", "team"], right_on=["season", "away_team"]).dropna()
    away_games["home"] = 0
    away_games["qb_id"] = away_games["away_qb_id"]
    games_merged = pd.concat([away_games, home_games])
    games_merged.to_csv('step_games_merged.csv', index=False)

    df = games_merged[[
        "gsis_id", "position", "full_name", "game_id", "team", "season", "week", "rookie",
        "home_team", "away_team", "home_wp", "home", "qb_id"
    ]]
    df = df.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
    df.to_csv('step_df_initial.csv', index=False)

    print(f"[STEP] Merging touchdowns")
    touchdowns = pbp[pbp["touchdown"] == 1][["td_player_id", "game_id"]].drop_duplicates().reset_index(drop=True)
    mdf = pd.merge(df, touchdowns, how='left', left_on=['player_id', 'game_id'], right_on=['td_player_id', 'game_id'], indicator=True)
    mdf['touchdown'] = mdf['_merge'].apply(lambda x: 1 if x == 'both' else 0)
    df = mdf.drop(["_merge", "td_player_id"], axis=1)
    df.to_csv('step_df_touchdowns.csv', index=False)

    print(f"[STEP] Setting matchup and home/away indicators")
    df["against"] = np.where(df["team"] == df["home_team"], df["away_team"], df["home_team"])
    df["home"] = np.where(df["team"] == df["home_team"], 1, 0)
    df["wp"] = np.where(df["team"] == df["home_team"], df["home_wp"], 1 - df["home_wp"])
    df = df.drop(["home_team", "away_team", "home_wp"], axis=1)
    df.to_csv('step_df_matchup.csv', index=False)

    print(f"[STEP] Feature engineering placeholder for injuries")
    df["report_status"] = 'Healthy'
    df.to_csv('step_df_injuries.csv', index=False)

    print(f"[STEP] Calculating and merging defensive yards allowed stats")
    yards_defense = pbp[["defteam", "game_id", "yards_gained", 'season', 'week']].groupby(
        ["game_id", "defteam"]).agg({"yards_gained": "sum", 'season': 'first', 'week': 'first'}).reset_index()
    yards_defense = yards_defense.drop(['season', "week"], axis=1)
    
    yards_defense.sort_values(by=["defteam", "game_id"], inplace=True)
    yards_defense['rolling_yapg'] = (
        yards_defense.groupby('defteam')['yards_gained']
            .rolling(window=3, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
    )
    yards_defense.to_csv('step_defense_yardes_allowed.csv', index=False)

    df = pd.merge(df, yards_defense, how='left', left_on=['game_id', 'against'], right_on=['game_id', 'defteam'])
    df.to_csv('step_df_rolling_def.csv', index=False)

    print(f"[STEP] Merging weekly_stats")
    df = pd.merge(df, weekly_stats, on=['player_id', 'season', 'week'], how='left')
    df.to_csv('step_df_weekly_stats.csv', index=False)

    print(f"[STEP] Calculating rolling averages for players")
    df = df.sort_values(by=['player_id', 'season', 'week'])
    df = df.groupby(['player_id']).apply(calculate_rolling_avg)
    df = df.drop(["rushing_yards", "rushing_tds", "receiving_yards", "receiving_tds", "passing_tds", "passing_yards"], axis=1)
    df.to_csv('step_df_rolling_avg.csv', index=False)

    print(f"[STEP] Calculating QB rolling stats")
    qbs = df["qb_id"].unique()
    qb_weekly_stats = weekly_stats[weekly_stats['player_id'].isin(qbs)][["player_id", "season", "week", "passing_tds", "passing_yards"]]
    df = pd.merge(df, qb_weekly_stats, left_on=['qb_id', 'season', 'week'], right_on=['player_id', 'season', 'week'], how='left', suffixes=('', '_drop'))
    df = df.sort_values(by=['qb_id', 'season', 'week'])
    df = df.groupby(['qb_id']).apply(get_qb_rolling_stats)
    df.to_csv('step_df_qb_rolling.csv', index=False)


    print(f"[STEP] Merging previous season stats")
    season_stats_slice = season_stats[["player_id", "season", "rushing_yards", "rushing_tds", "receiving_yards", "receiving_tds"]]
    season_stats_slice.to_csv('step_season_stats_slice.csv', index=False)

    df = pd.merge(df, season_stats_slice, how='left', on=["player_id", "season"])
    df = df.sort_values(by=['player_id', 'season'])
    df = df.groupby(['player_id']).apply(calculate_prev)
    df = df.drop(["rushing_yards", "rushing_tds", "receiving_yards", "receiving_tds"], axis=1)
    df.to_csv('step_df_prev_season.csv', index=False)

    print(f"[STEP] Dropping duplicates and cleaning report_status")
    df = df.drop_duplicates()
    df.loc[~df['report_status'].isin(["Healthy", "Minor", 'Questionable', 'Doubtful', 'Out']), 'report_status'] = 'Minor'
    df = df[~(df.report_status == 'Out')]
    df.to_csv('step_df_cleaned.csv', index=False)

    print(f"[STEP] Merging red zone scoring % features")
    red_zone_df = rz_stats_rolling[['posteam', 'season', 'week', 'rolling_red_zone']]
    prev_year_rz = rz_stats_prev[['posteam', 'next_year', 'prev_red_zone']]
    df = pd.merge(df, red_zone_df, how='left', left_on=['against', 'season', 'week'], right_on=['posteam', 'season', 'week'])
    df = df.drop(['posteam'], axis=1)
    df = pd.merge(df, prev_year_rz, how='left', left_on=['against', 'season'], right_on=['posteam', 'next_year'])
    df = df.drop(['posteam', 'next_year'], axis=1)
    df.loc[df['position'] == 'QB', ['qb_rolling_passing_yards', 'qb_rolling_passing_tds']] = 0
    df.to_csv('step_df_redzone.csv', index=False)

    print(f"[STEP] Adding played indicator")
    played = weekly_stats[["player_id", "season", "week"]].drop_duplicates()
    played["played"] = 1
    df = pd.merge(df, played, how='left', on=["player_id", "season", "week"])
    df.to_csv('step_df_played.csv', index=False)

    print(f"[STEP] Filtering data for training and prediction splits")
    current_week = df[(df.season == season) & (df.week == week)]
    if week == 1:
        ref_week = 20
    else:
        ref_week = week - 1
    df = df[(df['season'] < season) | ((df['season'] == season) & (df['week'] <= ref_week))]
    df = df[df.played == 1]
    df = df[df.season > (season - 3)]

    current_week.to_csv('step_current_week.csv', index=False)
    df.to_csv('step_final_df.csv', index=False)

    print(f"[COMPLETE] Returning processed dataframes: df shape {df.shape}, current_week shape {current_week.shape}")
    return {
        "df": df.reset_index(drop=True),
        "current_week": current_week.reset_index(drop=True)
    }