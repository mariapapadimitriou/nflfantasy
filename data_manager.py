"""
Data Manager - Handles data loading with intelligent caching
Stores historical data to avoid reloading
"""
import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple
from datetime import datetime
# Add this to the TOP of data_manager.py after the imports

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

from config import (
    HISTORICAL_DATA_FILE, CACHE_ENABLED, HISTORICAL_SEASONS,
    POSITIONS, REPORT_STATUS_ORDER, CACHE_DIR
)
from data_source import get_data_source
from utils import american_odds_to_probability, compute_red_zone_stats


class NFLDataManager:
    """Manages NFL data loading and caching"""
    
    def __init__(self, data_source_type: str = "nflreadpy"):
        self.data_source = get_data_source(data_source_type)
        self.historical_data = None
        self.rz_stats_rolling = None
        self.rz_stats_prev = None
    
    def load_historical_data(self) -> pd.DataFrame:
        """Load cached historical data if available"""
        if CACHE_ENABLED and os.path.exists(HISTORICAL_DATA_FILE):
            print(f"[Cache] Loading historical data from {HISTORICAL_DATA_FILE}")
            self.historical_data = pd.read_parquet(HISTORICAL_DATA_FILE)
            return self.historical_data
        return None
    
    def save_historical_data(self, df: pd.DataFrame):
        """Save historical data to cache"""
        if CACHE_ENABLED:
            print(f"[Cache] Saving historical data to {HISTORICAL_DATA_FILE}")
            df.to_parquet(HISTORICAL_DATA_FILE, index=False)
            self.historical_data = df
    
    def get_seasons_to_load(self, season: int, week: int) -> list:
        """Determine which seasons need to be loaded"""
        if week == 1:
            # For week 1, don't include current season in historical
            return list(range(season - HISTORICAL_SEASONS - 1, season))
        return list(range(season - HISTORICAL_SEASONS, season + 1))
    
    def load_and_process_data(self, season: int, week: int, force_reload: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load and process data for given season and week
        
        Args:
            season: NFL season year
            week: Week number (1-20)
            force_reload: Force reload from source instead of using cache
        
        Returns:
            Dictionary with 'df' (historical) and 'current_week' dataframes
        """
        print(f"\n{'='*60}")
        print(f"Loading data for Season {season}, Week {week}")
        print(f"{'='*60}\n")
        
        # Load historical data from cache if available
        if not force_reload:
            cached_data = self.load_historical_data()
            if cached_data is not None:
                max_cached_season = cached_data['season'].max()
                max_cached_week = cached_data[cached_data['season'] == max_cached_season]['week'].max()
                
                # Check if we need new data
                if max_cached_season > season or (max_cached_season == season and max_cached_week >= week - 1):
                    print(f"[Cache] Using cached data up to Season {max_cached_season}, Week {max_cached_week}")
                    # Filter to appropriate cutoff
                    ref_week = week - 1 if week > 1 else 20
                    historical = cached_data[
                        (cached_data['season'] < season) | 
                        ((cached_data['season'] == season) & (cached_data['week'] <= ref_week))
                    ].copy()
                    
                    # Load only current week
                    current_week = self._load_current_week(season, week)
                    return {
                        'df': historical.reset_index(drop=True),
                        'current_week': current_week.reset_index(drop=True)
                    }
        
        # Load fresh data
        print("[Loading] Fetching fresh data from source...")
        return self._load_fresh_data(season, week)
    
    def _load_current_week(self, season: int, week: int) -> pd.DataFrame:
        """Load only the current week's data"""
        print(f"[Loading] Current week data for Season {season}, Week {week}")
        
        try:
            # Load minimal data for current week
            seasons = [season]
            player_stats = self.data_source.load_player_stats(seasons)
            print(f"[Debug] Loaded {len(player_stats)} player stat records")
            
            pbp = self.data_source.load_pbp(seasons)
            print(f"[Debug] Loaded {len(pbp)} play-by-play records")
            
            roster = self.data_source.load_rosters(seasons)
            print(f"[Debug] Loaded {len(roster)} roster records")
            
            schedules = self.data_source.load_schedules(seasons)
            print(f"[Debug] Loaded {len(schedules)} schedule records")
            
            # Filter to current week
            schedules = schedules[
                (schedules['season'] == season) & 
                (schedules['week'] == week)
            ]
            print(f"[Debug] Filtered to {len(schedules)} games for week {week}")
            
            if len(schedules) == 0:
                raise ValueError(f"No schedule data found for Season {season}, Week {week}. Week may not have occurred yet or data not available.")
            
            # Load red zone stats if available
            rz_file = os.path.join(CACHE_DIR, 'red_zone_rolling.parquet')
            prev_rz_file = os.path.join(CACHE_DIR, 'red_zone_prev.parquet')
            
            if os.path.exists(rz_file) and os.path.exists(prev_rz_file):
                self.rz_stats_rolling = pd.read_parquet(rz_file)
                self.rz_stats_prev = pd.read_parquet(prev_rz_file)
                print(f"[Debug] Loaded cached red zone stats")
            else:
                print(f"[Debug] Computing red zone stats from scratch...")
                self.rz_stats_rolling, self.rz_stats_prev = compute_red_zone_stats(pbp)
                self.rz_stats_rolling.to_parquet(rz_file, index=False)
                self.rz_stats_prev.to_parquet(prev_rz_file, index=False)
            
            # Process current week
            print(f"[Debug] Processing current week data...")
            current_week = self._process_data(
                player_stats, pbp, roster, schedules, 
                season, week, current_week_only=True
            )
            
            print(f"[Debug] Current week processed: {len(current_week)} players")
            return current_week
            
        except Exception as e:
            print(f"[ERROR] Failed to load current week: {str(e)}")
            raise
    
    def _load_fresh_data(self, season: int, week: int) -> Dict[str, pd.DataFrame]:
        """Load fresh data from source and process"""
        seasons = self.get_seasons_to_load(season, week)
        print(f"[Loading] Seasons: {seasons}")
        
        # Load all data
        player_stats = self.data_source.load_player_stats(seasons)
        pbp = self.data_source.load_pbp(seasons)
        roster = self.data_source.load_rosters(seasons)
        schedules = self.data_source.load_schedules(seasons)
        team_names = self.data_source.load_teams()
        
        # Compute red zone stats (expensive operation)
        print("[Processing] Computing red zone statistics...")
        rz_file = os.path.join(CACHE_DIR, 'red_zone_rolling.parquet')
        prev_rz_file = os.path.join(CACHE_DIR, 'red_zone_prev.parquet')
        
        if os.path.exists(rz_file) and os.path.exists(prev_rz_file):
            self.rz_stats_rolling = pd.read_parquet(rz_file)
            self.rz_stats_prev = pd.read_parquet(prev_rz_file)
        else:
            self.rz_stats_rolling, self.rz_stats_prev = compute_red_zone_stats(pbp)
            self.rz_stats_rolling.to_parquet(rz_file, index=False)
            self.rz_stats_prev.to_parquet(prev_rz_file, index=False)
        
        # Filter schedules
        schedules = schedules[
            schedules['season'].isin(seasons) &
            (
                (schedules['season'] < season) |
                ((schedules['season'] == season) & (schedules['week'] <= week))
            )
        ]
        
        print(f"[Processing] Filtered schedules: {len(schedules)} games")
        
        # Process data
        df = self._process_data(player_stats, pbp, roster, schedules, season, week)
        
        print(f"[Processing] Processed data: {len(df)} total records")
        
        # Split into historical and current week
        current_week = df[(df['season'] == season) & (df['week'] == week)].copy()
        print(f"[Processing] Current week: {len(current_week)} records")
        
        # Create historical dataset
        ref_week = week - 1 if week > 1 else 20
        
        # FIXED: More lenient historical data filtering
        if week == 1:
            # For week 1, use all previous season data
            historical = df[df['season'] < season].copy()
        else:
            # For other weeks, include previous weeks of current season
            historical = df[
                (df['season'] < season) | 
                ((df['season'] == season) & (df['week'] < week))
            ].copy()
        
        print(f"[Processing] Historical before filtering: {len(historical)} records")
        
        # Filter to only games that were played
        historical = historical[historical['played'] == 1].copy()
        print(f"[Processing] Historical after 'played' filter: {len(historical)} records")
        
        # Keep enough historical data (at least 3 seasons)
        min_season = season - HISTORICAL_SEASONS
        historical = historical[historical['season'] >= min_season].copy()
        print(f"[Processing] Historical after season filter (>={min_season}): {len(historical)} records")
        
        # Save the full processed data for future use
        self.save_historical_data(df)
        
        return {
            'df': historical.reset_index(drop=True),
            'current_week': current_week.reset_index(drop=True)
        }
        
    def _process_data(self, player_stats, pbp, roster, schedules, season, week, current_week_only=False):
        """Process raw data into features"""
        from utils import (
            calculate_rolling_avg, calculate_prev, 
            get_qb_rolling_stats
        )
        
        print("[Processing] Building feature dataframe...")
        print(f"[Debug] Input data sizes:")
        print(f"  - player_stats: {len(player_stats)}")
        print(f"  - pbp: {len(pbp)}")
        print(f"  - roster: {len(roster)}")
        print(f"  - schedules: {len(schedules)}")
        
        # Build weekly stats
        weekly_stats = player_stats[[
            "player_id", "season", "week",
            "rushing_yards", "rushing_tds",
            "receiving_yards", "receiving_tds",
            "passing_tds", "passing_yards"
        ]].copy()
        print(f"[Debug] weekly_stats: {len(weekly_stats)} records")
        
        # Build roster summary
        roster_summary = roster[roster.position.isin(POSITIONS)][[
            "gsis_id", "position", "season", "team", "rookie_year", "full_name", "status"
        ]].drop_duplicates().reset_index(drop=True)
        print(f"[Debug] roster_summary after position filter: {len(roster_summary)} records")
        
        roster_summary["rookie"] = np.where(
            roster_summary.rookie_year == roster_summary.season, 1, 0
        )
        roster_summary = roster_summary[roster_summary.status == 'ACT']
        print(f"[Debug] roster_summary after ACT filter: {len(roster_summary)} records")
        
        # Build games
        games = schedules[[
            "season", "week", "home_moneyline", "game_id", 
            "home_team", "away_team", "home_qb_id", "away_qb_id"
        ]].copy()
        print(f"[Debug] games: {len(games)} records")
        
        games["home_wp"] = games["home_moneyline"].apply(
            lambda x: american_odds_to_probability(x) if pd.notnull(x) else 0.5
        )

        print(games)
        
        # Merge home and away games
        print(f"[Debug] Merging home games...")
        home_games = pd.merge(
            roster_summary, games, how='left', 
            left_on=["season", "team"], 
            right_on=["season", "home_team"]
        ).dropna()

        print(home_games)
        home_games["home"] = 1
        home_games["qb_id"] = home_games["home_qb_id"]
        print(f"[Debug] home_games: {len(home_games)} records")
        
        print(f"[Debug] Merging away games...")
        away_games = pd.merge(
            roster_summary, games, how='left',
            left_on=["season", "team"], 
            right_on=["season", "away_team"]
        ).dropna()
        away_games["home"] = 0
        away_games["qb_id"] = away_games["away_qb_id"]
        print(f"[Debug] away_games: {len(away_games)} records")
        
        games_merged = pd.concat([away_games, home_games])
        print(f"[Debug] games_merged: {len(games_merged)} records")
        
        if len(games_merged) == 0:
            print("[ERROR] No games after merging roster with schedule!")
            print(f"[Debug] Roster teams: {roster_summary['team'].unique()[:20]}")
            print(f"[Debug] Schedule home teams: {games['home_team'].unique()[:20]}")
            print(f"[Debug] Schedule away teams: {games['away_team'].unique()[:20]}")
            raise ValueError("No games found after merging roster with schedule. Team name mismatch?")
        
        # Build main dataframe
        df = games_merged[[
            "gsis_id", "position", "full_name", "game_id", "team", 
            "season", "week", "rookie", "home_team", "away_team", 
            "home_wp", "home", "qb_id"
        ]]
        df = df.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
        print(f"[Debug] df after initial build: {len(df)} records")
        
        # Merge touchdowns
        touchdowns = pbp[pbp["touchdown"] == 1][["td_player_id", "game_id"]].drop_duplicates()
        print(f"[Debug] touchdowns: {len(touchdowns)} records")
        
        mdf = pd.merge(
            df, touchdowns, how='left', 
            left_on=['player_id', 'game_id'], 
            right_on=['td_player_id', 'game_id'], 
            indicator=True
        )
        mdf['touchdown'] = mdf['_merge'].apply(lambda x: 1 if x == 'both' else 0)
        df = mdf.drop(["_merge", "td_player_id"], axis=1)
        print(f"[Debug] df after touchdown merge: {len(df)} records")
        
        # Set matchup and home/away
        df["against"] = np.where(df["team"] == df["home_team"], df["away_team"], df["home_team"])
        df["home"] = np.where(df["team"] == df["home_team"], 1, 0)
        df["wp"] = np.where(df["team"] == df["home_team"], df["home_wp"], 1 - df["home_wp"])
        df = df.drop(["home_team", "away_team", "home_wp"], axis=1)
        
        # Add injury status placeholder
        df["report_status"] = 'Healthy'
        
        # Calculate defensive yards allowed
        print(f"[Debug] Calculating defensive yards...")
        yards_defense = pbp[["defteam", "game_id", "yards_gained", 'season', 'week']].groupby(
            ["game_id", "defteam"]
        ).agg({"yards_gained": "sum", 'season': 'first', 'week': 'first'}).reset_index()
        yards_defense = yards_defense.drop(['season', "week"], axis=1)
        yards_defense.sort_values(by=["defteam", "game_id"], inplace=True)
        yards_defense['rolling_yapg'] = (
            yards_defense.groupby('defteam')['yards_gained']
                .rolling(window=3, min_periods=1)
                .mean()
                .shift(1)
                .reset_index(level=0, drop=True)
        )
        
        df = pd.merge(
            df, yards_defense, how='left', 
            left_on=['game_id', 'against'], 
            right_on=['game_id', 'defteam']
        )
        print(f"[Debug] df after yards_defense merge: {len(df)} records")
        
        # Merge weekly stats
        df = pd.merge(df, weekly_stats, on=['player_id', 'season', 'week'], how='left')
        print(f"[Debug] df after weekly_stats merge: {len(df)} records")
        
        # Calculate rolling averages
        print(f"[Debug] Calculating rolling averages...")
        df = df.sort_values(by=['player_id', 'season', 'week'])
        df = df.groupby(['player_id'], group_keys=False).apply(calculate_rolling_avg)
        print(f"[Debug] df after rolling avg: {len(df)} records")
        
        df = df.drop([
            "rushing_yards", "rushing_tds", "receiving_yards", 
            "receiving_tds", "passing_tds", "passing_yards"
        ], axis=1, errors='ignore')
        
        # Calculate QB rolling stats
        print(f"[Debug] Calculating QB rolling stats...")
        qbs = df["qb_id"].unique()
        qb_weekly_stats = weekly_stats[weekly_stats['player_id'].isin(qbs)][[
            "player_id", "season", "week", "passing_tds", "passing_yards"
        ]]
        df = pd.merge(
            df, qb_weekly_stats, 
            left_on=['qb_id', 'season', 'week'], 
            right_on=['player_id', 'season', 'week'], 
            how='left', suffixes=('', '_drop')
        )
        df = df.sort_values(by=['qb_id', 'season', 'week'])
        df = df.groupby(['qb_id'], group_keys=False).apply(get_qb_rolling_stats)
        print(f"[Debug] df after QB stats: {len(df)} records")
        
        # Merge previous season stats
        print(f"[Debug] Calculating previous season stats...")
        season_stats = player_stats.groupby(['player_id', 'season']).agg({
            'rushing_yards': 'sum',
            'rushing_tds': 'sum',
            'receiving_yards': 'sum',
            'receiving_tds': 'sum'
        }).reset_index()
        
        df = pd.merge(df, season_stats, how='left', on=["player_id", "season"])
        df = df.sort_values(by=['player_id', 'season'])
        df = df.groupby(['player_id'], group_keys=False).apply(calculate_prev)
        print(f"[Debug] df after prev stats: {len(df)} records")
        
        df = df.drop([
            "rushing_yards", "rushing_tds", 
            "receiving_yards", "receiving_tds"
        ], axis=1, errors='ignore')
        
        # Clean data
        df = df.drop_duplicates()
        print(f"[Debug] df after drop_duplicates: {len(df)} records")
        
        df.loc[
            ~df['report_status'].isin(REPORT_STATUS_ORDER), 
            'report_status'
        ] = 'Minor'
        df = df[~(df.report_status == 'Out')]
        print(f"[Debug] df after removing 'Out' status: {len(df)} records")
        
        # Merge red zone stats
        print(f"[Debug] Merging red zone stats...")
        if self.rz_stats_rolling is not None and self.rz_stats_prev is not None:
            red_zone_df = self.rz_stats_rolling[['posteam', 'season', 'week', 'rolling_red_zone']]
            prev_year_rz = self.rz_stats_prev[['posteam', 'next_year', 'prev_red_zone']]
            
            df = pd.merge(
                df, red_zone_df, how='left', 
                left_on=['against', 'season', 'week'], 
                right_on=['posteam', 'season', 'week']
            )
            df = df.drop(['posteam'], axis=1, errors='ignore')
            
            df = pd.merge(
                df, prev_year_rz, how='left', 
                left_on=['against', 'season'], 
                right_on=['posteam', 'next_year']
            )
            df = df.drop(['posteam', 'next_year'], axis=1, errors='ignore')
        else:
            print("[Warning] Red zone stats not available, skipping...")
            df['rolling_red_zone'] = 0
            df['prev_red_zone'] = 0
        
        print(f"[Debug] df after red zone merge: {len(df)} records")
        
        # Zero out QB stats for QB position
        df.loc[df['position'] == 'QB', ['qb_rolling_passing_yards', 'qb_rolling_passing_tds']] = 0
        
        # Add played indicator
        print(f"[Debug] Adding played indicator...")
        played = weekly_stats[["player_id", "season", "week"]].drop_duplicates()
        played["played"] = 1
        df = pd.merge(df, played, how='left', on=["player_id", "season", "week"])
        df['played'] = df['played'].fillna(0).astype(int)
        print(f"[Debug] df after played merge: {len(df)} records")
        print(f"[Debug] Played distribution: {df['played'].value_counts().to_dict()}")
        
        print(f"[Debug] FINAL df: {len(df)} records")
        return df