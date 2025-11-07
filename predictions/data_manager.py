"""
Data Manager - Handles data loading with intelligent caching
Stores historical data to avoid reloading
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple
from datetime import datetime
from django.db.models import Q

# Add this to the TOP of data_manager.py after the imports

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

from .config import (
    CACHE_ENABLED,
    HISTORICAL_SEASONS,
    POSITIONS,
    REPORT_STATUS_ORDER,
)
from .data_source import get_data_source
from .utils import american_odds_to_probability
from .models import TrainingData, DataCache


class NFLDataManager:
    """Manages NFL data loading and caching"""

    def __init__(self, data_source_type: str = "nflreadpy"):
        self.data_source = get_data_source(data_source_type)
        self.historical_data = None

    def load_historical_data(self, season: int = None, week: int = None) -> pd.DataFrame:
        """Load cached historical data from database"""
        if not CACHE_ENABLED:
            return None
        
        try:
            # Load all historical data or filter by season/week
            query = TrainingData.objects.all()
            if season is not None:
                query = query.filter(season__lt=season)
            if week is not None and season is not None:
                # For week 1, exclude current season; otherwise exclude current week
                if week == 1:
                    query = query.filter(season__lt=season)
                else:
                    query = query.filter(
                        Q(season__lt=season) | 
                        Q(season=season, week__lt=week)
                    )
            
            records = list(query.values())
            if records:
                print(f"[Cache] Loading {len(records)} historical records from database")
                df = pd.DataFrame(records)
                # Reconstruct features from JSON field
                if 'features' in df.columns and len(df) > 0:
                    # Convert features JSON to DataFrame columns
                    features_list = df['features'].tolist()
                    if features_list and any(features_list):
                        features_df = pd.json_normalize(features_list)
                        df = pd.concat([df.drop('features', axis=1), features_df], axis=1)
                    else:
                        df = df.drop('features', axis=1)
                self.historical_data = df
                return df
        except Exception as e:
            print(f"[Cache] Error loading from database: {str(e)}")
        
        return None

    def save_historical_data(self, df: pd.DataFrame):
        """Save historical data to database"""
        if not CACHE_ENABLED:
            return
        
        try:
            print(f"[Cache] Saving historical data to database...")
            
            # Prepare data for bulk insert
            training_records = []
            for _, row in df.iterrows():
                # Extract features (all columns except the main ones)
                main_cols = ['season', 'week', 'player_id', 'player_name', 'team', 
                           'position', 'against', 'touchdown', 'played', 'report_status']
                features = {k: v for k, v in row.items() if k not in main_cols}
                
                training_records.append(
                    TrainingData(
                        season=int(row.get('season', 0)),
                        week=int(row.get('week', 0)),
                        player_id=str(row.get('player_id', '')),
                        player_name=str(row.get('player_name', '')),
                        team=str(row.get('team', '')),
                        position=str(row.get('position', '')),
                        against=str(row.get('against', '')),
                        touchdown=int(row.get('touchdown', 0)),
                        played=int(row.get('played', 0)),
                        report_status=str(row.get('report_status', 'Healthy')),
                        features=features,
                    )
                )
            
            # Bulk create/update (delete existing and insert new for efficiency)
            # For better performance, we could use bulk_update_or_create, but Django doesn't have it
            # So we'll delete existing records for this season/week range and insert new
            if training_records:
                # Delete existing records for the seasons/weeks we're about to insert
                seasons = set(r.season for r in training_records)
                TrainingData.objects.filter(season__in=seasons).delete()
                
                # Bulk insert
                TrainingData.objects.bulk_create(training_records, ignore_conflicts=True)
                print(f"[Cache] Saved {len(training_records)} records to database")
            
            self.historical_data = df
        except Exception as e:
            print(f"[Cache] Error saving to database: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_seasons_to_load(self, season: int, week: int) -> list:
        """Determine which seasons need to be loaded
        For training: only load season - 2 (so for 2025: 2023, 2024, 2025)
        """
        # Always include current season (for future weeks) and last 2 seasons
        # For 2025: load [2023, 2024, 2025]
        return list(range(season - HISTORICAL_SEASONS, season + 1))

    def load_and_process_data(
        self, season: int, week: int, force_reload: bool = False
    ) -> Dict[str, pd.DataFrame]:
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

        # Load historical data from database if available
        if not force_reload:
            cached_data = self.load_historical_data(season, week)
            if cached_data is not None and len(cached_data) > 0:
                max_cached_season = cached_data["season"].max()
                max_cached_week = cached_data[
                    cached_data["season"] == max_cached_season
                ]["week"].max()

                # Check if we need new data
                if max_cached_season > season or (
                    max_cached_season == season and max_cached_week >= week - 1
                ):
                    print(
                        f"[Cache] Using cached data up to Season {max_cached_season}, Week {max_cached_week}"
                    )
                    # Filter to appropriate cutoff
                    ref_week = week - 1 if week > 1 else 20
                    historical = cached_data[
                        (cached_data["season"] < season)
                        | (
                            (cached_data["season"] == season)
                            & (cached_data["week"] <= ref_week)
                        )
                    ].copy()

                    # Load only current week
                    current_week = self._load_current_week(season, week)
                    return {
                        "df": historical.reset_index(drop=True),
                        "current_week": current_week.reset_index(drop=True),
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
                (schedules["season"] == season) & (schedules["week"] == week)
            ]
            print(f"[Debug] Filtered to {len(schedules)} games for week {week}")

            if len(schedules) == 0:
                raise ValueError(
                    f"No schedule data found for Season {season}, Week {week}. Week may not have occurred yet or data not available."
                )


            # Process current week
            print(f"[Debug] Processing current week data...")
            current_week = self._process_data(
                player_stats,
                pbp,
                roster,
                schedules,
                season,
                week,
                current_week_only=True,
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
        # injuries = self.data_source.load_injuries()

        # print(injuries.head())


        # Filter schedules
        schedules = schedules[
            schedules["season"].isin(seasons)
            & (
                (schedules["season"] < season)
                | ((schedules["season"] == season) & (schedules["week"] <= week))
            )
        ]

        print(f"[Processing] Filtered schedules: {len(schedules)} games")

        # Process data
        df = self._process_data(player_stats, pbp, roster, schedules, season, week)

        print(f"[Processing] Processed data: {len(df)} total records")

        # Split into historical and current week
        current_week = df[(df["season"] == season) & (df["week"] == week)].copy()
        print(f"[Processing] Current week: {len(current_week)} records")

        # Create historical dataset
        ref_week = week - 1 if week > 1 else 20

        # FIXED: More lenient historical data filtering
        if week == 1:
            # For week 1, use all previous season data
            historical = df[df["season"] < season].copy()
        else:
            # For other weeks, include previous weeks of current season
            historical = df[
                (df["season"] < season)
                | ((df["season"] == season) & (df["week"] < week))
            ].copy()

        print(f"[Processing] Historical before filtering: {len(historical)} records")

        # Filter to only games that were played
        historical = historical[historical["played"] == 1].copy()
        print(
            f"[Processing] Historical after 'played' filter: {len(historical)} records"
        )

        # Keep only last 2 seasons (season - 2) plus completed weeks of current season
        # For 2025: use 2023, 2024, and weeks in 2025 that have already happened
        min_season = season - HISTORICAL_SEASONS  # season - 2
        historical = historical[historical["season"] >= min_season].copy()
        print(
            f"[Processing] Historical after season filter (>={min_season}, last {HISTORICAL_SEASONS} seasons): {len(historical)} records"
        )

        # Save the full processed data for future use
        self.save_historical_data(df)

        return {
            "df": historical.reset_index(drop=True),
            "current_week": current_week.reset_index(drop=True),
        }

    def _process_data(
        self,
        player_stats,
        pbp,
        roster,
        schedules,
        season,
        week,
        current_week_only=False,
    ):
        """Process raw data into features - Complete Feature Overhaul"""
        from .feature_engineering import (
            calculate_player_features,
            calculate_team_context_features,
            calculate_team_shares,
            calculate_defensive_features,
        )
        from .config import POSITIONS

        print("[Processing] Building feature dataframe with new feature blueprint...")
        print(f"[Debug] Input data sizes:")
        print(f"  - player_stats: {len(player_stats)}")
        print(f"  - pbp: {len(pbp)}")
        print(f"  - roster: {len(roster)}")
        print(f"  - schedules: {len(schedules)}")

        # Build weekly stats from player_stats (base stats for feature engineering)
        weekly_stats_cols = [
                "player_id",
                "season",
                "week",
                "rushing_yards",
                "rushing_tds",
                "receiving_yards",
                "receiving_tds",
                "passing_tds",
                "passing_yards",
            ]
        # Add receptions and targets if they exist
        if "receptions" in player_stats.columns:
            weekly_stats_cols.append("receptions")
        if "targets" in player_stats.columns:
            weekly_stats_cols.append("targets")
        if "carries" in player_stats.columns:
            weekly_stats_cols.append("carries")
            
        weekly_stats = player_stats[weekly_stats_cols].copy()
        print(f"[Debug] weekly_stats from player_stats: {len(weekly_stats)} records")
        
        # For future weeks, create placeholder rows so EWMA features can be calculated
        # Get all players from schedule (for current week) and roster
        if len(schedules) > 0:
            current_week_schedule = schedules[(schedules["season"] == season) & (schedules["week"] == week)]
            if len(current_week_schedule) > 0:
                # Get all players from roster who are active
                active_players = roster[roster["status"] == "ACT"][
                    ["gsis_id", "season", "team"]
                ].drop_duplicates()
                active_players = active_players.rename(columns={"gsis_id": "player_id"})
                
                # Create placeholder rows for current week players
                current_week_players = []
                for _, game in current_week_schedule.iterrows():
                    # Add home team players
                    home_players = active_players[
                        (active_players["season"] == season) & 
                        (active_players["team"] == game["home_team"])
                    ][["player_id"]].copy()
                    home_players["season"] = season
                    home_players["week"] = week
                    
                    # Add away team players
                    away_players = active_players[
                        (active_players["season"] == season) & 
                        (active_players["team"] == game["away_team"])
                    ][["player_id"]].copy()
                    away_players["season"] = season
                    away_players["week"] = week
                    
                    current_week_players.append(home_players)
                    current_week_players.append(away_players)
                
                if len(current_week_players) > 0:
                    current_week_df = pd.concat(current_week_players, ignore_index=True).drop_duplicates()
                    
                    # Check which players don't have weekly_stats for current week
                    existing_stats = weekly_stats[
                        (weekly_stats["season"] == season) & (weekly_stats["week"] == week)
                    ][["player_id"]].drop_duplicates()
                    
                    missing_players = current_week_df[
                        ~current_week_df["player_id"].isin(existing_stats["player_id"])
                    ]
                    
                    if len(missing_players) > 0:
                        # Create placeholder rows with NaN/0 for all stat columns
                        placeholder_stats = missing_players.copy()
                        stat_cols = [col for col in weekly_stats_cols if col not in ["player_id", "season", "week"]]
                        for col in stat_cols:
                            placeholder_stats[col] = 0  # Use 0 for counts, will be handled properly in EWMA
                        
                        # Concatenate with existing weekly_stats
                        weekly_stats = pd.concat([weekly_stats, placeholder_stats], ignore_index=True)
                        print(f"[Debug] Added {len(missing_players)} placeholder rows for current week players without stats")
                        print(f"[Debug] weekly_stats after adding placeholders: {len(weekly_stats)} records")
        
        # Ensure pbp has game_id, season, week for merging
        if "game_id" not in pbp.columns and "gameId" in pbp.columns:
            pbp = pbp.rename(columns={"gameId": "game_id"})
        if "season" not in pbp.columns and "game_id" in pbp.columns:
            # Try to extract season from game_id or add from schedules
            pbp = pd.merge(
                pbp,
                schedules[["game_id", "season", "week"]].drop_duplicates(),
                on="game_id",
                how="left"
            )

        # Build roster summary
        roster_summary = (
            roster[roster.position.isin(POSITIONS)][
                [
                    "gsis_id",
                    "position",
                    "season",
                    "team",
                    "rookie_year",
                    "full_name",
                    "status",
                ]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        print(
            f"[Debug] roster_summary after position filter: {len(roster_summary)} records"
        )

        roster_summary["rookie"] = np.where(
            roster_summary.rookie_year == roster_summary.season, 1, 0
        )
        roster_summary = roster_summary[roster_summary.status == "ACT"]
        print(f"[Debug] roster_summary after ACT filter: {len(roster_summary)} records")

        # Build games
        games = schedules[
            [
                "season",
                "week",
                "home_moneyline",
                "game_id",
                "home_team",
                "away_team",
                "home_qb_id",
                "away_qb_id",
            ]
        ].copy()
        print(f"[Debug] games: {len(games)} records")

        games["home_wp"] = games["home_moneyline"].apply(
            lambda x: american_odds_to_probability(x) if pd.notnull(x) else 0.5
        )

        print(games)

        # Merge home and away games
        print(f"[Debug] Merging home games...")
        home_games = pd.merge(
            roster_summary,
            games,
            how="left",
            left_on=["season", "team"],
            right_on=["season", "home_team"],
        ).dropna()

        print(home_games)
        home_games["home"] = 1
        home_games["qb_id"] = home_games["home_qb_id"]
        print(f"[Debug] home_games: {len(home_games)} records")

        print(f"[Debug] Merging away games...")
        away_games = pd.merge(
            roster_summary,
            games,
            how="left",
            left_on=["season", "team"],
            right_on=["season", "away_team"],
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
            raise ValueError(
                "No games found after merging roster with schedule. Team name mismatch?"
            )

        # Build main dataframe
        df = games_merged[
            [
                "gsis_id",
                "position",
                "full_name",
                "game_id",
                "team",
                "season",
                "week",
                "rookie",
                "home_team",
                "away_team",
                "home_wp",
                "home",
                "qb_id",
            ]
        ]
        df = df.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
        print(f"[Debug] df after initial build: {len(df)} records")

        # Merge touchdowns
        # Target variable: Binary (1 if player scored ANY touchdown, 0 otherwise)
        # - For TE/WR/RB: Includes both rushing AND receiving touchdowns
        # - For QB: Only rushing touchdowns (QBs don't catch passes)
        # td_player_id in play-by-play data captures the scorer regardless of play type
        touchdowns = pbp[pbp["touchdown"] == 1][
            ["td_player_id", "game_id"]
        ].drop_duplicates()
        print(f"[Debug] touchdowns: {len(touchdowns)} records")

        mdf = pd.merge(
            df,
            touchdowns,
            how="left",
            left_on=["player_id", "game_id"],
            right_on=["td_player_id", "game_id"],
            indicator=True,
        )
        # Binary target: 1 = scored any TD (rushing or receiving), 0 = no TD
        mdf["touchdown"] = mdf["_merge"].apply(lambda x: 1 if x == "both" else 0)
        df = mdf.drop(["_merge", "td_player_id"], axis=1)
        print(f"[Debug] df after touchdown merge: {len(df)} records")
        
        # Calculate touchdown attempts and red zone completion % (red zone targets + red zone carries combined)
        print(f"[Debug] Calculating touchdown attempts and red zone stats from red zone plays...")
        
        # Red zone attempts (within 20 yards of end zone)
        if "yardline_100" in pbp.columns:
            rz_stats_list = []
            
            # Red zone targets for receivers (all pass attempts in red zone)
            if "receiver_player_id" in pbp.columns and "pass" in pbp.columns:
                rz_targets_data = pbp[
                    (pbp["yardline_100"] <= 20) & 
                    (pbp["pass"] == 1) & 
                    (pbp["receiver_player_id"].notna())
                ].groupby(["receiver_player_id", "game_id", "season", "week"]).size().reset_index(name="red_zone_targets")
                rz_targets_data = rz_targets_data.rename(columns={"receiver_player_id": "player_id"})
                
                # Red zone receptions (completed passes in red zone)
                rz_receptions_data = pbp[
                    (pbp["yardline_100"] <= 20) & 
                    (pbp["pass"] == 1) & 
                    (pbp["complete_pass"] == 1) &
                    (pbp["receiver_player_id"].notna())
                ].groupby(["receiver_player_id", "game_id", "season", "week"]).size().reset_index(name="red_zone_receptions")
                rz_receptions_data = rz_receptions_data.rename(columns={"receiver_player_id": "player_id"})
                
                # Merge red zone targets and receptions
                rz_pass_stats = pd.merge(
                    rz_targets_data,
                    rz_receptions_data,
                    on=["player_id", "game_id", "season", "week"],
                    how="outer"
                )
                # Don't fillna - preserve NaN for players without red zone stats
                rz_pass_stats["red_zone_targets"] = rz_pass_stats["red_zone_targets"].fillna(0).astype(int)  # Keep 0 for count stats
                rz_pass_stats["red_zone_receptions"] = rz_pass_stats["red_zone_receptions"].fillna(0).astype(int)  # Keep 0 for count stats
                rz_stats_list.append(rz_pass_stats)
            
            # Red zone carries for rushers (all rush attempts in red zone)
            if "rusher_player_id" in pbp.columns and "rush" in pbp.columns:
                rz_carries_data = pbp[
                    (pbp["yardline_100"] <= 20) & 
                    (pbp["rush"] == 1) & 
                    (pbp["rusher_player_id"].notna())
                ].groupby(["rusher_player_id", "game_id", "season", "week"]).size().reset_index(name="red_zone_carries")
                rz_carries_data = rz_carries_data.rename(columns={"rusher_player_id": "player_id"})
                rz_stats_list.append(rz_carries_data)
            
            # Combine all red zone stats
            if len(rz_stats_list) > 0:
                rz_stats = rz_stats_list[0].copy()
                for stat_df in rz_stats_list[1:]:
                    rz_stats = pd.merge(
                        rz_stats,
                        stat_df,
                        on=["player_id", "game_id", "season", "week"],
                        how="outer"
                    )
                
                # Fill missing values
                if "red_zone_targets" in rz_stats.columns:
                    rz_stats["red_zone_targets"] = rz_stats["red_zone_targets"].fillna(0).astype(int)
                else:
                    rz_stats["red_zone_targets"] = 0
                    
                if "red_zone_receptions" in rz_stats.columns:
                    rz_stats["red_zone_receptions"] = rz_stats["red_zone_receptions"].fillna(0).astype(int)
                else:
                    rz_stats["red_zone_receptions"] = 0
                    
                if "red_zone_carries" in rz_stats.columns:
                    rz_stats["red_zone_carries"] = rz_stats["red_zone_carries"].fillna(0).astype(int)
                else:
                    rz_stats["red_zone_carries"] = 0
                
                # Calculate touchdown attempts (targets + carries)
                rz_stats["touchdown_attempts"] = rz_stats["red_zone_targets"] + rz_stats["red_zone_carries"]
                
                # Calculate red zone completion % (receptions / targets, avoid division by zero)
                rz_stats["red_zone_completion_pct"] = np.where(
                    rz_stats["red_zone_targets"] > 0,
                    rz_stats["red_zone_receptions"] / rz_stats["red_zone_targets"],
                    0.0
                )
                
                # Merge with main dataframe (include red_zone_receptions and red_zone_targets for season aggregation)
                rz_merge_cols = ["player_id", "game_id", "season", "week", "touchdown_attempts", "red_zone_completion_pct"]
                if "red_zone_receptions" in rz_stats.columns:
                    rz_merge_cols.append("red_zone_receptions")
                if "red_zone_targets" in rz_stats.columns:
                    rz_merge_cols.append("red_zone_targets")
                    
                df = pd.merge(
                    df,
                    rz_stats[rz_merge_cols],
                    on=["player_id", "game_id", "season", "week"],
                    how="left"
                )
                # Don't fillna - preserve NaN for players without red zone attempts
                df["touchdown_attempts"] = df["touchdown_attempts"]  # Keep as int or NaN
                df["red_zone_completion_pct"] = df["red_zone_completion_pct"]  # Keep as float or NaN
                if "red_zone_receptions" in df.columns:
                    df["red_zone_receptions"] = df["red_zone_receptions"]  # Keep as int or NaN
                if "red_zone_targets" in df.columns:
                    df["red_zone_targets"] = df["red_zone_targets"]  # Keep as int or NaN
            else:
                df["touchdown_attempts"] = np.nan
                df["red_zone_completion_pct"] = np.nan
        else:
            df["touchdown_attempts"] = np.nan
            df["red_zone_completion_pct"] = np.nan
        
        print(f"[Debug] df after touchdown attempts and red zone stats merge: {len(df)} records")

        # Set matchup and home/away
        df["against"] = np.where(
            df["team"] == df["home_team"], df["away_team"], df["home_team"]
        )
        df["is_home"] = np.where(df["team"] == df["home_team"], 1, 0)
        
        # Note: defense categorical feature removed - high cardinality causes overfitting
        # Defense quality is captured by defensive EWMA features instead
        
        # Merge weekly stats for feature engineering
        df = pd.merge(df, weekly_stats, on=["player_id", "season", "week"], how="left")
        print(f"[Debug] df after weekly_stats merge: {len(df)} records")
        
        # Use new feature engineering to calculate all player features
        print(f"[Debug] Calculating player features with new feature engineering...")
        # Add game_id to weekly_stats for feature engineering
        weekly_stats_with_game = pd.merge(
            weekly_stats,
            df[["player_id", "game_id", "season", "week"]].drop_duplicates(),
            on=["player_id", "season", "week"],
            how="left"
        )
        player_features_df = calculate_player_features(player_stats, pbp, weekly_stats_with_game)
        
        # Add team and position columns to player_features_df for team context and QB stats calculations
        # Merge team and position from df to player_features_df
        team_pos_mapping = df[["player_id", "game_id", "season", "week", "team", "position"]].drop_duplicates()
        merge_cols = ["player_id", "season", "week"]
        if "game_id" in player_features_df.columns:
            merge_cols.append("game_id")
        player_features_df = pd.merge(
            player_features_df,
            team_pos_mapping,
            on=merge_cols,
            how="left"
        )
        print(f"[Debug] Added team and position columns to player_features_df")

        # Red zone touches covers both targets and carries for all positions
        
        # Set position-inapplicable stats to NaN (not 0) to indicate feature doesn't apply
        # This allows the model to distinguish between "0 = no activity" vs "NaN = feature doesn't apply"
        # - QBs: receiving stats = NaN (QBs never catch the ball)
        # - WR/TE: rushing stats = NaN (rushing is not their primary way to score)
        # - RB: receiving stats = NaN (receiving is less significant, 0 receiving is normal)
        # For applicable positions, 0 is valid (means they had no activity in that category)
        if "position" in player_features_df.columns:
            # QB receiving stats
            qb_mask = player_features_df["position"] == "QB"
            qb_receiving_features = ["targets_ewma", "receptions_ewma", "receiving_yards_ewma", "receiving_touchdowns_ewma"]
            for feature in qb_receiving_features:
                if feature in player_features_df.columns:
                    player_features_df.loc[qb_mask, feature] = np.nan
            print(f"[Debug] Set receiving stats to NaN for {qb_mask.sum()} QB players")
            
            # WR/TE rushing stats (rushing is less significant for WR/TE)
            wr_te_mask = player_features_df["position"].isin(["WR", "TE"])
            wr_te_rushing_features = ["carries_ewma", "rushing_yards_ewma", "rushing_touchdowns_ewma"]
            for feature in wr_te_rushing_features:
                if feature in player_features_df.columns:
                    player_features_df.loc[wr_te_mask, feature] = np.nan
            print(f"[Debug] Set rushing stats to NaN for {wr_te_mask.sum()} WR/TE players")
            
            # RB receiving stats (receiving is less significant for RB)
            rb_mask = player_features_df["position"] == "RB"
            rb_receiving_features = ["targets_ewma", "receptions_ewma", "receiving_yards_ewma", "receiving_touchdowns_ewma"]
            for feature in rb_receiving_features:
                if feature in player_features_df.columns:
                    player_features_df.loc[rb_mask, feature] = np.nan
            print(f"[Debug] Set receiving stats to NaN for {rb_mask.sum()} RB players")
        
        # Calculate QB stats separately (only for QBs, then will merge onto all players)
        from .feature_engineering import calculate_qb_stats
        print(f"[Debug] Calculating QB stats for merging onto all players...")
        qb_stats_df = calculate_qb_stats(player_features_df, position_col=player_features_df.get("position"))
        
        # Merge QB stats onto all players by qb_id, season, week
        # qb_stats_df has player_id (which is the qb_id), season, week, and QB stats
        # df has qb_id column that identifies which QB played for each player's team
        if len(qb_stats_df) > 0 and "qb_id" in df.columns:
            # Drop existing QB stat columns from player_features_df if they exist
            # Only keep yardage stats, not TD counts (to avoid spurious correlations)
            qb_stat_cols = ["qb_passing_yards_ewma", "qb_rushing_yards_ewma"]
            for col in qb_stat_cols:
                if col in player_features_df.columns:
                    player_features_df = player_features_df.drop(columns=[col])
            
            # First, merge qb_id from df onto player_features_df
            if "qb_id" not in player_features_df.columns:
                # Get qb_id from df by player_id, season, week
                qb_id_map = df[["player_id", "season", "week", "qb_id"]].drop_duplicates()
                player_features_df = pd.merge(
                    player_features_df,
                    qb_id_map,
                    on=["player_id", "season", "week"],
                    how="left"
                )
            
            # Rename qb_stats_df player_id to qb_id for merging
            qb_stats_for_merge = qb_stats_df.rename(columns={"player_id": "qb_id"})
            
            # Merge QB stats onto player_features_df by qb_id, season, week
            player_features_df = pd.merge(
                player_features_df,
                qb_stats_for_merge[["qb_id", "season", "week"] + qb_stat_cols],
                on=["qb_id", "season", "week"],
                how="left"
            )
            
            # Set QB stats to NaN for QBs themselves (they should only have their personal stats)
            if "position" in player_features_df.columns:
                qb_mask = player_features_df["position"] == "QB"
                for col in qb_stat_cols:
                    if col in player_features_df.columns:
                        player_features_df.loc[qb_mask, col] = np.nan
                print(f"[Debug] Set QB stats to NaN for {qb_mask.sum()} QB players (they use personal stats)")
            
            # Note: QB TD stats removed to avoid spurious correlations
            # Only QB yardage stats are kept (qb_passing_yards_ewma, qb_rushing_yards_ewma)
            
            print(f"[Debug] Merged QB stats onto all players by qb_id, season, week")
        else:
            print(f"[Warning] Could not merge QB stats - qb_stats_df is empty or qb_id column missing from df")
        
        # Merge player features back (exclude base columns that are already in df)
        # Drop base stat columns that exist in both df and player_features_df to avoid _feat suffixes
        base_stats_to_drop = ["rushing_yards", "rushing_tds", "receiving_yards", 
                              "receiving_tds", "passing_tds", "passing_yards",
                              "receptions", "targets", "carries", "touches"]  # Add base stats that might cause conflicts
        player_features_clean = player_features_df.drop(
            columns=[col for col in base_stats_to_drop if col in player_features_df.columns], 
            errors="ignore"
        )
        
        merge_cols = ["player_id", "season", "week"]
        if "game_id" in player_features_df.columns:
            merge_cols.append("game_id")

        df = pd.merge(
            df,
            player_features_clean,
            on=merge_cols,
            how="left",
            suffixes=("", "_feat")
        )
        
        # Clean up any _feat columns - if they exist, they're duplicates from the merge
        # Drop the _feat versions since we want to keep the original columns from df
        feat_cols = [col for col in df.columns if col.endswith('_feat')]
        if feat_cols:
            print(f"[Debug] Dropping {len(feat_cols)} duplicate _feat columns: {feat_cols[:5]}...")
            df = df.drop(columns=feat_cols)
        
        print(f"[Debug] df after player features merge: {len(df)} records")

        # Calculate team context features
        print(f"[Debug] Calculating team context features...")
        team_context = calculate_team_context_features(player_features_df, pbp, schedules)
        
        # Drop any existing team context columns from df to avoid merge conflicts
        team_context_cols = ["team_total_red_zone_touches_ewma", 
                             "team_play_volume_ewma", "team_win_probability", "spread_line"]
        for col in team_context_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        df = pd.merge(
            df,
            team_context,
            on=["team", "game_id", "season", "week"],
            how="left"
        )
        
        # Clean up any duplicate columns with suffixes from merge
        for col in df.columns:
            if col.endswith('_x') or col.endswith('_y'):
                base_col = col[:-2]  # Remove _x or _y suffix
                if base_col in team_context.columns:
                    # Keep the merged value (usually _x), drop the other
                    if col.endswith('_y'):
                        df = df.drop(columns=[col])
                    else:
                        # Rename _x back to original name
                        df = df.rename(columns={col: base_col})
        
        print(f"[Debug] df after team context merge: {len(df)} records")

        # Calculate team shares
        print(f"[Debug] Calculating team shares...")
        df = calculate_team_shares(df, team_context)
        print(f"[Debug] df after team shares: {len(df)} records")
        
        # Calculate defensive features
        print(f"[Debug] Calculating defensive features...")
        defensive_features = calculate_defensive_features(pbp, schedules)
        df = pd.merge(
            df,
            defensive_features,
            left_on=["against", "game_id", "season", "week"],
            right_on=["team", "game_id", "season", "week"],
            how="left",
            suffixes=("", "_def")
        )
        df = df.drop(["team_def"], axis=1, errors="ignore")
        print(f"[Debug] df after defensive features merge: {len(df)} records")
        
        # Calculate position-normalized touches_ewma to remove RB/QB bias
        # Normalize touches_ewma by position average (WRs/TEs naturally have lower touches than RBs)
        if "touches_ewma" in df.columns and "position" in df.columns:
            print(f"[Debug] Calculating position-normalized touches_ewma...")
            # Calculate position average touches_ewma (grouped by position, season, week)
            # This gives us the average touches for each position in each week
            position_avg_touches = df.groupby(["position", "season", "week"])["touches_ewma"].mean().reset_index()
            position_avg_touches = position_avg_touches.rename(columns={"touches_ewma": "position_avg_touches_ewma"})
            
            # Merge position averages back
            df = pd.merge(
                df,
                position_avg_touches,
                on=["position", "season", "week"],
                how="left"
            )
            
            # Calculate normalized touches: player_touches / position_avg_touches
            # This makes a WR with 5 touches (high for a WR) comparable to an RB with 15 touches (high for an RB)
            df["touches_ewma_position_normalized"] = (
                df["touches_ewma"] / (df["position_avg_touches_ewma"] + 1e-6)  # Add small epsilon to avoid division by zero
            )
            
            # Set to NaN if touches_ewma was NaN (preserve missing values)
            df.loc[df["touches_ewma"].isna(), "touches_ewma_position_normalized"] = np.nan
            
            # Drop the intermediate position_avg_touches_ewma column
            df = df.drop(columns=["position_avg_touches_ewma"])
            
            # Debug: Show sample of normalized values
            if len(df) > 0:
                sample_normalized = df[df["touches_ewma_position_normalized"].notna()].groupby("position")["touches_ewma_position_normalized"].describe()
                print(f"[Debug] Position-normalized touches_ewma stats by position:")
                print(sample_normalized)

        # Calculate position-normalized red_zone_touches_ewma to remove RB/QB bias
        # Normalize red_zone_touches_ewma by position average (WRs/TEs naturally have fewer red zone touches than RBs)
        if "red_zone_touches_ewma" in df.columns and "position" in df.columns:
            print(f"[Debug] Calculating position-normalized red_zone_touches_ewma...")
            # Calculate position average red_zone_touches_ewma (grouped by position, season, week)
            position_avg_rz_touches = df.groupby(["position", "season", "week"])["red_zone_touches_ewma"].mean().reset_index()
            position_avg_rz_touches = position_avg_rz_touches.rename(columns={"red_zone_touches_ewma": "position_avg_red_zone_touches_ewma"})
            
            # Merge position averages back
            df = pd.merge(
                df,
                position_avg_rz_touches,
                on=["position", "season", "week"],
                how="left"
            )
            
            # Calculate normalized red zone touches: player_rz_touches / position_avg_rz_touches
            df["red_zone_touches_ewma_position_normalized"] = (
                df["red_zone_touches_ewma"] / (df["position_avg_red_zone_touches_ewma"] + 1e-6)
            )
            
            # Set to NaN if red_zone_touches_ewma was NaN (preserve missing values)
            df.loc[df["red_zone_touches_ewma"].isna(), "red_zone_touches_ewma_position_normalized"] = np.nan
            
            # Drop the intermediate position_avg_red_zone_touches_ewma column
            df = df.drop(columns=["position_avg_red_zone_touches_ewma"])
            
            # Debug: Show sample of normalized values
            if len(df) > 0:
                sample_rz_normalized = df[df["red_zone_touches_ewma_position_normalized"].notna()].groupby("position")["red_zone_touches_ewma_position_normalized"].describe()
                print(f"[Debug] Position-normalized red_zone_touches_ewma stats by position:")
                print(sample_rz_normalized)
        
        # Calculate reception_rate_ewma (receptions / targets)
        # This is more meaningful than separate receptions/targets features:
        # - High rate = reliable hands (good for TD scoring)
        # - Low rate = drops/poor hands (bad for TD scoring)
        # - Works for all positions (WR/TE/RB who catch passes)
        if "receptions_ewma" in df.columns and "targets_ewma" in df.columns:
            print(f"[Debug] Calculating reception_rate_ewma...")
            # Calculate rate: receptions / targets
            # Use small epsilon to avoid division by zero
            df["reception_rate_ewma"] = (
                df["receptions_ewma"] / (df["targets_ewma"] + 1e-6)
            )
            
            # Set to NaN if either component is NaN (preserve missing values)
            df.loc[df["receptions_ewma"].isna() | df["targets_ewma"].isna(), "reception_rate_ewma"] = np.nan
            
            # Cap at 1.0 (can't have more receptions than targets)
            df.loc[df["reception_rate_ewma"] > 1.0, "reception_rate_ewma"] = 1.0
            
            # Debug: Show sample of reception rates by position
            if len(df) > 0:
                sample_reception_rate = df[df["reception_rate_ewma"].notna()].groupby("position")["reception_rate_ewma"].describe()
                print(f"[Debug] Reception rate (receptions/targets) stats by position:")
                print(sample_reception_rate)
        
        # Calculate combined total_yards_ewma and total_touchdowns_ewma (position-agnostic)
        # This combines rushing + receiving into unified metrics that work for all positions
        # WRs get most from receiving, RBs from rushing, but both contribute to total production
        print(f"[Debug] Calculating combined total yards and touchdowns...")
        
        # Total yards = receiving + rushing (works for all positions)
        if "receiving_yards_ewma" in df.columns and "rushing_yards_ewma" in df.columns:
            # Sum receiving and rushing yards, handling NaN properly
            df["total_yards_ewma"] = (
                df["receiving_yards_ewma"].fillna(0) + df["rushing_yards_ewma"].fillna(0)
            )
            # Set to NaN if both were originally NaN (not just 0)
            both_nan = df["receiving_yards_ewma"].isna() & df["rushing_yards_ewma"].isna()
            df.loc[both_nan, "total_yards_ewma"] = np.nan
            # If one is NaN and other is 0, set to NaN (no production)
            one_nan_zero = (
                (df["receiving_yards_ewma"].isna() & (df["rushing_yards_ewma"].fillna(0) == 0)) |
                (df["rushing_yards_ewma"].isna() & (df["receiving_yards_ewma"].fillna(0) == 0))
            )
            df.loc[one_nan_zero, "total_yards_ewma"] = np.nan
        elif "receiving_yards_ewma" in df.columns:
            df["total_yards_ewma"] = df["receiving_yards_ewma"]
        elif "rushing_yards_ewma" in df.columns:
            df["total_yards_ewma"] = df["rushing_yards_ewma"]
        else:
            df["total_yards_ewma"] = np.nan
        
        # Total touchdowns = receiving + rushing (works for all positions)
        if "receiving_touchdowns_ewma" in df.columns and "rushing_touchdowns_ewma" in df.columns:
            # Sum receiving and rushing TDs, handling NaN properly
            df["total_touchdowns_ewma"] = (
                df["receiving_touchdowns_ewma"].fillna(0) + df["rushing_touchdowns_ewma"].fillna(0)
            )
            # Set to NaN if both were originally NaN
            both_nan = df["receiving_touchdowns_ewma"].isna() & df["rushing_touchdowns_ewma"].isna()
            df.loc[both_nan, "total_touchdowns_ewma"] = np.nan
            # If one is NaN and other is 0, set to NaN (no production)
            one_nan_zero = (
                (df["receiving_touchdowns_ewma"].isna() & (df["rushing_touchdowns_ewma"].fillna(0) == 0)) |
                (df["rushing_touchdowns_ewma"].isna() & (df["receiving_touchdowns_ewma"].fillna(0) == 0))
            )
            df.loc[one_nan_zero, "total_touchdowns_ewma"] = np.nan
        elif "receiving_touchdowns_ewma" in df.columns:
            df["total_touchdowns_ewma"] = df["receiving_touchdowns_ewma"]
        elif "rushing_touchdowns_ewma" in df.columns:
            df["total_touchdowns_ewma"] = df["rushing_touchdowns_ewma"]
        else:
            df["total_touchdowns_ewma"] = np.nan
        
        # Debug: Show sample of combined totals by position
        if len(df) > 0:
            if "total_yards_ewma" in df.columns:
                sample_total_yds = df[df["total_yards_ewma"].notna()].groupby("position")["total_yards_ewma"].describe()
                print(f"[Debug] Total yards (receiving + rushing) stats by position:")
                print(sample_total_yds)
            if "total_touchdowns_ewma" in df.columns:
                sample_total_tds = df[df["total_touchdowns_ewma"].notna()].groupby("position")["total_touchdowns_ewma"].describe()
                print(f"[Debug] Total touchdowns (receiving + rushing) stats by position:")
                print(sample_total_tds)
        
        # Normalize all player usage and performance features by position to remove position bias
        # This scales features relative to position average (e.g., high-usage WR comparable to high-usage RB)
        # Must happen AFTER total_yards_ewma and total_touchdowns_ewma are calculated
        print(f"[Debug] Normalizing player usage and performance features by position...")
        
        # Features to normalize by position (player usage and performance metrics)
        features_to_normalize = [
            "targets_ewma",
            "total_yards_ewma",
            "total_touchdowns_ewma",
            "reception_rate_ewma",
            "red_zone_touch_share_ewma",
            "recent_total_breakout_tds",
            "recent_total_breakout_yards",
        ]
        
        # Normalize each feature by position average
        for feature in features_to_normalize:
            if feature in df.columns and "position" in df.columns:
                print(f"[Debug] Normalizing {feature} by position...")
                
                # Calculate position average for this feature (grouped by position, season, week)
                position_avg = df.groupby(["position", "season", "week"])[feature].mean().reset_index()
                position_avg = position_avg.rename(columns={feature: f"position_avg_{feature}"})
                
                # Merge position averages back
                df = pd.merge(
                    df,
                    position_avg,
                    on=["position", "season", "week"],
                    how="left"
                )
                
                # Calculate normalized value: player_value / position_avg
                normalized_feature = f"{feature}_position_normalized"
                min_avg = 0.1 if "td" in feature.lower() else 1.0  # Lower threshold for TD features
                df[normalized_feature] = (
                    df[feature] / (df[f"position_avg_{feature}"].clip(lower=min_avg))
                )
                
                # Set to NaN if original was NaN
                df.loc[df[feature].isna(), normalized_feature] = np.nan
                
                # Cap at reasonable maximum (5x average) to prevent extreme values
                max_normalized = 5.0
                df.loc[df[normalized_feature] > max_normalized, normalized_feature] = max_normalized
                
                # Drop intermediate column
                df = df.drop(columns=[f"position_avg_{feature}"])
                
                # Debug: Show sample of normalized values
                if len(df) > 0:
                    sample_normalized = df[df[normalized_feature].notna()].groupby("position")[normalized_feature].describe()
                    print(f"[Debug] Position-normalized {feature} stats by position:")
                    print(sample_normalized)
        
        # Rename 'home' to 'is_home' for consistency with feature blueprint
        if "home" in df.columns:
            df["is_home"] = df["home"]
            df = df.drop(["home"], axis=1)
        
        # Add injury status placeholder (not in new features, but keep for compatibility)
        df["report_status"] = "Healthy"
        
        # Clean up duplicate columns from merges
        df = df.loc[:, ~df.columns.duplicated()]

        # Clean data
        df = df.drop_duplicates()
        print(f"[Debug] df after drop_duplicates: {len(df)} records")

        # Filter out players with fewer than EWMA_WEEKS total records (minimum history requirement)
        # This ensures EWMA features have meaningful values
        # NOTE: This counts ALL records across ALL historical seasons (season - 2, so for 2025: 2023, 2024, 2025 completed weeks)
        # So a player who played 5 games in 2023 and nothing else will be included if they have >= EWMA_WEEKS total games
        # This filter applies to both training data and ensures only players with sufficient history are used
        from .config import EWMA_WEEKS
        if "player_id" in df.columns:
            player_record_counts = df.groupby("player_id").size()
            players_with_enough_history = player_record_counts[player_record_counts >= EWMA_WEEKS].index
            before_count = len(df)
            df = df[df["player_id"].isin(players_with_enough_history)].copy()
            after_count = len(df)
            if before_count > after_count:
                print(f"[Filter] Removed {before_count - after_count} players with < {EWMA_WEEKS} total records from training data")
                print(f"[Filter] Remaining players: {len(players_with_enough_history)} ({after_count} records)")
                min_season_used = season - HISTORICAL_SEASONS
                print(f"[Filter] This includes players from seasons {min_season_used}+ (season - {HISTORICAL_SEASONS}) as long as they have >= {EWMA_WEEKS} total games")

        # Filter out players marked as "Out" (if report_status exists)
        if "report_status" in df.columns:
            df.loc[~df["report_status"].isin(REPORT_STATUS_ORDER), "report_status"] = "Minor"
        df = df[~(df.report_status == "Out")]
        print(f"[Debug] df after removing 'Out' status: {len(df)} records")

        # Add played indicator (1 if player had stats, 0 otherwise)
        print(f"[Debug] Adding played indicator...")
        played = weekly_stats[["player_id", "season", "week"]].drop_duplicates()
        played["played"] = 1
        df = pd.merge(df, played, how="left", on=["player_id", "season", "week"])
        df["played"] = df["played"].fillna(0).astype(int)
        print(f"[Debug] df after played merge: {len(df)} records")
        print(f"[Debug] Played distribution: {df['played'].value_counts().to_dict()}")

        # Ensure all required features from config are present (fill missing with NaN, not 0)
        from .config import FEATURES
        for feature in FEATURES:
            if feature not in df.columns:
                print(f"[Warning] Missing feature '{feature}', adding with NaN (not 0)")
                if feature == "position":
                    # Position should already be there, but if not, set to unknown
                    df[feature] = df.get("position", "UNK")
                elif feature == "is_home":
                    df[feature] = df.get("is_home", 0)  # Keep 0 for is_home (it's a binary flag)
                else:
                    df[feature] = np.nan  # Use NaN instead of 0 for missing features

        print(f"[Debug] FINAL df: {len(df)} records, columns: {len(df.columns)}")
        print(f"[Debug] Sample feature columns: {[col for col in df.columns if col in FEATURES][:10]}")
        return df
