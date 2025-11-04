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
from .utils import american_odds_to_probability, compute_red_zone_stats
from .models import TrainingData, DataCache


class NFLDataManager:
    """Manages NFL data loading and caching"""

    def __init__(self, data_source_type: str = "nflreadpy"):
        self.data_source = get_data_source(data_source_type)
        self.historical_data = None
        self.rz_stats_rolling = None
        self.rz_stats_prev = None

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
        """Determine which seasons need to be loaded"""
        if week == 1:
            # For week 1, don't include current season in historical
            return list(range(season - HISTORICAL_SEASONS - 1, season))
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

            # Compute red zone stats from play-by-play data
            print(f"[Debug] Computing red zone stats from play-by-play data...")
            self.rz_stats_rolling, self.rz_stats_prev = compute_red_zone_stats(pbp)

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

        # Compute red zone stats from play-by-play data
        print("[Processing] Computing red zone statistics from play-by-play data...")
        self.rz_stats_rolling, self.rz_stats_prev = compute_red_zone_stats(pbp)

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

        # Keep enough historical data (at least 3 seasons)
        min_season = season - HISTORICAL_SEASONS
        historical = historical[historical["season"] >= min_season].copy()
        print(
            f"[Processing] Historical after season filter (>={min_season}): {len(historical)} records"
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
        """Process raw data into features"""
        from .utils import calculate_rolling_avg, calculate_prev, get_qb_rolling_stats

        print("[Processing] Building feature dataframe...")
        print(f"[Debug] Input data sizes:")
        print(f"  - player_stats: {len(player_stats)}")
        print(f"  - pbp: {len(pbp)}")
        print(f"  - roster: {len(roster)}")
        print(f"  - schedules: {len(schedules)}")

        # Build weekly stats from player_stats (includes receptions and targets)
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
        # Add receptions and targets if they exist in player_stats
        if "receptions" in player_stats.columns:
            weekly_stats_cols.append("receptions")
        if "targets" in player_stats.columns:
            weekly_stats_cols.append("targets")
            
        weekly_stats = player_stats[weekly_stats_cols].copy()
        print(f"[Debug] weekly_stats: {len(weekly_stats)} records")

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
                rz_pass_stats["red_zone_targets"] = rz_pass_stats["red_zone_targets"].fillna(0).astype(int)
                rz_pass_stats["red_zone_receptions"] = rz_pass_stats["red_zone_receptions"].fillna(0).astype(int)
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
                df["touchdown_attempts"] = df["touchdown_attempts"].fillna(0).astype(int)
                df["red_zone_completion_pct"] = df["red_zone_completion_pct"].fillna(0.0)
                if "red_zone_receptions" in df.columns:
                    df["red_zone_receptions"] = df["red_zone_receptions"].fillna(0).astype(int)
                if "red_zone_targets" in df.columns:
                    df["red_zone_targets"] = df["red_zone_targets"].fillna(0).astype(int)
            else:
                df["touchdown_attempts"] = 0
                df["red_zone_completion_pct"] = 0.0
        else:
            df["touchdown_attempts"] = 0
            df["red_zone_completion_pct"] = 0.0
        
        print(f"[Debug] df after touchdown attempts and red zone stats merge: {len(df)} records")

        # Set matchup and home/away
        df["against"] = np.where(
            df["team"] == df["home_team"], df["away_team"], df["home_team"]
        )
        df["home"] = np.where(df["team"] == df["home_team"], 1, 0)
        df["wp"] = np.where(
            df["team"] == df["home_team"], df["home_wp"], 1 - df["home_wp"]
        )
        df = df.drop(["home_team", "away_team", "home_wp"], axis=1)

        # Add injury status placeholder
        df["report_status"] = "Healthy"

        # Calculate comprehensive defensive stats
        print(f"[Debug] Calculating defensive stats...")
        
        # Defensive stats per game - total yards and touchdowns allowed
        defensive_stats = pbp.groupby(["game_id", "defteam", "season", "week"]).agg({
            "yards_gained": "sum",
            "touchdown": "sum",  # Touchdowns allowed
        }).reset_index()
        
        # Calculate passing yards allowed (yards on pass plays)
        if "pass" in pbp.columns:
            passing_defense = pbp[pbp["pass"] == 1].groupby(["game_id", "defteam"]).agg({
                "yards_gained": "sum"
            }).reset_index()
            passing_defense = passing_defense.rename(columns={"yards_gained": "passing_yards"})
            defensive_stats = pd.merge(defensive_stats, passing_defense, on=["game_id", "defteam"], how="left")
            defensive_stats["passing_yards"] = defensive_stats["passing_yards"].fillna(0)
        else:
            defensive_stats["passing_yards"] = 0
            
        # Calculate rushing yards allowed (yards on rush plays)
        if "rush" in pbp.columns:
            rushing_defense = pbp[pbp["rush"] == 1].groupby(["game_id", "defteam"]).agg({
                "yards_gained": "sum"
            }).reset_index()
            rushing_defense = rushing_defense.rename(columns={"yards_gained": "rushing_yards"})
            defensive_stats = pd.merge(defensive_stats, rushing_defense, on=["game_id", "defteam"], how="left")
            defensive_stats["rushing_yards"] = defensive_stats["rushing_yards"].fillna(0)
        else:
            defensive_stats["rushing_yards"] = 0
        
        # Calculate points allowed (touchdowns * 7 as approximation)
        defensive_stats["points_allowed"] = defensive_stats["touchdown"] * 7
        
        # Store season/week before dropping for later use
        defensive_stats_season_week = defensive_stats[["game_id", "season", "week"]].copy()
        
        defensive_stats = defensive_stats.drop(["season", "week"], axis=1)
        defensive_stats.sort_values(by=["defteam", "game_id"], inplace=True)
        
        # Verify required columns exist
        required_cols = ["yards_gained", "touchdown", "passing_yards", "rushing_yards", "points_allowed"]
        for col in required_cols:
            if col not in defensive_stats.columns:
                print(f"[Warning] Missing column '{col}' in defensive_stats, adding with 0")
                defensive_stats[col] = 0
        
        # Calculate rolling averages for defensive stats
        for col in required_cols:
            if col in defensive_stats.columns:
                defensive_stats[f"rolling_def_{col}"] = (
                    defensive_stats.groupby("defteam")[col]
                    .rolling(window=3, min_periods=1)
                    .mean()
                    .shift(1)
                    .reset_index(level=0, drop=True)
                )
        
        # Keep original rolling_yapg for backward compatibility
        if "rolling_def_yards_gained" in defensive_stats.columns:
            defensive_stats["rolling_yapg"] = defensive_stats["rolling_def_yards_gained"]
        else:
            defensive_stats["rolling_yapg"] = 0
        
        # Add season and week back for previous season calculations
        # Use the season/week we stored before dropping
        defensive_stats_with_season = pd.merge(
            defensive_stats,
            defensive_stats_season_week,
            on="game_id",
            how="left"
        )
        
        # Calculate previous season defensive stats
        print(f"[Debug] Calculating previous season defensive stats...")
        # Make sure all required columns exist before aggregating
        required_cols = ["yards_gained", "touchdown", "passing_yards", "rushing_yards", "points_allowed"]
        missing_cols = [col for col in required_cols if col not in defensive_stats_with_season.columns]
        if missing_cols:
            print(f"[Warning] Missing columns in defensive_stats_with_season: {missing_cols}")
            for col in missing_cols:
                defensive_stats_with_season[col] = 0
        
        def_season_stats = defensive_stats_with_season.groupby(["defteam", "season"]).agg({
            "yards_gained": "mean",
            "touchdown": "mean",
            "passing_yards": "mean",
            "rushing_yards": "mean",
            "points_allowed": "mean",
        }).reset_index()
        def_season_stats.columns = ["defteam", "season", "prev_def_yards_gained", 
                                    "prev_def_touchdown", "prev_def_passing_yards",
                                    "prev_def_rushing_yards", "prev_def_points_allowed"]
        def_season_stats["next_season"] = def_season_stats["season"] + 1

        df = pd.merge(
            df,
            defensive_stats[["game_id", "defteam", "rolling_def_yards_gained", "rolling_def_touchdown", 
                           "rolling_def_passing_yards", "rolling_def_rushing_yards", 
                           "rolling_def_points_allowed", "rolling_yapg"]],
            how="left",
            left_on=["game_id", "against"],
            right_on=["game_id", "defteam"],
        )
        
        # Merge previous season defensive stats
        df = pd.merge(
            df,
            def_season_stats[["defteam", "next_season", "prev_def_yards_gained", 
                             "prev_def_touchdown", "prev_def_passing_yards",
                             "prev_def_rushing_yards", "prev_def_points_allowed"]],
            how="left",
            left_on=["against", "season"],
            right_on=["defteam", "next_season"],
        )
        df = df.drop(["defteam", "next_season"], axis=1, errors="ignore")
        
        # Fill missing defensive stats with 0
        for col in ["rolling_def_yards_gained", "rolling_def_touchdown", 
                   "rolling_def_passing_yards", "rolling_def_rushing_yards", 
                   "rolling_def_points_allowed", "rolling_yapg",
                   "prev_def_yards_gained", "prev_def_touchdown",
                   "prev_def_passing_yards", "prev_def_rushing_yards",
                   "prev_def_points_allowed"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        print(f"[Debug] df after defensive_stats merge: {len(df)} records")

        # Merge weekly stats
        df = pd.merge(df, weekly_stats, on=["player_id", "season", "week"], how="left")
        print(f"[Debug] df after weekly_stats merge: {len(df)} records")

        # Calculate rolling averages
        print(f"[Debug] Calculating rolling averages...")
        df = df.sort_values(by=["player_id", "season", "week"])
        df = df.groupby(["player_id"], group_keys=False).apply(calculate_rolling_avg)
        print(f"[Debug] df after rolling avg: {len(df)} records")

        df = df.drop(
            [
                "rushing_yards",
                "rushing_tds",
                "receiving_yards",
                "receiving_tds",
                "passing_tds",
                "passing_yards",
                "receptions",
                "targets",
                "carries",
                # Note: touchdown_attempts and red_zone_completion_pct are kept as features
            ],
            axis=1,
            errors="ignore",
        )

        # Calculate QB rolling stats
        print(f"[Debug] Calculating QB rolling stats...")
        qbs = df["qb_id"].unique()
        qb_weekly_stats = weekly_stats[weekly_stats["player_id"].isin(qbs)][
            ["player_id", "season", "week", "passing_tds", "passing_yards"]
        ]
        df = pd.merge(
            df,
            qb_weekly_stats,
            left_on=["qb_id", "season", "week"],
            right_on=["player_id", "season", "week"],
            how="left",
            suffixes=("", "_drop"),
        )
        df = df.sort_values(by=["qb_id", "season", "week"])
        df = df.groupby(["qb_id"], group_keys=False).apply(get_qb_rolling_stats)
        print(f"[Debug] df after QB stats: {len(df)} records")

        # Merge previous season stats
        print(f"[Debug] Calculating previous season stats...")
        # Calculate season totals from weekly_stats (which includes pbp-calculated receptions/targets)
        season_stats_agg = {
            "rushing_yards": "sum",
            "rushing_tds": "sum",
            "receiving_yards": "sum",
            "receiving_tds": "sum",
        }
        # Add receptions and targets from weekly_stats
        if "receptions" in weekly_stats.columns:
            season_stats_agg["receptions"] = "sum"
        if "targets" in weekly_stats.columns:
            season_stats_agg["targets"] = "sum"
            
        season_stats = (
            weekly_stats.groupby(["player_id", "season"])
            .agg(season_stats_agg)
            .reset_index()
        )
        
        # Calculate season totals for red zone stats from main df (for accurate completion %)
        if "red_zone_receptions" in df.columns and "red_zone_targets" in df.columns:
            rz_season_stats = df.groupby(["player_id", "season"]).agg({
                "red_zone_receptions": "sum",
                "red_zone_targets": "sum"
            }).reset_index()
            # Calculate season red zone completion %
            rz_season_stats["red_zone_completion_pct"] = np.where(
                rz_season_stats["red_zone_targets"] > 0,
                rz_season_stats["red_zone_receptions"] / rz_season_stats["red_zone_targets"],
                0.0
            )
            # Merge with season_stats - rename to avoid conflict with per-game red_zone_completion_pct
            season_stats = pd.merge(
                season_stats,
                rz_season_stats[["player_id", "season", "red_zone_completion_pct"]].rename(
                    columns={"red_zone_completion_pct": "season_rz_completion_pct"}
                ),
                on=["player_id", "season"],
                how="left"
            )
            season_stats["season_rz_completion_pct"] = season_stats["season_rz_completion_pct"].fillna(0.0)

        df = pd.merge(df, season_stats, how="left", on=["player_id", "season"])
        df = df.sort_values(by=["player_id", "season"])
        
        # Calculate prev_red_zone_completion_pct from season aggregated value
        if "season_rz_completion_pct" in df.columns:
            df["prev_red_zone_completion_pct"] = df.groupby("player_id")["season_rz_completion_pct"].shift(1).fillna(0.0)
            df = df.drop("season_rz_completion_pct", axis=1, errors="ignore")
        else:
            # Fallback: calculate from per-game value if season aggregate not available
            df["prev_red_zone_completion_pct"] = df.groupby("player_id")["red_zone_completion_pct"].shift(1).fillna(0.0)
        
        df = df.groupby(["player_id"], group_keys=False).apply(calculate_prev)
        print(f"[Debug] df after prev stats: {len(df)} records")

        df = df.drop(
            ["rushing_yards", "rushing_tds", "receiving_yards", "receiving_tds", "receptions", "targets"],
            axis=1,
            errors="ignore",
        )

        # Clean data
        df = df.drop_duplicates()
        print(f"[Debug] df after drop_duplicates: {len(df)} records")

        df.loc[~df["report_status"].isin(REPORT_STATUS_ORDER), "report_status"] = (
            "Minor"
        )
        df = df[~(df.report_status == "Out")]
        print(f"[Debug] df after removing 'Out' status: {len(df)} records")

        # Merge red zone stats
        print(f"[Debug] Merging red zone stats...")
        if self.rz_stats_rolling is not None and self.rz_stats_prev is not None:
            red_zone_df = self.rz_stats_rolling[
                ["posteam", "season", "week", "rolling_red_zone"]
            ]
            prev_year_rz = self.rz_stats_prev[["posteam", "next_year", "prev_red_zone"]]

            df = pd.merge(
                df,
                red_zone_df,
                how="left",
                left_on=["against", "season", "week"],
                right_on=["posteam", "season", "week"],
            )
            df = df.drop(["posteam"], axis=1, errors="ignore")

            df = pd.merge(
                df,
                prev_year_rz,
                how="left",
                left_on=["against", "season"],
                right_on=["posteam", "next_year"],
            )
            df = df.drop(["posteam", "next_year"], axis=1, errors="ignore")
        else:
            print("[Warning] Red zone stats not available, skipping...")
            df["rolling_red_zone"] = 0
            df["prev_red_zone"] = 0

        print(f"[Debug] df after red zone merge: {len(df)} records")

        # Zero out QB stats for QB position
        df.loc[
            df["position"] == "QB",
            ["qb_rolling_passing_yards", "qb_rolling_passing_tds"],
        ] = 0

        # Add played indicator
        print(f"[Debug] Adding played indicator...")
        played = weekly_stats[["player_id", "season", "week"]].drop_duplicates()
        played["played"] = 1
        df = pd.merge(df, played, how="left", on=["player_id", "season", "week"])
        df["played"] = df["played"].fillna(0).astype(int)
        print(f"[Debug] df after played merge: {len(df)} records")
        print(f"[Debug] Played distribution: {df['played'].value_counts().to_dict()}")

        print(f"[Debug] FINAL df: {len(df)} records")
        return df
