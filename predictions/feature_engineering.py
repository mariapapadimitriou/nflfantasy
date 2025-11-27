"""
Comprehensive Feature Engineering for NFL Touchdown Prediction Model
Implements all features from the feature blueprint
"""
import logging
import pandas as pd
import numpy as np
from .config import EWMA_ALPHA, WINSORIZE_ENABLED, WINSORIZE_PERCENTILE


logger = logging.getLogger(__name__)


def winsorize_values(series, percentile=WINSORIZE_PERCENTILE):
    """
    Winsorize (cap) extreme values to prevent outliers from skewing EWMA.
    
    Args:
        series: pandas Series with values to winsorize
        percentile: Percentile to cap at (0.95 = cap at 95th percentile)
    
    Returns:
        Series with extreme values capped
    """
    if not WINSORIZE_ENABLED or len(series) == 0:
        return series
    
    # Only winsorize if we have enough non-null values
    valid_values = series.dropna()
    if len(valid_values) < 10:  # Need at least 10 values for meaningful percentile
        return series
    
    # Calculate cap threshold
    cap_value = valid_values.quantile(percentile)
    
    # Cap values at the threshold
    capped_series = series.copy()
    capped_series = capped_series.clip(upper=cap_value)
    
    return capped_series


def calculate_ewma_feature(series, alpha=EWMA_ALPHA, winsorize=True):
    """
    Calculate EWMA for a single feature with shift.
    
    Args:
        series: pandas Series with values to calculate EWMA on
        alpha: EWMA decay factor
        winsorize: Whether to winsorize values before calculating EWMA
    
    Returns:
        Series with EWMA values (shifted by 1 to avoid lookahead bias)
    """
    if winsorize and WINSORIZE_ENABLED:
        series = winsorize_values(series, percentile=WINSORIZE_PERCENTILE)
    
    return series.ewm(alpha=alpha, adjust=False).mean().shift(1)


def calculate_player_features(player_stats, pbp, weekly_stats):
    """
    Calculate all player-level features from play-by-play and player stats data
    
    Args:
        player_stats: DataFrame with player statistics
        pbp: DataFrame with play-by-play data
        weekly_stats: DataFrame with weekly aggregated stats (must include player_id, season, week)
        
    Returns:
        DataFrame with player features per game (includes player_id, season, week, game_id if available)
    """
    from .config import EWMA_ALPHA
    
    # Start with weekly stats as base
    player_features = weekly_stats.copy()
    
    # Ensure we have game_id if available in pbp
    if "game_id" in pbp.columns and "player_id" in pbp.columns:
        # Try to get game_id from pbp if not in weekly_stats
        if "game_id" not in player_features.columns:
            # Get game_id from pbp for players
            pbp_game_info = pbp[["player_id", "game_id", "season", "week"]].drop_duplicates()
            if len(pbp_game_info) > 0:
                player_features = pd.merge(
                    player_features,
                    pbp_game_info,
                    on=["player_id", "season", "week"],
                    how="left"
                )
    
    # Ensure sorted by player, season, week
    player_features = player_features.sort_values(by=["player_id", "season", "week"]).reset_index(drop=True)
    
    # Initialize red_zone_touches column for EWMA calculations
    player_features["red_zone_touches"] = 0
    
    # Forward-fill targets and receptions to future weeks before calculating EWMA
    # This ensures future weeks have values for EWMA calculation
    if "targets" in player_features.columns:
        player_features = player_features.sort_values(by=["player_id", "season", "week"])
        player_features["targets"] = player_features.groupby("player_id")["targets"].ffill()
    if "receptions" in player_features.columns:
        player_features = player_features.sort_values(by=["player_id", "season", "week"])
        player_features["receptions"] = player_features.groupby("player_id")["receptions"].ffill()
    
    # Ensure carries and targets exist before calculating touches
    if "carries" not in player_features.columns:
        player_features["carries"] = 0
    if "targets" not in player_features.columns:
        player_features["targets"] = 0
    
    # Calculate touches - preserve NaN if either component is NaN
    player_features["touches"] = (
        player_features["carries"].fillna(0) + player_features["targets"].fillna(0)
    )
    # Replace sum of 0s back to NaN if both were originally NaN
    player_features["touches"] = player_features["touches"].replace(0, np.nan).where(
        (player_features["carries"].notna()) | (player_features["targets"].notna()),
        np.nan
    )
    # Ensure touches column exists (set to 0 if somehow missing)
    if "touches" not in player_features.columns:
        player_features["touches"] = 0
    
    # Group by player to calculate EWMA features (after touches is created)
    player_groups = player_features.groupby("player_id", group_keys=False)
    
    # Basic counts (EWMA)
    if "targets" in player_features.columns:
        player_features["targets_ewma"] = player_groups["targets"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
        # Forward-fill targets_ewma to future weeks (carries last known EWMA forward)
        player_features = player_features.sort_values(by=["player_id", "season", "week"])
        player_features["targets_ewma"] = player_features.groupby("player_id")["targets_ewma"].ffill()
    else:
        player_features["targets_ewma"] = 0
    
    if "receptions" in player_features.columns:
        player_features["receptions_ewma"] = player_groups["receptions"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
        # Forward-fill receptions_ewma to future weeks (carries last known EWMA forward)
        player_features = player_features.sort_values(by=["player_id", "season", "week"])
        player_features["receptions_ewma"] = player_features.groupby("player_id")["receptions_ewma"].ffill()
    else:
        player_features["receptions_ewma"] = 0
    
    # Carries (rushing attempts)
    if "carries" in player_features.columns:
        player_features["carries_ewma"] = player_groups["carries"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    else:
        # Try to get from pbp if available
        if "rusher_player_id" in pbp.columns and "rush" in pbp.columns:
            carries_data = pbp[pbp["rush"] == 1].groupby(["rusher_player_id", "game_id", "season", "week"]).size().reset_index(name="carries")
            carries_data = carries_data.rename(columns={"rusher_player_id": "player_id"})
            player_features = pd.merge(
                player_features,
                carries_data,
                on=["player_id", "game_id", "season", "week"],
                how="left"
            )
            # Recreate player_groups after adding carries
            player_groups = player_features.groupby("player_id", group_keys=False)
            # Don't fillna - preserve NaN for players without carries
            player_features["carries_ewma"] = player_groups["carries"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
        else:
            player_features["carries_ewma"] = np.nan
    
    # Calculate touches_ewma (touches should already exist)
    if "touches" in player_features.columns:
        # Recreate player_groups to ensure it has the latest columns
        player_groups = player_features.groupby("player_id", group_keys=False)
        player_features["touches_ewma"] = player_groups["touches"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    else:
        # Fallback if touches somehow doesn't exist
        player_features["touches_ewma"] = np.nan
    
    # Red zone touches (from play-by-play)
    if "yardline_100" in pbp.columns:
        rz_touches_list = []
        
        # Red zone targets
        if "receiver_player_id" in pbp.columns and "pass" in pbp.columns:
            rz_targets = pbp[
                (pbp["yardline_100"] <= 20) & 
                (pbp["pass"] == 1) & 
                (pbp["receiver_player_id"].notna())
            ].groupby(["receiver_player_id", "game_id", "season", "week"]).size().reset_index(name="rz_targets")
            rz_targets = rz_targets.rename(columns={"receiver_player_id": "player_id"})
            rz_touches_list.append(rz_targets)
        
        # Red zone carries
        if "rusher_player_id" in pbp.columns and "rush" in pbp.columns:
            rz_carries = pbp[
                (pbp["yardline_100"] <= 20) & 
                (pbp["rush"] == 1) & 
                (pbp["rusher_player_id"].notna())
            ].groupby(["rusher_player_id", "game_id", "season", "week"]).size().reset_index(name="rz_carries")
            rz_carries = rz_carries.rename(columns={"rusher_player_id": "player_id"})
            rz_touches_list.append(rz_carries)
        
        if rz_touches_list:
            # Merge all red zone touches data
            rz_touches = rz_touches_list[0]
            for df_rz in rz_touches_list[1:]:
                rz_touches = pd.merge(
                    rz_touches,
                    df_rz,
                    on=["player_id", "game_id", "season", "week"],
                    how="outer"
                )
            rz_touches = rz_touches.fillna(0)
            if "rz_targets" not in rz_touches.columns:
                rz_touches["rz_targets"] = 0
            if "rz_carries" not in rz_touches.columns:
                rz_touches["rz_carries"] = 0
            rz_touches["red_zone_touches"] = rz_touches["rz_targets"] + rz_touches["rz_carries"]
            
            # Merge using a temporary column name to avoid conflicts
            rz_touches_merge = rz_touches[["player_id", "game_id", "season", "week", "red_zone_touches"]].rename(
                columns={"red_zone_touches": "red_zone_touches_from_pbp"}
            )
            player_features = pd.merge(
                player_features,
                rz_touches_merge,
                on=["player_id", "game_id", "season", "week"],
                how="left"
            )
            # Update red_zone_touches with merged values
            if "red_zone_touches_from_pbp" in player_features.columns:
                player_features["red_zone_touches"] = player_features["red_zone_touches_from_pbp"].fillna(player_features["red_zone_touches"])
                player_features = player_features.drop("red_zone_touches_from_pbp", axis=1, errors="ignore")
            # Ensure red_zone_touches exists
            if "red_zone_touches" not in player_features.columns:
                player_features["red_zone_touches"] = 0
    
    # Calculate EWMA for red zone touches - ensure it exists first
    if "red_zone_touches" not in player_features.columns:
        player_features["red_zone_touches"] = np.nan
    
    # Recreate player_groups to ensure it has the latest columns
    player_groups = player_features.groupby("player_id", group_keys=False)
    
    player_features["red_zone_touches_ewma"] = player_groups["red_zone_touches"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    
    # Receiving stats (EWMA)
    if "receiving_yards" in player_features.columns:
        player_features["receiving_yards_ewma"] = player_groups["receiving_yards"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    else:
        player_features["receiving_yards_ewma"] = np.nan
    
    if "receiving_tds" in player_features.columns:
        player_features["receiving_touchdowns_ewma"] = player_groups["receiving_tds"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    else:
        player_features["receiving_touchdowns_ewma"] = np.nan
    
    # Rushing stats (EWMA)
    if "rushing_yards" in player_features.columns:
        player_features["rushing_yards_ewma"] = player_groups["rushing_yards"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    else:
        player_features["rushing_yards_ewma"] = np.nan
    
    if "rushing_tds" in player_features.columns:
        player_features["rushing_touchdowns_ewma"] = player_groups["rushing_tds"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    else:
        player_features["rushing_touchdowns_ewma"] = np.nan
    
    # Add breakout game indicators (captures recent surge in performance)
    # Includes both split (rushing/receiving) and combined totals for comparison
    # Combined totals will be normalized by position later in data_manager
    from .config import BREAKOUT_ENABLED
    if BREAKOUT_ENABLED:
        from .config import BREAKOUT_TOTAL_TDS, BREAKOUT_TOTAL_YARDS
        
        # Sort by player, season, week to identify "last game"
        player_features = player_features.sort_values(by=["player_id", "season", "week"]).reset_index(drop=True)
        player_groups = player_features.groupby("player_id", group_keys=False)
        
        # Calculate total touchdowns and total yards for combined breakout detection
        if "receiving_tds" in player_features.columns and "rushing_tds" in player_features.columns:
            player_features["total_tds"] = (
                player_features["receiving_tds"].fillna(0) + player_features["rushing_tds"].fillna(0)
            )
        elif "receiving_tds" in player_features.columns:
            player_features["total_tds"] = player_features["receiving_tds"].fillna(0)
        elif "rushing_tds" in player_features.columns:
            player_features["total_tds"] = player_features["rushing_tds"].fillna(0)
        else:
            player_features["total_tds"] = 0
        
        if "receiving_yards" in player_features.columns and "rushing_yards" in player_features.columns:
            player_features["total_yards"] = (
                player_features["receiving_yards"].fillna(0) + player_features["rushing_yards"].fillna(0)
            )
        elif "receiving_yards" in player_features.columns:
            player_features["total_yards"] = player_features["receiving_yards"].fillna(0)
        elif "rushing_yards" in player_features.columns:
            player_features["total_yards"] = player_features["rushing_yards"].fillna(0)
        else:
            player_features["total_yards"] = 0
        
        # Recreate player_groups after adding total columns
        player_groups = player_features.groupby("player_id", group_keys=False)
        
        # Total TD breakout (raw value, to be normalized later)
        player_features["recent_total_breakout_tds"] = player_groups["total_tds"].apply(
            lambda x: x.shift(1).where(x.shift(1) >= BREAKOUT_TOTAL_TDS, 0)
        )
        
        # Total yards breakout (raw value, to be normalized later)
        player_features["recent_total_breakout_yards"] = player_groups["total_yards"].apply(
            lambda x: x.shift(1).where(x.shift(1) >= BREAKOUT_TOTAL_YARDS, 0)
        )
        
        # Combined breakout indicator: 1 if any breakout occurred in last game
        player_features["recent_breakout_game"] = (
            (player_features["recent_total_breakout_tds"] > 0) | 
            (player_features["recent_total_breakout_yards"] > 0)
        ).astype(int)
        
        # Clean up temporary columns
        player_features = player_features.drop(columns=["total_tds", "total_yards"], errors="ignore")
    
    return player_features


def calculate_team_context_features(player_features, pbp, schedules):
    """
    Calculate team-level context features
    
    Args:
        player_features: DataFrame with player features (must include 'team' column)
        pbp: DataFrame with play-by-play data
        schedules: DataFrame with schedule/game data
        
    Returns:
        DataFrame with team context features per game
    """
    # Ensure team column exists
    if "team" not in player_features.columns:
        raise ValueError("'team' column is required in player_features for team context calculations")
    
    # Team play volume (total offensive plays)
    if "posteam" in pbp.columns:
        team_plays = pbp.groupby(["posteam", "game_id", "season", "week"]).size().reset_index(name="total_plays")
        team_plays = team_plays.rename(columns={"posteam": "team"})
        logger.warning("Calculated team_play_volume from pbp.posteam: %s records", len(team_plays))
    elif "play_type" in pbp.columns or "down" in pbp.columns:
        # Try alternative grouping
        team_plays = pbp.groupby(["posteam", "game_id", "season", "week"]).size().reset_index(name="total_plays")
        team_plays = team_plays.rename(columns={"posteam": "team"})
        logger.warning("Calculated team_play_volume from pbp (alternative): %s records", len(team_plays))
    else:
        logger.warning("Cannot calculate team_play_volume: missing 'posteam', 'play_type', and 'down' columns in pbp. Available columns: %s", list(pbp.columns)[:20])
        # Fallback: use player touches as proxy
        # Ensure touches exists - calculate it if needed
        if "touches" not in player_features.columns:
            # Calculate touches from carries and targets if available
            if "carries" not in player_features.columns:
                player_features["carries"] = 0
            if "targets" not in player_features.columns:
                player_features["targets"] = 0
            player_features["touches"] = (
                player_features["carries"].fillna(0) + player_features["targets"].fillna(0)
            )
            # If both carries and targets are NaN/0, set touches to 0
            player_features["touches"] = player_features["touches"].fillna(0)
        
        # Ensure touches column exists before grouping
        if "touches" in player_features.columns:
            team_plays = player_features.groupby(["team", "game_id", "season", "week"])["touches"].sum().reset_index(name="total_plays")
            logger.warning("Using player touches as fallback for team_play_volume: %s records", len(team_plays))
        else:
            # Last resort: create empty dataframe with required columns
            logger.warning("Cannot calculate team_play_volume: 'touches' column not available. Creating empty team_plays.")
            team_plays = pd.DataFrame(columns=["team", "game_id", "season", "week", "total_plays"])
    
    # EWMA of team play volume
    team_plays = team_plays.sort_values(by=["team", "season", "week"])
    team_plays["team_play_volume_ewma"] = (
        team_plays.groupby("team")["total_plays"]
        .apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    )
    
    # Forward-fill team_play_volume_ewma within each team to handle early weeks
    team_plays = team_plays.sort_values(by=["team", "season", "week"])
    team_plays["team_play_volume_ewma"] = team_plays.groupby("team")["team_play_volume_ewma"].ffill()
    # Fill any remaining NaN with league median
    if team_plays["team_play_volume_ewma"].isna().any():
        league_median = team_plays["team_play_volume_ewma"].median()
        if pd.notna(league_median):
            team_plays["team_play_volume_ewma"] = team_plays["team_play_volume_ewma"].fillna(league_median)
        else:
            team_plays["team_play_volume_ewma"] = team_plays["team_play_volume_ewma"].fillna(65.0)  # Average plays per game
    
    # Team total red zone touches
    # Ensure red_zone_touches exists in player_features
    if "red_zone_touches" not in player_features.columns:
        player_features["red_zone_touches"] = 0
    
    team_rz_touches = player_features.groupby(["team", "game_id", "season", "week"])["red_zone_touches"].sum().reset_index(name="team_total_red_zone_touches")
    team_rz_touches = team_rz_touches.sort_values(by=["team", "season", "week"])
    team_rz_touches["team_total_red_zone_touches_ewma"] = (
        team_rz_touches.groupby("team")["team_total_red_zone_touches"]
        .apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    )
    
    # Forward-fill team_total_red_zone_touches_ewma within each team
    team_rz_touches = team_rz_touches.sort_values(by=["team", "season", "week"])
    team_rz_touches["team_total_red_zone_touches_ewma"] = team_rz_touches.groupby("team")["team_total_red_zone_touches_ewma"].ffill()
    # Fill any remaining NaN with league median
    if team_rz_touches["team_total_red_zone_touches_ewma"].isna().any():
        league_median = team_rz_touches["team_total_red_zone_touches_ewma"].median()
        if pd.notna(league_median):
            team_rz_touches["team_total_red_zone_touches_ewma"] = team_rz_touches["team_total_red_zone_touches_ewma"].fillna(league_median)
        else:
            team_rz_touches["team_total_red_zone_touches_ewma"] = team_rz_touches["team_total_red_zone_touches_ewma"].fillna(0.0)
    
    # Merge team context
    team_context = pd.merge(
        team_plays[["team", "game_id", "season", "week", "team_play_volume_ewma"]],
        team_rz_touches[["team", "game_id", "season", "week", "team_total_red_zone_touches_ewma"]],
        on=["team", "game_id", "season", "week"],
        how="outer"
    )
    
    # Extend team context to future weeks using schedules
    if len(schedules) > 0 and "game_id" in schedules.columns:
        # Get all games from schedules (including future weeks)
        all_games = schedules[["game_id", "season", "week", "home_team", "away_team"]].copy()
        
        # Create entries for both home and away teams
        home_games = all_games[["game_id", "season", "week", "home_team"]].copy()
        home_games = home_games.rename(columns={"home_team": "team"})
        
        away_games = all_games[["game_id", "season", "week", "away_team"]].copy()
        away_games = away_games.rename(columns={"away_team": "team"})
        
        all_team_games = pd.concat([home_games, away_games]).drop_duplicates()
        
        # Merge with existing team context
        all_team_games = pd.merge(
            all_team_games,
            team_context[["team", "game_id", "season", "week", "team_play_volume_ewma", "team_total_red_zone_touches_ewma"]],
            on=["team", "game_id", "season", "week"],
            how="left"
        )
        
        # Sort by team, season, week for forward-filling
        all_team_games = all_team_games.sort_values(by=["team", "season", "week"])
        
        # Forward-fill team context features to future weeks
        for col in ["team_play_volume_ewma", "team_total_red_zone_touches_ewma"]:
            if col in all_team_games.columns:
                # Forward-fill within each team (carries last known value forward to future weeks)
                all_team_games[col] = all_team_games.groupby("team")[col].ffill()
                # For teams with no history, use league median
                if all_team_games[col].isna().any():
                    league_median = team_context[col].median() if len(team_context) > 0 else None
                    if pd.notna(league_median):
                        all_team_games[col] = all_team_games[col].fillna(league_median)
                    else:
                        # Use reasonable defaults
                        if "play_volume" in col:
                            all_team_games[col] = all_team_games[col].fillna(65.0)
                        else:
                            all_team_games[col] = all_team_games[col].fillna(0.0)
        
        # Count how many records we had before extension
        original_count = len(team_context)
        
        team_context = all_team_games[["team", "game_id", "season", "week", "team_play_volume_ewma", "team_total_red_zone_touches_ewma"]].copy()
        
        future_count = len(team_context) - original_count
        logger.warning("Extended team context features to future weeks: %s total records (added %s future games)", 
                      len(team_context), future_count)
    
    # Add win probability and spread from schedules
    if "home_moneyline" in schedules.columns:
        games = schedules[["game_id", "season", "week", "home_team", "away_team", "home_moneyline", "spread_line"]].copy()
        
        # Calculate win probability from moneyline
        from .utils import american_odds_to_probability
        games["home_wp"] = games["home_moneyline"].apply(
            lambda x: american_odds_to_probability(x) if pd.notnull(x) else 0.5
        )
        
        # Merge for home team
        home_context = games[["game_id", "season", "week", "home_team", "home_wp", "spread_line"]].copy()
        home_context = home_context.rename(columns={"home_team": "team", "home_wp": "team_win_probability"})
        
        # Merge for away team (inverse win prob)
        away_context = games[["game_id", "season", "week", "away_team", "home_wp", "spread_line"]].copy()
        away_context = away_context.rename(columns={"away_team": "team"})
        away_context["team_win_probability"] = 1 - away_context["home_wp"]
        away_context = away_context.drop(["home_wp"], axis=1)
        
        game_context = pd.concat([home_context, away_context])
        
        team_context = pd.merge(
            team_context,
            game_context[["team", "game_id", "season", "week", "team_win_probability", "spread_line"]],
            on=["team", "game_id", "season", "week"],
            how="left"
        )
    else:
        team_context["team_win_probability"] = 0.5
        team_context["spread_line"] = 0
    
    # Don't fillna - preserve NaN for missing team context data
    # Only fill critical columns that must have values (spread_line, team_win_probability default to 0.5/0)
    if "team_win_probability" in team_context.columns:
        team_context["team_win_probability"] = team_context["team_win_probability"].fillna(0.5)
    if "spread_line" in team_context.columns:
        team_context["spread_line"] = team_context["spread_line"].fillna(0)
    
    return team_context


def calculate_team_shares(player_features, team_context):
    """
    Calculate team-relative shares for player features
    
    Args:
        player_features: DataFrame with player features
        team_context: DataFrame with team context features
        
    Returns:
        DataFrame with share features added
    """
    # Merge team context - ensure required columns exist in team_context
    # Drop any existing columns from player_features to avoid merge conflicts
    cols_to_drop = ["team_total_red_zone_touches_ewma"]
    for col in cols_to_drop:
        if col in player_features.columns:
            player_features = player_features.drop(columns=[col])
    
    required_cols = ["team", "game_id", "season", "week"]
    if "team_total_red_zone_touches_ewma" in team_context.columns:
        required_cols.append("team_total_red_zone_touches_ewma")
    
    df = pd.merge(
        player_features,
        team_context[required_cols],
        on=["team", "game_id", "season", "week"],
        how="left"
    )
    
    # Clean up any duplicate columns with suffixes from merge
    for col in df.columns:
        if col.endswith('_x') or col.endswith('_y'):
            base_col = col[:-2]  # Remove _x or _y suffix
            if base_col in required_cols:
                # Keep the merged value (usually _x), drop the other
                if col.endswith('_y'):
                    df = df.drop(columns=[col])
                else:
                    # Rename _x back to original name
                    df = df.rename(columns={col: base_col})
    
    # Red zone touch share
    # Ensure red_zone_touches_ewma exists
    if "red_zone_touches_ewma" not in df.columns:
        df["red_zone_touches_ewma"] = 0
    if "team_total_red_zone_touches_ewma" not in df.columns:
        df["team_total_red_zone_touches_ewma"] = 0
    
    # Calculate share - preserve NaN if player has no red_zone_touches_ewma
    df["red_zone_touch_share_ewma"] = (
        df["red_zone_touches_ewma"] / (df["team_total_red_zone_touches_ewma"] + 1)
    )
    # Set to NaN if player has no red_zone_touches (not 0)
    df.loc[df["red_zone_touches_ewma"].isna() | (df["red_zone_touches_ewma"] == 0), "red_zone_touch_share_ewma"] = np.nan
    
    return df


def calculate_defensive_features(pbp, schedules):
    """
    Calculate defensive context features (opponent defense stats)
    
    Args:
        pbp: DataFrame with play-by-play data
        schedules: DataFrame with schedule/game data
        
    Returns:
        DataFrame with defensive features per team per game
    """
    # Calculate defensive stats from opponent's perspective
    if "defteam" not in pbp.columns:
        # Return empty dataframe with required columns
        return pd.DataFrame(columns=["team", "game_id", "season", "week", 
                                     "def_ewma_yards_allowed_per_game", "def_ewma_TDs_allowed_per_game",
                                     "def_ewma_red_zone_completion_pct_allowed", "def_ewma_interceptions_per_game",
                                     "opponent_red_zone_def_rank"])
    
    defensive_stats = pbp.groupby(["defteam", "game_id", "season", "week"]).agg({
        "yards_gained": "sum",
        "touchdown": "sum",
        "interception": "sum" if "interception" in pbp.columns else lambda x: 0,
    }).reset_index()
    
    defensive_stats = defensive_stats.rename(columns={"defteam": "team"})
    logger.warning("Calculated defensive stats: %s records", len(defensive_stats))
    
    # Calculate red zone completion % allowed
    if "yardline_100" in pbp.columns and "complete_pass" in pbp.columns:
        rz_passes = pbp[
            (pbp["yardline_100"] <= 20) & 
            (pbp["pass"] == 1)
        ].groupby(["defteam", "game_id", "season", "week"]).agg({
            "complete_pass": ["sum", "count"]
        }).reset_index()
        rz_passes.columns = ["team", "game_id", "season", "week", "rz_completions_allowed", "rz_attempts_allowed"]
        rz_passes["rz_completion_pct_allowed"] = (
            rz_passes["rz_completions_allowed"] / (rz_passes["rz_attempts_allowed"] + 1)
        ).fillna(0)
        
        defensive_stats = pd.merge(
            defensive_stats,
            rz_passes[["team", "game_id", "season", "week", "rz_completion_pct_allowed"]],
            on=["team", "game_id", "season", "week"],
            how="left"
        )
        # Don't fillna - preserve NaN if no red zone data
    else:
        defensive_stats["rz_completion_pct_allowed"] = np.nan
    
    # Calculate EWMA for defensive stats
    defensive_stats = defensive_stats.sort_values(by=["team", "season", "week"])
    
    defensive_stats["def_ewma_yards_allowed_per_game"] = (
        defensive_stats.groupby("team")["yards_gained"]
        .apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    )
    
    defensive_stats["def_ewma_TDs_allowed_per_game"] = (
        defensive_stats.groupby("team")["touchdown"]
        .apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    )
    
    defensive_stats["def_ewma_red_zone_completion_pct_allowed"] = (
        defensive_stats.groupby("team")["rz_completion_pct_allowed"]
        .apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    )
    
    if "interception" in pbp.columns:
        defensive_stats["def_ewma_interceptions_per_game"] = (
            defensive_stats.groupby("team")["interception"]
            .apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
        )
    else:
        defensive_stats["def_ewma_interceptions_per_game"] = np.nan
    
    # Forward-fill defensive EWMA features within each team to handle early weeks
    # This ensures we have values even for week 1 (uses league average or forward-fills from previous season)
    defensive_stats = defensive_stats.sort_values(by=["team", "season", "week"])
    
    # Forward-fill within each team to propagate values to early weeks
    for col in ["def_ewma_yards_allowed_per_game", "def_ewma_TDs_allowed_per_game", 
                "def_ewma_red_zone_completion_pct_allowed", "def_ewma_interceptions_per_game"]:
        if col in defensive_stats.columns:
            # Forward-fill within each team (carries last known value forward)
            defensive_stats[col] = defensive_stats.groupby("team")[col].ffill()
            # For teams with no history, use league average (median to avoid outliers)
            if defensive_stats[col].isna().any():
                league_median = defensive_stats[col].median()
                if pd.notna(league_median):
                    defensive_stats[col] = defensive_stats[col].fillna(league_median)
                else:
                    # If still NaN, use reasonable defaults
                    if "yards" in col:
                        defensive_stats[col] = defensive_stats[col].fillna(350.0)  # Average yards allowed
                    elif "TDs" in col:
                        defensive_stats[col] = defensive_stats[col].fillna(2.0)  # Average TDs allowed
                    elif "interception" in col:
                        defensive_stats[col] = defensive_stats[col].fillna(1.0)  # Average INTs
                    else:
                        defensive_stats[col] = defensive_stats[col].fillna(0.5)  # Default for percentages
    
    # Opponent red zone defense rank (simplified - rank by rz_completion_pct_allowed EWMA)
    defensive_stats = defensive_stats.sort_values(by=["season", "week", "def_ewma_red_zone_completion_pct_allowed"], ascending=[True, True, False])
    defensive_stats["opponent_red_zone_def_rank"] = (
        defensive_stats.groupby(["season", "week"])["def_ewma_red_zone_completion_pct_allowed"]
        .rank(method="min", ascending=False)
    ).fillna(16)  # Default to middle rank (16 out of 32) instead of worst (32)
    
    # Extend defensive features to future weeks using schedules
    # For games that haven't been played yet, use the team's most recent defensive stats
    if len(schedules) > 0 and "game_id" in schedules.columns:
        # Get all games from schedules (including future weeks)
        all_games = schedules[["game_id", "season", "week", "home_team", "away_team"]].copy()
        
        # Create entries for both home and away teams
        home_games = all_games[["game_id", "season", "week", "home_team"]].copy()
        home_games = home_games.rename(columns={"home_team": "team"})
        
        away_games = all_games[["game_id", "season", "week", "away_team"]].copy()
        away_games = away_games.rename(columns={"away_team": "team"})
        
        all_team_games = pd.concat([home_games, away_games]).drop_duplicates()
        
        # Merge with existing defensive stats
        all_team_games = pd.merge(
            all_team_games,
            defensive_stats[["team", "game_id", "season", "week", 
                             "def_ewma_yards_allowed_per_game", "def_ewma_TDs_allowed_per_game",
                             "def_ewma_red_zone_completion_pct_allowed", "def_ewma_interceptions_per_game",
                             "opponent_red_zone_def_rank"]],
            on=["team", "game_id", "season", "week"],
            how="left"
        )
        
        # Sort by team, season, week for forward-filling
        all_team_games = all_team_games.sort_values(by=["team", "season", "week"])
        
        # Forward-fill defensive features within each team to future weeks
        for col in ["def_ewma_yards_allowed_per_game", "def_ewma_TDs_allowed_per_game", 
                    "def_ewma_red_zone_completion_pct_allowed", "def_ewma_interceptions_per_game"]:
            if col in all_team_games.columns:
                # Forward-fill within each team (carries last known value forward to future weeks)
                all_team_games[col] = all_team_games.groupby("team")[col].ffill()
                # For teams with no history, use league median
                if all_team_games[col].isna().any():
                    league_median = defensive_stats[col].median() if len(defensive_stats) > 0 else None
                    if pd.notna(league_median):
                        all_team_games[col] = all_team_games[col].fillna(league_median)
                    else:
                        # Use reasonable defaults
                        if "yards" in col:
                            all_team_games[col] = all_team_games[col].fillna(350.0)
                        elif "TDs" in col:
                            all_team_games[col] = all_team_games[col].fillna(2.0)
                        elif "interception" in col:
                            all_team_games[col] = all_team_games[col].fillna(1.0)
                        else:
                            all_team_games[col] = all_team_games[col].fillna(0.5)
        
        # Recalculate opponent_red_zone_def_rank for all weeks (including future)
        # For each week, calculate rank based on forward-filled stats
        all_team_games = all_team_games.sort_values(by=["season", "week", "def_ewma_red_zone_completion_pct_allowed"], ascending=[True, True, False])
        
        # For each week, calculate rank
        for (season, week), group in all_team_games.groupby(["season", "week"]):
            if group["def_ewma_red_zone_completion_pct_allowed"].notna().any():
                # Calculate rank for this week based on forward-filled stats
                group_rank = group["def_ewma_red_zone_completion_pct_allowed"].rank(method="min", ascending=False)
                all_team_games.loc[group.index, "opponent_red_zone_def_rank"] = group_rank
            else:
                # If no data for this week, forward-fill the rank from the most recent week
                all_team_games.loc[group.index, "opponent_red_zone_def_rank"] = np.nan
        
        # Forward-fill opponent_red_zone_def_rank within each team to future weeks
        all_team_games = all_team_games.sort_values(by=["team", "season", "week"])
        all_team_games["opponent_red_zone_def_rank"] = all_team_games.groupby("team")["opponent_red_zone_def_rank"].ffill().fillna(16)
        
        defensive_stats = all_team_games[["team", "game_id", "season", "week", 
                                         "def_ewma_yards_allowed_per_game", "def_ewma_TDs_allowed_per_game",
                                         "def_ewma_red_zone_completion_pct_allowed", "def_ewma_interceptions_per_game",
                                         "opponent_red_zone_def_rank"]].copy()
        
        logger.warning("Extended defensive features to future weeks: %s total records (including %s future games)", 
                      len(defensive_stats), len(all_team_games) - len(defensive_stats))
    
    return defensive_stats[["team", "game_id", "season", "week", 
                           "def_ewma_yards_allowed_per_game", "def_ewma_TDs_allowed_per_game",
                           "def_ewma_red_zone_completion_pct_allowed", "def_ewma_interceptions_per_game",
                           "opponent_red_zone_def_rank"]]


def calculate_qb_stats(player_features, position_col=None):
    """
    Calculate QB-specific stats (passing and rushing) for QBs only,
    then these will be merged onto all players on the same team/game/week.
    
    Args:
        player_features: DataFrame with player features (must have passing/rushing stats)
        position_col: Series or column name indicating player position
        
    Returns:
        DataFrame with QB stats per game (team, game_id, season, week, qb stats)
        Ready to merge onto all players by team/game/week
    """
    from .config import EWMA_ALPHA
    
    # Identify QBs - need position column
    if position_col is None:
        if "position" in player_features.columns:
            position_col = player_features["position"]
        else:
            return pd.DataFrame()
    
    # Filter to QBs only
    qb_mask = position_col == "QB"
    if not qb_mask.any():
        return pd.DataFrame()
    
    qb_features = player_features[qb_mask].copy()
    
    # Ensure sorted by player, season, week for proper EWMA calculation
    qb_features = qb_features.sort_values(by=["player_id", "season", "week"]).reset_index(drop=True)
    
    # Group by player_id to calculate EWMA
    qb_groups = qb_features.groupby("player_id", group_keys=False)
    
    # Start with base columns
    qb_stats = qb_features[["player_id", "season", "week"]].copy()
    if "game_id" in qb_features.columns:
        qb_stats["game_id"] = qb_features["game_id"]
    if "team" in qb_features.columns:
        qb_stats["team"] = qb_features["team"]
    
    # QB passing stats - only yardage, not TD counts (to avoid spurious correlations)
    if "passing_yards" in qb_features.columns:
        qb_stats["qb_passing_yards_ewma"] = qb_groups["passing_yards"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    else:
        qb_stats["qb_passing_yards_ewma"] = np.nan
    
    # QB rushing stats - only yardage, not TD counts (to avoid spurious correlations)
    if "rushing_yards" in qb_features.columns:
        qb_stats["qb_rushing_yards_ewma"] = qb_groups["rushing_yards"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    else:
        qb_stats["qb_rushing_yards_ewma"] = np.nan
    
    # Note: QB TD stats removed (qb_passing_TDs_ewma, qb_rushing_TDs_ewma) to avoid spurious correlations
    # Especially qb_rushing_TDs_ewma = 0 was causing model to over-weight RBs
    
    # Return QB stats with player_id (which is the qb_id) for merging
    # Merge will be done by qb_id, season, week in data_manager
    return qb_stats[["player_id", "season", "week", "qb_passing_yards_ewma", 
                     "qb_rushing_yards_ewma"]].copy()
