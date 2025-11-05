"""
Comprehensive Feature Engineering for NFL Touchdown Prediction Model
Implements all features from the feature blueprint
"""
import pandas as pd
import numpy as np
from .config import EWMA_ALPHA, EWMA_WEEKS, REGRESSION_LAMBDA


def calculate_ewma_feature(series, alpha=EWMA_ALPHA):
    """Calculate EWMA for a single feature with shift"""
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
    
    # Initialize all intermediate columns that will be used for EWMA calculations
    # This ensures they exist before we try to calculate EWMA
    player_features["yards_after_catch"] = 0
    player_features["yards_after_contact"] = 0
    player_features["red_zone_touches"] = 0
    player_features["end_zone_targets"] = 0
    player_features["air_yards"] = 0
    player_features["aDOT"] = 0
    player_features["designed_rush_attempts"] = 0
    player_features["pass_attempts_inside10"] = 0
    
    # Group by player to calculate EWMA features (after initializing columns)
    player_groups = player_features.groupby("player_id", group_keys=False)
    
    # Basic counts (EWMA)
    if "targets" in player_features.columns:
        player_features["targets_ewma"] = player_groups["targets"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    else:
        player_features["targets_ewma"] = 0
    
    if "receptions" in player_features.columns:
        player_features["receptions_ewma"] = player_groups["receptions"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
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
            # Don't fillna - preserve NaN for players without carries
            player_features["carries_ewma"] = player_groups["carries"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
        else:
            player_features["carries_ewma"] = np.nan
    
    # Touches = carries + targets
    # Ensure carries exists
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
    player_features["touches_ewma"] = player_groups["touches"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    
    # Red zone touches (from play-by-play)
    # Note: red_zone_touches already initialized above
    
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
            # Ensure red_zone_touches exists (it should from initialization, but double-check)
            if "red_zone_touches" not in player_features.columns:
                player_features["red_zone_touches"] = 0
            # Don't fillna - preserve NaN for players without red zone touches
    
    # Calculate EWMA for red zone touches - ensure it exists first
    if "red_zone_touches" not in player_features.columns:
        player_features["red_zone_touches"] = np.nan
    
    # Recreate player_groups to ensure it has the latest columns
    player_groups = player_features.groupby("player_id", group_keys=False)
    
    player_features["red_zone_touches_ewma"] = player_groups["red_zone_touches"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    
    # Yards after catch - from play-by-play
    # Total yards gained after the catch on completed passes
    # Note: yards_after_catch already initialized above to 0
    
    if "yards_after_catch" in pbp.columns and "receiver_player_id" in pbp.columns:
        yac_data = pbp[
            (pbp["yards_after_catch"].notna()) & 
            (pbp["receiver_player_id"].notna())
        ].groupby(["receiver_player_id", "game_id", "season", "week"])["yards_after_catch"].sum().reset_index(name="yards_after_catch_sum")
        yac_data = yac_data.rename(columns={"receiver_player_id": "player_id"})
        
        # Merge with a different column name to avoid suffix conflicts
        player_features = pd.merge(
            player_features,
            yac_data,
            on=["player_id", "game_id", "season", "week"],
            how="left"
        )
        # Update yards_after_catch with merged values
        if "yards_after_catch_sum" in player_features.columns:
            player_features["yards_after_catch"] = player_features["yards_after_catch_sum"].fillna(player_features["yards_after_catch"])
            player_features = player_features.drop("yards_after_catch_sum", axis=1, errors="ignore")
        # Don't fillna - preserve NaN for players without yards_after_catch
    
    # Final safety check - ensure yards_after_catch exists before calculating EWMA
    if "yards_after_catch" not in player_features.columns:
        player_features["yards_after_catch"] = np.nan
    
    # Calculate EWMA for yards_after_catch - preserve NaN
    player_features["yards_after_catch_ewma"] = player_groups["yards_after_catch"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    
    # Yards after contact - may not be available, set to 0
    # Total yards gained after first contact on rushing attempts
    # Note: yards_after_contact already initialized above to 0
    
    if "yards_after_contact" in pbp.columns and "rusher_player_id" in pbp.columns:
        rac_data = pbp[
            (pbp["yards_after_contact"].notna()) & 
            (pbp["rusher_player_id"].notna())
        ].groupby(["rusher_player_id", "game_id", "season", "week"])["yards_after_contact"].sum().reset_index(name="yards_after_contact_sum")
        rac_data = rac_data.rename(columns={"rusher_player_id": "player_id"})
        
        # Merge with a different column name to avoid suffix conflicts
        player_features = pd.merge(
            player_features,
            rac_data,
            on=["player_id", "game_id", "season", "week"],
            how="left"
        )
        # Update yards_after_contact with merged values
        if "yards_after_contact_sum" in player_features.columns:
            player_features["yards_after_contact"] = player_features["yards_after_contact_sum"].fillna(player_features["yards_after_contact"])
            player_features = player_features.drop("yards_after_contact_sum", axis=1, errors="ignore")
        # Don't fillna - preserve NaN for players without yards_after_contact
    
    # Final safety check - ensure yards_after_contact exists before calculating EWMA
    if "yards_after_contact" not in player_features.columns:
        player_features["yards_after_contact"] = np.nan
    
    # Recreate player_groups to ensure it has the latest columns
    player_groups = player_features.groupby("player_id", group_keys=False)
    
    # Calculate EWMA for yards_after_contact - preserve NaN
    player_features["yards_after_contact_ewma"] = player_groups["yards_after_contact"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    
    # End zone targets (inside 10-yard line)
    # Note: end_zone_targets already initialized above
    
    if "yardline_100" in pbp.columns and "receiver_player_id" in pbp.columns and "pass" in pbp.columns:
        ez_targets = pbp[
            (pbp["yardline_100"] <= 10) & 
            (pbp["pass"] == 1) & 
            (pbp["receiver_player_id"].notna())
        ].groupby(["receiver_player_id", "game_id", "season", "week"]).size().reset_index(name="end_zone_targets")
        ez_targets = ez_targets.rename(columns={"receiver_player_id": "player_id"})
        # Merge with a different column name to avoid suffix conflicts
        ez_targets_merge = ez_targets.rename(columns={"end_zone_targets": "end_zone_targets_from_pbp"})
        player_features = pd.merge(
            player_features,
            ez_targets_merge,
            on=["player_id", "game_id", "season", "week"],
            how="left"
        )
        # Update end_zone_targets with merged values
        if "end_zone_targets_from_pbp" in player_features.columns:
            player_features["end_zone_targets"] = player_features["end_zone_targets_from_pbp"].fillna(player_features["end_zone_targets"])
            player_features = player_features.drop("end_zone_targets_from_pbp", axis=1, errors="ignore")
        # Ensure end_zone_targets exists
        if "end_zone_targets" not in player_features.columns:
            player_features["end_zone_targets"] = 0
        # Don't fillna - preserve NaN for players without end_zone_targets
    
    # Calculate EWMA for end_zone_targets - ensure it exists first
    if "end_zone_targets" not in player_features.columns:
        player_features["end_zone_targets"] = np.nan
    
    # Recreate player_groups to ensure it has the latest columns
    player_groups = player_features.groupby("player_id", group_keys=False)
    
    player_features["end_zone_targets_ewma"] = player_groups["end_zone_targets"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    
    # Air yards and aDOT (average depth of target)
    # Note: aDOT should count ALL targets (completed and incomplete), not just completions
    # air_yards and aDOT already initialized above
    
    if "air_yards" in pbp.columns and "receiver_player_id" in pbp.columns:
        # Filter for pass attempts (all targets, not just completions)
        # air_yards should be available for all pass attempts where receiver is targeted
        # If air_yards is null, it might mean the pass wasn't actually targeted or was a different play type
        air_yards_data = pbp[
            (pbp["air_yards"].notna()) & 
            (pbp["receiver_player_id"].notna()) &
            (pbp.get("pass", 0) == 1)  # Only pass plays
        ].groupby(["receiver_player_id", "game_id", "season", "week"]).agg({
            "air_yards": ["sum", "mean", "count"]
        }).reset_index()
        air_yards_data.columns = ["player_id", "game_id", "season", "week", "air_yards_total", "aDOT", "targets_for_adot"]
        # Rename columns before merge to avoid conflicts
        air_yards_data_merge = air_yards_data[["player_id", "game_id", "season", "week", "air_yards_total", "aDOT"]].rename(
            columns={"air_yards_total": "air_yards_from_pbp", "aDOT": "aDOT_from_pbp"}
        )
        player_features = pd.merge(
            player_features,
            air_yards_data_merge,
            on=["player_id", "game_id", "season", "week"],
            how="left"
        )
        # Update with merged values
        if "air_yards_from_pbp" in player_features.columns:
            player_features["air_yards"] = player_features["air_yards_from_pbp"].fillna(player_features["air_yards"])
            player_features = player_features.drop("air_yards_from_pbp", axis=1, errors="ignore")
        if "aDOT_from_pbp" in player_features.columns:
            player_features["aDOT"] = player_features["aDOT_from_pbp"].fillna(player_features["aDOT"])
            player_features = player_features.drop("aDOT_from_pbp", axis=1, errors="ignore")
        # Ensure columns exist
        if "air_yards" not in player_features.columns:
            player_features["air_yards"] = 0
        if "aDOT" not in player_features.columns:
            player_features["aDOT"] = 0
        # Don't fillna - preserve NaN for players without air_yards/aDOT
    
    # Calculate EWMA for air_yards and aDOT - ensure they exist first
    if "air_yards" not in player_features.columns:
        player_features["air_yards"] = np.nan
    if "aDOT" not in player_features.columns:
        player_features["aDOT"] = np.nan
    
    # Recreate player_groups to ensure it has the latest columns
    player_groups = player_features.groupby("player_id", group_keys=False)
    
    player_features["air_yards_ewma"] = player_groups["air_yards"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    player_features["aDOT_ewma"] = player_groups["aDOT"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    
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
    
    # QB-specific features
    if "passing_yards" in player_features.columns:
        player_features["qb_rolling_passing_yards_ewma"] = player_groups["passing_yards"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    else:
        player_features["qb_rolling_passing_yards_ewma"] = 0
    
    if "passing_tds" in player_features.columns:
        player_features["qb_rolling_passing_TDs_ewma"] = player_groups["passing_tds"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    else:
        player_features["qb_rolling_passing_TDs_ewma"] = 0
    
    if "rushing_yards" in player_features.columns:
        player_features["qb_rolling_rushing_yards_ewma"] = player_groups["rushing_yards"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    else:
        player_features["qb_rolling_rushing_yards_ewma"] = 0
    
    if "rushing_tds" in player_features.columns:
        player_features["qb_rolling_rushing_TDs_ewma"] = player_groups["rushing_tds"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    else:
        player_features["qb_rolling_rushing_TDs_ewma"] = 0
    
    # QB rushing attempts (designed rush attempts)
    # Note: designed_rush_attempts already initialized above
    
    if "rusher_player_id" in pbp.columns and "qb_kneel" in pbp.columns:
        qb_rush_attempts = pbp[
            (pbp["rush"] == 1) & 
            (pbp["qb_kneel"] == 0) & 
            (pbp["rusher_player_id"].notna())
        ].groupby(["rusher_player_id", "game_id", "season", "week"]).size().reset_index(name="designed_rush_attempts_from_pbp")
        qb_rush_attempts = qb_rush_attempts.rename(columns={"rusher_player_id": "player_id"})
        
        # Merge using a temporary column name to avoid conflicts
        player_features = pd.merge(
            player_features,
            qb_rush_attempts,
            on=["player_id", "game_id", "season", "week"],
            how="left"
        )
        # Update designed_rush_attempts with merged values
        if "designed_rush_attempts_from_pbp" in player_features.columns:
            player_features["designed_rush_attempts"] = player_features["designed_rush_attempts_from_pbp"].fillna(player_features["designed_rush_attempts"])
            player_features = player_features.drop("designed_rush_attempts_from_pbp", axis=1, errors="ignore")
        # Don't fillna - preserve NaN for players without designed_rush_attempts
    
    # Ensure designed_rush_attempts exists before calculating EWMA
    if "designed_rush_attempts" not in player_features.columns:
        player_features["designed_rush_attempts"] = np.nan
    
    # Recreate player_groups to ensure it has the latest columns
    player_groups = player_features.groupby("player_id", group_keys=False)
    
    player_features["designed_rush_attempts_ewma"] = player_groups["designed_rush_attempts"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    
    # Pass attempts inside 10 (QB stat)
    # Note: pass_attempts_inside10 already initialized above
    
    if "yardline_100" in pbp.columns and "passer_player_id" in pbp.columns and "pass" in pbp.columns:
        pass_inside10 = pbp[
            (pbp["yardline_100"] <= 10) & 
            (pbp["pass"] == 1) & 
            (pbp["passer_player_id"].notna())
        ].groupby(["passer_player_id", "game_id", "season", "week"]).size().reset_index(name="pass_attempts_inside10_from_pbp")
        pass_inside10 = pass_inside10.rename(columns={"passer_player_id": "player_id"})
        
        # Merge using a temporary column name to avoid conflicts
        player_features = pd.merge(
            player_features,
            pass_inside10,
            on=["player_id", "game_id", "season", "week"],
            how="left"
        )
        # Update pass_attempts_inside10 with merged values
        if "pass_attempts_inside10_from_pbp" in player_features.columns:
            player_features["pass_attempts_inside10"] = player_features["pass_attempts_inside10_from_pbp"].fillna(player_features["pass_attempts_inside10"])
            player_features = player_features.drop("pass_attempts_inside10_from_pbp", axis=1, errors="ignore")
        # Don't fillna - preserve NaN for players without pass_attempts_inside10
    
    # Ensure pass_attempts_inside10 exists before calculating EWMA
    if "pass_attempts_inside10" not in player_features.columns:
        player_features["pass_attempts_inside10"] = np.nan
    
    # Recreate player_groups to ensure it has the latest columns
    player_groups = player_features.groupby("player_id", group_keys=False)
    
    player_features["pass_attempts_inside10_ewma"] = player_groups["pass_attempts_inside10"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    
    # Export data before EWMA calculations for spot checking
    # This happens after all intermediate values are calculated but before final EWMA features
    try:
        import os
        from django.conf import settings
        base_dir = settings.BASE_DIR if hasattr(settings, 'BASE_DIR') else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pre_ewma_path = os.path.join(base_dir, 'pre_ewma_data.csv')
        player_features.to_csv(pre_ewma_path, index=False)
        print(f"[Export] Saved pre-EWMA data to {pre_ewma_path} with {len(player_features)} rows and {len(player_features.columns)} columns")
        print(f"[Export] Includes raw values: targets, receptions, carries, touches, yards_after_catch, yards_after_contact, red_zone_touches, etc.")
    except Exception as e:
        print(f"[Warning] Could not export pre-EWMA data: {e}")
    
    # TD streak factor (std deviation of TDs last 3 games)
    if "touchdown" in player_features.columns:
        # Recreate player_groups to ensure it has latest columns
        player_groups = player_features.groupby("player_id", group_keys=False)
        # Calculate rolling std deviation of touchdowns over last 3 games, shifted by 1
        td_streak = player_groups["touchdown"].apply(
            lambda x: x.rolling(window=EWMA_WEEKS, min_periods=1).std().shift(1)
        )
        # The result should be a Series aligned with player_features index
        # Convert to Series if needed and ensure proper alignment
        if isinstance(td_streak, pd.Series):
            player_features["td_streak_factor"] = td_streak.reindex(player_features.index, fill_value=0)
        else:
            # If it's a numpy array or other, convert to Series
            player_features["td_streak_factor"] = pd.Series(td_streak, index=player_features.index, dtype=float)
        # Don't fillna - preserve NaN for players without TD history
    else:
        player_features["td_streak_factor"] = np.nan
    
    # Regression TD factor: EWMA + λ(xTD - EWMA)
    # xTD = expected touchdowns based on historical conversion per opportunity
    # xTD = historical TD conversion rate * current opportunities (touches)
    if "touchdown" in player_features.columns and "touches" in player_features.columns:
        # Recreate player_groups to ensure it has latest columns
        player_groups = player_features.groupby("player_id", group_keys=False)
        
        # Calculate EWMA of touchdowns
        td_ewma = player_groups["touchdown"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
        td_ewma = td_ewma.values if hasattr(td_ewma, 'values') else td_ewma
        # Don't fillna - preserve NaN for players without TD history
        td_ewma = pd.Series(td_ewma, index=player_features.index)
        
        # Calculate EWMA of touches (opportunities)
        if "touches_ewma" not in player_features.columns:
            touches_ewma = player_groups["touches"].apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
            touches_ewma = touches_ewma.values if hasattr(touches_ewma, 'values') else touches_ewma
            touches_ewma = pd.Series(touches_ewma, index=player_features.index)
        else:
            touches_ewma = player_features["touches_ewma"]
        
        # Historical conversion rate per opportunity = EWMA(TDs) / EWMA(touches)
        # Preserve NaN if either component is NaN
        historical_conversion_rate = td_ewma / (touches_ewma + 1)
        
        # xTD = current touches * historical conversion rate
        # Preserve NaN if touches is NaN
        xTD = player_features["touches"] * historical_conversion_rate
        xTD = xTD.where(player_features["touches"].notna(), np.nan)
        
        # Regression factor: EWMA + λ(xTD - EWMA)
        # Preserve NaN if components are missing
        player_features["regression_td_factor"] = td_ewma + REGRESSION_LAMBDA * (xTD - td_ewma)
    else:
        player_features["regression_td_factor"] = np.nan
    
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
    if "play_type" in pbp.columns or "down" in pbp.columns:
        team_plays = pbp.groupby(["posteam", "game_id", "season", "week"]).size().reset_index(name="total_plays")
        team_plays = team_plays.rename(columns={"posteam": "team"})
    else:
        # Fallback: use player touches as proxy
        if "touches" not in player_features.columns:
            player_features["touches"] = 0
        team_plays = player_features.groupby(["team", "game_id", "season", "week"])["touches"].sum().reset_index(name="total_plays")
    
    # EWMA of team play volume
    team_plays = team_plays.sort_values(by=["team", "season", "week"])
    team_plays["team_play_volume_ewma"] = (
        team_plays.groupby("team")["total_plays"]
        .apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    )
    
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
    
    # Team total air yards
    # Ensure air_yards exists
    if "air_yards" not in player_features.columns:
        player_features["air_yards"] = 0
    
    team_air_yards = player_features.groupby(["team", "game_id", "season", "week"])["air_yards"].sum().reset_index(name="team_total_air_yards")
    team_air_yards = team_air_yards.sort_values(by=["team", "season", "week"])
    team_air_yards["team_total_air_yards_ewma"] = (
        team_air_yards.groupby("team")["team_total_air_yards"]
        .apply(calculate_ewma_feature, alpha=EWMA_ALPHA)
    )
    
    # Merge team context
    team_context = pd.merge(
        team_plays[["team", "game_id", "season", "week", "team_play_volume_ewma"]],
        team_rz_touches[["team", "game_id", "season", "week", "team_total_red_zone_touches_ewma"]],
        on=["team", "game_id", "season", "week"],
        how="outer"
    )
    team_context = pd.merge(
        team_context,
        team_air_yards[["team", "game_id", "season", "week", "team_total_air_yards_ewma"]],
        on=["team", "game_id", "season", "week"],
        how="outer"
    )
    
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
    cols_to_drop = ["team_total_red_zone_touches_ewma", "team_total_air_yards_ewma"]
    for col in cols_to_drop:
        if col in player_features.columns:
            player_features = player_features.drop(columns=[col])
    
    required_cols = ["team", "game_id", "season", "week"]
    if "team_total_red_zone_touches_ewma" in team_context.columns:
        required_cols.append("team_total_red_zone_touches_ewma")
    if "team_total_air_yards_ewma" in team_context.columns:
        required_cols.append("team_total_air_yards_ewma")
    
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
    
    # Air yards share (normalized by team passing volume)
    # Normalize share by team volume to account for low-volume vs high-volume passing teams
    # Formula: share * (team_volume / league_average) = share * volume_factor
    # This ensures a 20% share on a high-volume team is recognized as better than 67% on low-volume team
    
    # Ensure air_yards_ewma exists
    if "air_yards_ewma" not in df.columns:
        df["air_yards_ewma"] = 0
    if "team_total_air_yards_ewma" not in df.columns:
        df["team_total_air_yards_ewma"] = 0
    
    # Calculate basic share first
    # Only calculate share for players who actually have air_yards_ewma > 0
    # For players with 0 air_yards, set share to NaN (will be kept as NaN for position-aware imputation)
    df["air_yards_share_ewma"] = np.where(
        df["air_yards_ewma"] > 0,
        df["air_yards_ewma"] / (df["team_total_air_yards_ewma"] + 1),
        np.nan  # Use NaN for players with no air_yards (not 0, which could be misinterpreted)
    )
    
    # Calculate league average team air yards per game for normalization baseline
    # Use median to be robust to outliers, but only on teams with meaningful air yards
    team_air_yards_nonzero = df[df["team_total_air_yards_ewma"] > 0]["team_total_air_yards_ewma"]
    if len(team_air_yards_nonzero) > 0:
        league_avg_team_air_yards = team_air_yards_nonzero.median()
        if pd.isna(league_avg_team_air_yards) or league_avg_team_air_yards == 0:
            league_avg_team_air_yards = team_air_yards_nonzero.mean()
            if pd.isna(league_avg_team_air_yards) or league_avg_team_air_yards == 0:
                league_avg_team_air_yards = 2750.0
    else:
        league_avg_team_air_yards = 2750.0
    
    # Normalize the share by team volume (only for players with actual share values)
    # Share * (team_volume / league_avg) = effective share adjusted for team passing volume
    volume_factor = df["team_total_air_yards_ewma"] / league_avg_team_air_yards
    df["air_yards_share_ewma"] = (
        df["air_yards_share_ewma"] * volume_factor
    )
    
    # For players with no air_yards_ewma (NaN share), keep as NaN so position-aware imputation handles it
    # This ensures WR/TE with 0 air_yards get NaN (not applicable), while RBs also get NaN (not applicable)
    # The model will learn that NaN = feature doesn't apply, not that 0 = good
    
    # aDOT share (player aDOT / team aDOT)
    # Ensure aDOT_ewma exists
    if "aDOT_ewma" not in df.columns:
        df["aDOT_ewma"] = 0
    
    team_adot = df.groupby(["team", "game_id", "season", "week"])["aDOT_ewma"].mean().reset_index(name="team_aDOT_ewma")
    df = pd.merge(
        df,
        team_adot,
        on=["team", "game_id", "season", "week"],
        how="left"
    )
    if "team_aDOT_ewma" not in df.columns:
        df["team_aDOT_ewma"] = 0
    
    # Calculate aDOT share - preserve NaN if player has no aDOT
    df["aDOT_share_ewma"] = (
        df["aDOT_ewma"] / (df["team_aDOT_ewma"] + 1)
    )
    # Set to NaN if player has no aDOT (not 0)
    df.loc[df["aDOT_ewma"].isna() | (df["aDOT_ewma"] == 0), "aDOT_share_ewma"] = np.nan
    
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
    defensive_stats = pbp.groupby(["defteam", "game_id", "season", "week"]).agg({
        "yards_gained": "sum",
        "touchdown": "sum",
        "interception": "sum" if "interception" in pbp.columns else lambda x: 0,
    }).reset_index()
    
    defensive_stats = defensive_stats.rename(columns={"defteam": "team"})
    
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
    
    # Opponent red zone defense rank (simplified - rank by rz_completion_pct_allowed EWMA)
    defensive_stats = defensive_stats.sort_values(by=["season", "week", "def_ewma_red_zone_completion_pct_allowed"], ascending=[True, True, False])
    defensive_stats["opponent_red_zone_def_rank"] = (
        defensive_stats.groupby(["season", "week"])["def_ewma_red_zone_completion_pct_allowed"]
        .rank(method="min", ascending=False)
    ).fillna(32)
    
    return defensive_stats[["team", "game_id", "season", "week", 
                           "def_ewma_yards_allowed_per_game", "def_ewma_TDs_allowed_per_game",
                           "def_ewma_red_zone_completion_pct_allowed", "def_ewma_interceptions_per_game",
                           "opponent_red_zone_def_rank"]]

