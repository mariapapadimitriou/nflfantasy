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


def calculate_ewma(df, alpha=0.5, ewma_weeks=3):
    """
    Calculate Exponentially Weighted Moving Averages with recency features.
    Uses alpha decay to give more weight to recent games.
    Also adds features to handle rare breakout games vs consistent performance.
    
    Note: DataFrame must be sorted by season and week before calling this function.
    """
    # Ensure dataframe is sorted by season and week for proper time series calculations
    if "season" in df.columns and "week" in df.columns:
        df = df.sort_values(by=["season", "week"]).reset_index(drop=True)
    
    # Helper function to calculate EWMA with shift
    def calc_ewma(series, alpha=alpha):
        return series.ewm(alpha=alpha, adjust=False).mean().shift(1)
    
    # EWMA for receiving/rushing touchdowns/yards
    df["ewma_receiving_touchdowns"] = calc_ewma(df["receiving_tds"], alpha)
    df["ewma_receiving_yards"] = calc_ewma(df["receiving_yards"], alpha)
    df["ewma_rushing_touchdowns"] = calc_ewma(df["rushing_tds"], alpha)
    df["ewma_rushing_yards"] = calc_ewma(df["rushing_yards"], alpha)
    
    # EWMA for receptions and targets if available
    if "receptions" in df.columns:
        df["ewma_receptions"] = calc_ewma(df["receptions"], alpha)
    if "targets" in df.columns:
        df["ewma_targets"] = calc_ewma(df["targets"], alpha)
    if "touchdown_attempts" in df.columns:
        df["ewma_touchdown_attempts"] = calc_ewma(df["touchdown_attempts"], alpha)
    # EWMA for red zone completion %
    if "red_zone_completion_pct" in df.columns:
        df["ewma_red_zone_completion_pct"] = calc_ewma(df["red_zone_completion_pct"], alpha)
    
    # Recency features to handle rare breakout games
    # Time since last touchdown (weeks ago, 0 = last game, 1 = 2 games ago, etc.)
    # Lower is better (more recent)
    if "receiving_tds" in df.columns and "rushing_tds" in df.columns:
        total_tds = df["receiving_tds"] + df["rushing_tds"]
        # Find last game with touchdown
        td_occurred = (total_tds > 0).astype(int)
        
        # Calculate weeks since last TD: for each game, count backwards to last TD
        # Reset index to work with position-based indexing
        df_reset = df.reset_index(drop=True)
        td_occurred_reset = td_occurred.reset_index(drop=True)
        
        weeks_since = []
        for i in range(len(df_reset)):
            # Look backwards from current position to find last TD
            if i == 0:
                # First game: no history, set to max
                weeks_since.append(ewma_weeks + 1)
            else:
                # Count backwards to find last TD
                found = False
                for j in range(i - 1, -1, -1):
                    if td_occurred_reset.iloc[j] > 0:
                        weeks_since.append(i - j)
                        found = True
                        break
                if not found:
                    # Never scored before, set to max
                    weeks_since.append(ewma_weeks + 1)
        
        # Shift by 1 to use previous game's value for prediction
        weeks_since_series = pd.Series(weeks_since, index=df_reset.index)
        df["weeks_since_last_td"] = weeks_since_series.shift(1).fillna(ewma_weeks + 1)
        
        # Recent streak: count touchdowns in last ewma_weeks games
        # Use rolling window to count recent touchdowns
        recent_td_count = (
            td_occurred.rolling(window=ewma_weeks, min_periods=1).sum().shift(1).fillna(0)
        )
        df["recent_td_count"] = recent_td_count
        
        # Consistency score: how many of last ewma_weeks games had touchdowns
        # (0 to 1, higher = more consistent)
        df["td_consistency"] = (recent_td_count / ewma_weeks).fillna(0)
    
    return df


def calculate_prev_game(df):
    """
    Calculate previous game stats (most recent single game before current).
    This replaces previous season stats for more recent and relevant data.
    """
    # Previous game's stats (shift by 1 game)
    df["prev_game_receiving_yards"] = df["receiving_yards"].shift(1)
    df["prev_game_receiving_touchdowns"] = df["receiving_tds"].shift(1)
    df["prev_game_rushing_yards"] = df["rushing_yards"].shift(1)
    df["prev_game_rushing_touchdowns"] = df["rushing_tds"].shift(1)
    
    # Previous game's receptions and targets if available
    if "receptions" in df.columns:
        df["prev_game_receptions"] = df["receptions"].shift(1)
    if "targets" in df.columns:
        df["prev_game_targets"] = df["targets"].shift(1)
    
    return df


def get_ewma_yards(df):
    # Defensive EWMA feature
    # Note: This function is deprecated - ewma_yapg is now calculated from defensive stats
    # and merged into the main dataframe. Keeping for backward compatibility but it won't
    # work if yards_gained doesn't exist in df.
    if "yards_gained" in df.columns:
        df["ewma_yapg"] = (
            df["yards_gained"].ewm(alpha=0.5, adjust=False).mean().shift(1)
        )
    return df


def get_qb_ewma_stats(df, alpha=0.5):
    """Calculate EWMA for QB stats"""
    df["qb_ewma_passing_yards"] = (
        df["passing_yards"].ewm(alpha=alpha, adjust=False).mean().shift(1)
    )
    df["qb_ewma_passing_tds"] = (
        df["passing_tds"].ewm(alpha=alpha, adjust=False).mean().shift(1)
    )
    return df


def american_odds_to_probability(odds):
    odds = int(odds)
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)


def compute_red_zone_stats(pbp: pd.DataFrame) -> tuple:
    """
    Calculate red zone stats from DEFENSIVE team perspective.
    Returns how often defenses allow touchdowns when opponents reach the red zone.
    """
    pbp["is_red_zone"] = pbp["yardline_100"] <= 20
    red_zone_drives = (
        pbp[pbp["is_red_zone"]]
        .groupby(["game_id", "drive", "posteam", "defteam"])
        .size()
        .reset_index()
        .drop(0, axis=1)
    )

    def drive_scored(row):
        mask = (
            (pbp["game_id"] == row["game_id"])
            & (pbp["drive"] == row["drive"])
            & (pbp["posteam"] == row["posteam"])
        )
        return (pbp[mask]["touchdown"] == 1).any()

    red_zone_drives["red_zone_scored_td"] = red_zone_drives.apply(drive_scored, axis=1)

    drive_info = (
        pbp.groupby(["game_id", "drive"])[["season", "week"]].first().reset_index()
    )
    red_zone_drives = red_zone_drives.merge(
        drive_info, on=["game_id", "drive"], how="left"
    )

    # Calculate from DEFENSIVE team perspective (defteam)
    # This represents how often the defense allows touchdowns in red zone
    rz_team_week = (
        red_zone_drives.groupby(["defteam", "season", "week"])["red_zone_scored_td"]
        .agg(["sum", "count"])
        .reset_index()
    )
    rz_team_week["red_zone_scoring_pct"] = rz_team_week["sum"] / rz_team_week["count"]
    # Rename defteam to posteam for compatibility with existing merge code
    rz_team_week = rz_team_week.rename(columns={"defteam": "posteam"})

    rz_team_week = rz_team_week.sort_values(["posteam", "season", "week"])
    # Use EWMA for red zone stats
    # Import config to get EWMA_ALPHA, but allow fallback if not available
    try:
        from .config import EWMA_ALPHA
    except (ImportError, AttributeError):
        EWMA_ALPHA = 0.5  # Default fallback
    rz_team_week["ewma_red_zone"] = (
        rz_team_week.groupby("posteam")["red_zone_scoring_pct"]
        .ewm(alpha=EWMA_ALPHA, adjust=False)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    prev_year = (
        rz_team_week.groupby(["posteam", "season"])
        .agg(prev_year_td=("sum", "sum"), prev_year_drives=("count", "sum"))
        .reset_index()
    )
    prev_year["prev_red_zone"] = (
        prev_year["prev_year_td"] / prev_year["prev_year_drives"]
    )
    prev_year["next_year"] = prev_year["season"] + 1

    return rz_team_week, prev_year

