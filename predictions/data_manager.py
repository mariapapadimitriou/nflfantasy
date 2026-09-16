"""
Loads raw NFL data and turns it into the modelling frame.

Historical rows and the upcoming week are produced by a single pass over the
same sources, so the features the model trains on are built exactly the way the
features it scores are built. Processing the two halves separately is how
training and serving drift apart.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from django.db.models import Q

from .config import (
    CACHE_ENABLED,
    EWMA_WEEKS,
    HISTORICAL_SEASONS,
    POSITION_NORMALIZED_CAP,
    POSITION_NORMALIZED_FEATURES,
    POSITIONS,
    REPORT_STATUS_ORDER,
)
from .data_source import get_data_source
from .feature_engineering import (
    GAME_KEYS,
    PLAYER_ORDER,
    TEAM_GAME_KEYS,
    add_position_normalized_features,
    calculate_defensive_features,
    calculate_player_features,
    calculate_qb_stats,
    calculate_team_context_features,
    calculate_team_shares,
)
from .models import TrainingData

logger = logging.getLogger(__name__)

STAT_COLUMNS = [
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "passing_tds",
    "passing_yards",
    "receptions",
    "targets",
    "carries",
]

# Receiving production is not a meaningful signal for these positions, and
# rushing is not for those. NaN says "does not apply"; 0 would say "had the
# chance and did nothing", which is a different claim.
INAPPLICABLE_STATS = {
    "QB": ["targets_ewma", "receptions_ewma", "receiving_yards_ewma", "receiving_touchdowns_ewma"],
    "RB": ["targets_ewma", "receptions_ewma", "receiving_yards_ewma", "receiving_touchdowns_ewma"],
    "WR": ["carries_ewma", "rushing_yards_ewma", "rushing_touchdowns_ewma"],
    "TE": ["carries_ewma", "rushing_yards_ewma", "rushing_touchdowns_ewma"],
}

QB_CONTEXT_COLUMNS = ["qb_passing_yards_ewma", "qb_rushing_yards_ewma"]


class NFLDataManager:
    """Loads NFL data and builds the feature frame."""

    def __init__(self, data_source_type: str = "nflreadpy"):
        self.data_source = get_data_source(data_source_type)
        self._source_cache: Dict[tuple, pd.DataFrame] = {}

    # ------------------------------------------------------------- loading

    def _load(self, name: str, seasons: List[int]) -> pd.DataFrame:
        """Fetch a source table once per season set, per manager instance."""
        key = (name, tuple(seasons))
        if key not in self._source_cache:
            self._source_cache[key] = getattr(self.data_source, f"load_{name}")(seasons)
        return self._source_cache[key]

    def get_seasons_to_load(self, season: int) -> List[int]:
        """Current season plus the configured number of prior ones."""
        return list(range(season - HISTORICAL_SEASONS, season + 1))

    def load_and_process_data(
        self, season: int, week: int, force_reload: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Build the training frame and the frame for the week being predicted."""
        if force_reload:
            self._source_cache.clear()

        seasons = self.get_seasons_to_load(season)
        schedules = self._load("schedules", seasons)
        schedules = schedules[
            (schedules["season"] < season)
            | ((schedules["season"] == season) & (schedules["week"] <= week))
        ]

        df = self._process_data(
            player_stats=self._load("player_stats", seasons),
            pbp=self._load("pbp", seasons),
            roster=self._load("rosters", seasons),
            schedules=schedules,
        )

        current_week = df[(df["season"] == season) & (df["week"] == week)].copy()

        # Train only on games that were actually played; an upcoming game has no
        # outcome, and a rostered player who never appeared has no label either.
        historical = df[
            (df["season"] < season) | ((df["season"] == season) & (df["week"] < week))
        ]
        historical = historical[historical["played"] == 1].copy()

        logger.info(
            "Season %s week %s: %s training rows, %s players to score",
            season,
            week,
            len(historical),
            len(current_week),
        )

        if len(current_week) == 0:
            raise ValueError(
                f"No players found for Season {season}, Week {week}. "
                "The schedule for that week may not be published yet."
            )

        self.save_historical_data(historical)

        return {
            "df": historical.reset_index(drop=True),
            "current_week": current_week.reset_index(drop=True),
        }

    # --------------------------------------------------------- persistence

    def load_historical_data(self, season: int, week: int) -> Optional[pd.DataFrame]:
        """Read previously stored training rows for everything before this week."""
        if not CACHE_ENABLED:
            return None

        # Prior seasons, plus completed weeks of the current one. Chaining a
        # season__lt filter ahead of this would make the second clause
        # unreachable and silently drop the current season entirely.
        records = list(
            TrainingData.objects.filter(
                Q(season__lt=season) | Q(season=season, week__lt=week)
            ).values()
        )
        if not records:
            return None

        df = pd.DataFrame(records)
        if "features" in df.columns:
            features = pd.json_normalize(df["features"].tolist())
            df = pd.concat([df.drop(columns=["features"]), features], axis=1)
        return df

    def save_historical_data(self, df: pd.DataFrame) -> None:
        """Persist completed games as training rows.

        Only played games are stored. Writing an upcoming week here would record
        every player as having failed to score, and those fabricated negatives
        would be indistinguishable from real ones on the next read.
        """
        if not CACHE_ENABLED or df.empty:
            return

        playable = df[df["played"] == 1]
        if playable.empty:
            return

        identity = [
            "season", "week", "player_id", "player_name", "team",
            "position", "against", "touchdown", "played", "report_status",
        ]
        records = [
            TrainingData(
                season=int(row["season"]),
                week=int(row["week"]),
                player_id=str(row["player_id"]),
                player_name=str(row.get("player_name", "")),
                team=str(row.get("team", "")),
                position=str(row.get("position", "")),
                against=str(row.get("against", "")),
                touchdown=int(row["touchdown"]) if pd.notna(row["touchdown"]) else 0,
                played=int(row["played"]),
                report_status=str(row.get("report_status", "Healthy")),
                features={
                    k: (None if pd.isna(v) else v)
                    for k, v in row.items()
                    if k not in identity and not isinstance(v, (list, dict))
                },
            )
            for _, row in playable.iterrows()
        ]

        try:
            touched = sorted({(r.season, r.week) for r in records})
            for season, week in touched:
                TrainingData.objects.filter(season=season, week=week).delete()
            TrainingData.objects.bulk_create(records, batch_size=1000)
            logger.info("Stored %s training rows across %s weeks", len(records), len(touched))
        except Exception:
            logger.exception("Could not persist training rows")

    # ---------------------------------------------------------- processing

    def _process_data(
        self,
        player_stats: pd.DataFrame,
        pbp: pd.DataFrame,
        roster: pd.DataFrame,
        schedules: pd.DataFrame,
    ) -> pd.DataFrame:
        """Turn raw source tables into the modelling frame."""
        weekly_stats, appeared = self._build_weekly_stats(player_stats, schedules, roster)
        pbp = self._ensure_game_keys(pbp, schedules)

        df = self._build_player_game_frame(roster, schedules)
        df = self._add_outcome(df, pbp, appeared)

        # Identity travels with the stat lines into feature engineering, so the
        # rolling features come back already keyed to a game, team and position.
        weekly_stats = weekly_stats.merge(
            df[PLAYER_ORDER + ["game_id", "team", "position"]].drop_duplicates(
                subset=PLAYER_ORDER
            ),
            on=PLAYER_ORDER,
            how="left",
        )

        player_features = calculate_player_features(pbp, weekly_stats)
        player_features = self._blank_inapplicable_stats(player_features)
        player_features = self._add_qb_context(player_features, df)

        df = df.merge(
            player_features.drop(columns=STAT_COLUMNS, errors="ignore"),
            on=PLAYER_ORDER + ["game_id"],
            how="left",
            suffixes=("", "_dup"),
        )
        df = df.drop(columns=[c for c in df.columns if c.endswith("_dup")])

        team_context = calculate_team_context_features(player_features, pbp, schedules)
        df = df.drop(columns=[c for c in team_context.columns if c in df.columns and c not in TEAM_GAME_KEYS])
        df = df.merge(team_context, on=TEAM_GAME_KEYS, how="left")
        df = calculate_team_shares(df, team_context)

        defense = calculate_defensive_features(pbp, schedules)
        if not defense.empty:
            df = df.merge(
                defense.rename(columns={"team": "against"}),
                on=["against"] + GAME_KEYS,
                how="left",
            )

        df = self._add_derived_features(df)
        df = add_position_normalized_features(
            df, POSITION_NORMALIZED_FEATURES, POSITION_NORMALIZED_CAP
        )

        return self._finalize(df)

    def _build_weekly_stats(
        self, player_stats: pd.DataFrame, schedules: pd.DataFrame, roster: pd.DataFrame
    ) -> tuple:
        """Assemble weekly stat lines and record who actually appeared.

        Placeholder rows are added for the upcoming week so rolling features
        extend into it, but the record of who appeared is taken *before* they
        are added. Deriving it afterwards would mark every rostered player as
        having played a game that has not kicked off.
        """
        columns = ["player_id", "season", "week"] + [
            c for c in STAT_COLUMNS if c in player_stats.columns
        ]
        missing = set(STAT_COLUMNS) - set(player_stats.columns)
        if missing:
            logger.warning("player_stats is missing %s", ", ".join(sorted(missing)))

        weekly_stats = player_stats[columns].copy()
        appeared = weekly_stats[PLAYER_ORDER].drop_duplicates()

        placeholders = self._upcoming_week_placeholders(weekly_stats, schedules, roster)
        if not placeholders.empty:
            weekly_stats = pd.concat([weekly_stats, placeholders], ignore_index=True)
            logger.debug("Added %s placeholder rows for upcoming games", len(placeholders))

        return weekly_stats, appeared

    def _upcoming_week_placeholders(
        self, weekly_stats: pd.DataFrame, schedules: pd.DataFrame, roster: pd.DataFrame
    ) -> pd.DataFrame:
        """Empty stat lines for rostered players whose game has not been played."""
        if schedules.empty:
            return pd.DataFrame()

        active = roster[roster["status"] == "ACT"][["gsis_id", "season", "team"]]
        active = active.rename(columns={"gsis_id": "player_id"}).drop_duplicates()

        slate = pd.concat(
            [
                schedules[["season", "week", side]].rename(columns={side: "team"})
                for side in ("home_team", "away_team")
            ],
            ignore_index=True,
        )
        scheduled = slate.merge(active, on=["season", "team"], how="inner")[PLAYER_ORDER]

        known = weekly_stats[PLAYER_ORDER].drop_duplicates()
        placeholders = scheduled.merge(known, on=PLAYER_ORDER, how="left", indicator=True)
        placeholders = placeholders[placeholders["_merge"] == "left_only"].drop(columns=["_merge"])

        # Left as NaN, not zero: these players have no stat line yet, and a zero
        # would enter the rolling averages as a genuine scoreless performance.
        for column in STAT_COLUMNS:
            placeholders[column] = np.nan
        return placeholders.drop_duplicates()

    @staticmethod
    def _ensure_game_keys(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
        if "game_id" not in pbp.columns and "gameId" in pbp.columns:
            pbp = pbp.rename(columns={"gameId": "game_id"})
        if "season" not in pbp.columns and "game_id" in pbp.columns:
            pbp = pbp.merge(
                schedules[GAME_KEYS].drop_duplicates(), on="game_id", how="left"
            )
        return pbp

    @staticmethod
    def _build_player_game_frame(roster: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
        """One row per rostered player per scheduled game."""
        roster_summary = (
            roster[roster["position"].isin(POSITIONS) & (roster["status"] == "ACT")][
                ["gsis_id", "position", "season", "team", "rookie_year", "full_name"]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        roster_summary["rookie"] = (
            roster_summary["rookie_year"] == roster_summary["season"]
        ).astype(int)

        games = schedules[
            GAME_KEYS + ["home_team", "away_team", "home_qb_id", "away_qb_id"]
        ].copy()

        sides = []
        for side, opponent, qb in (
            ("home_team", "away_team", "home_qb_id"),
            ("away_team", "home_team", "away_qb_id"),
        ):
            # Joined on the identifying keys only. Dropping rows with any null
            # would discard every player whenever a game lacks a betting line.
            merged = roster_summary.merge(
                games, how="inner", left_on=["season", "team"], right_on=["season", side]
            )
            merged["against"] = merged[opponent]
            merged["is_home"] = int(side == "home_team")
            merged["qb_id"] = merged[qb]
            sides.append(merged)

        df = pd.concat(sides, ignore_index=True)
        if df.empty:
            raise ValueError(
                "No games found after joining rosters to the schedule. "
                "Team abbreviations may not match between the two sources."
            )

        df = df.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
        # A player traded mid-season appears on both rosters for that season and
        # would otherwise be duplicated across both clubs' full schedules.
        df = df.drop_duplicates(subset=["player_id", "game_id"], keep="last")

        return df[
            PLAYER_ORDER
            + ["game_id", "player_name", "position", "team", "against", "is_home", "rookie", "qb_id"]
        ]

    @staticmethod
    def _add_outcome(
        df: pd.DataFrame, pbp: pd.DataFrame, appeared: pd.DataFrame
    ) -> pd.DataFrame:
        """Attach the played flag and the touchdown label.

        The label is deliberately three-valued. A game that has not been played
        carries NaN rather than 0, so an upcoming week cannot be mistaken for a
        week in which nobody scored -- by the trainer, the evaluator or the UI.
        """
        df = df.merge(appeared.assign(_appeared=1), on=PLAYER_ORDER, how="left")
        completed = set(pbp.loc[pbp["game_id"].notna(), "game_id"].unique())
        game_completed = df["game_id"].isin(completed)

        df["played"] = np.where(game_completed & df["_appeared"].notna(), 1, 0)
        df = df.drop(columns=["_appeared"])

        scorers = pbp[pbp["touchdown"] == 1][["td_player_id", "game_id"]].drop_duplicates()
        scorers = scorers.rename(columns={"td_player_id": "player_id"}).assign(_scored=1)
        df = df.merge(scorers, on=["player_id", "game_id"], how="left")

        df["touchdown"] = np.where(
            df["_scored"].notna(), 1.0, np.where(df["played"] == 1, 0.0, np.nan)
        )
        return df.drop(columns=["_scored"])

    @staticmethod
    def _blank_inapplicable_stats(player_features: pd.DataFrame) -> pd.DataFrame:
        if "position" not in player_features.columns:
            return player_features
        for position, columns in INAPPLICABLE_STATS.items():
            mask = player_features["position"] == position
            for column in columns:
                if column in player_features.columns:
                    player_features.loc[mask, column] = np.nan
        return player_features

    @staticmethod
    def _add_qb_context(player_features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Attach each player's starting quarterback's rolling yardage."""
        qb_stats = calculate_qb_stats(player_features)
        if qb_stats.empty or "qb_id" not in df.columns:
            for column in QB_CONTEXT_COLUMNS:
                player_features[column] = np.nan
            return player_features

        player_features = player_features.drop(columns=QB_CONTEXT_COLUMNS, errors="ignore")
        if "qb_id" not in player_features.columns:
            player_features = player_features.merge(
                df[PLAYER_ORDER + ["qb_id"]].drop_duplicates(), on=PLAYER_ORDER, how="left"
            )

        player_features = player_features.merge(
            qb_stats.rename(columns={"player_id": "qb_id"}),
            on=["qb_id", "season", "week"],
            how="left",
        )

        # A quarterback's own passing volume is not supporting-cast context for
        # himself; it is already captured by his individual rushing features.
        if "position" in player_features.columns:
            is_qb = player_features["position"] == "QB"
            player_features.loc[is_qb, QB_CONTEXT_COLUMNS] = np.nan

        return player_features

    @staticmethod
    def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """Combine rushing and receiving into position-agnostic totals."""
        if {"receptions_ewma", "targets_ewma"} <= set(df.columns):
            rate = df["receptions_ewma"] / df["targets_ewma"].replace(0, np.nan)
            df["reception_rate_ewma"] = rate.clip(upper=1.0)

        for label, parts in (
            ("total_yards_ewma", ("receiving_yards_ewma", "rushing_yards_ewma")),
            ("total_touchdowns_ewma", ("receiving_touchdowns_ewma", "rushing_touchdowns_ewma")),
        ):
            present = [c for c in parts if c in df.columns]
            if not present:
                df[label] = np.nan
                continue
            # NaN means the category does not apply to the position, so the
            # total is the sum of whichever categories do.
            total = sum(df[c].fillna(0) for c in present)
            df[label] = total.where(df[present].notna().any(axis=1))

        return df

    @staticmethod
    def _finalize(df: pd.DataFrame) -> pd.DataFrame:
        """Apply history requirements and tidy the frame."""
        df = df.loc[:, ~df.columns.duplicated()]

        if "report_status" not in df.columns:
            df["report_status"] = "Healthy"
        df.loc[~df["report_status"].isin(REPORT_STATUS_ORDER), "report_status"] = "Minor"

        # Rolling features need a few games behind them to say anything.
        appearances = df[df["played"] == 1].groupby("player_id").size()
        experienced = appearances[appearances >= EWMA_WEEKS].index
        before = len(df)
        df = df[df["player_id"].isin(experienced)].copy()
        logger.debug(
            "History filter kept %s of %s rows (>= %s games played)",
            len(df),
            before,
            EWMA_WEEKS,
        )

        return df.sort_values(PLAYER_ORDER).reset_index(drop=True)
