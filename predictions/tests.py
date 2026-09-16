"""
End-to-end regression tests.

These cover the failure modes that are invisible from the UI: an upcoming week
being labelled as played and scoreless, preprocessing statistics captured after
the transform they describe, and position normalization silently degrading into
the raw value.
"""

import unittest.mock

import numpy as np
import pandas as pd
from django.test import TestCase

from .config import (
    FEATURES,
    NUMERIC_FEATURES,
    POSITION_NORMALIZED_CAP,
    POSITION_NORMALIZED_FEATURES,
)
from .data_manager import NFLDataManager
from .data_source import NFLDataSource
from .feature_engineering import add_position_normalized_features, calculate_ewma_feature
from .ml_model import NFLTouchdownModel, precision_at_k
from .preprocessing import FeaturePipeline

TEAMS = ["AAA", "BBB", "CCC", "DDD"]
POSITION_PLAN = [("QB", 1), ("RB", 2), ("WR", 3), ("TE", 1)]
SEASON_WEEKS = {2023: 10, 2024: 10, 2025: 9}
TARGET_SEASON, TARGET_WEEK = 2025, 9  # week 9 is scheduled but not yet played


def build_fixture():
    """Generate deterministic raw tables shaped like the nflreadpy sources."""
    rng = np.random.default_rng(7)

    players = []
    for team in TEAMS:
        for position, count in POSITION_PLAN:
            for index in range(count):
                players.append(
                    {
                        "gsis_id": f"{team}-{position}{index}",
                        "position": position,
                        "team": team,
                        "full_name": f"{team} {position}{index}",
                    }
                )

    rosters, schedules, stats, plays = [], [], [], []

    for season, weeks in SEASON_WEEKS.items():
        for player in players:
            rosters.append(
                {**player, "season": season, "rookie_year": 2019, "status": "ACT"}
            )

        for week in range(1, weeks + 1):
            played = not (season == TARGET_SEASON and week == TARGET_WEEK)

            for home, away in ((TEAMS[0], TEAMS[1]), (TEAMS[2], TEAMS[3])):
                game_id = f"{season}_{week:02d}_{away}_{home}"
                schedules.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week,
                        "home_team": home,
                        "away_team": away,
                        "home_moneyline": int(rng.choice([-150, 120, -200, 110])),
                        "spread_line": float(rng.choice([-3.5, 2.5, -7.0, 1.0])),
                        "home_qb_id": f"{home}-QB0",
                        "away_qb_id": f"{away}-QB0",
                    }
                )
                if not played:
                    continue

                for team, opponent in ((home, away), (away, home)):
                    for player in (p for p in players if p["team"] == team):
                        stat, player_plays = _simulate(
                            player, game_id, season, week, team, opponent, rng
                        )
                        stats.append(stat)
                        plays.extend(player_plays)

    return (
        pd.DataFrame(stats),
        pd.DataFrame(plays),
        pd.DataFrame(rosters),
        pd.DataFrame(schedules),
    )


def _simulate(player, game_id, season, week, team, opponent, rng):
    """One player's stat line and the plays behind it."""
    position = player["position"]
    volume = {"QB": 4, "RB": 14, "WR": 7, "TE": 4}[position]

    carries = int(rng.poisson(volume if position in ("RB", "QB") else 0.3))
    targets = int(rng.poisson(volume if position in ("WR", "TE") else 2.5))
    receptions = int(rng.binomial(targets, 0.65)) if targets else 0

    scored = rng.random() < {"QB": 0.12, "RB": 0.28, "WR": 0.18, "TE": 0.14}[position]

    stat = {
        "player_id": player["gsis_id"],
        "season": season,
        "week": week,
        "carries": carries,
        "targets": targets,
        "receptions": receptions,
        "rushing_yards": float(carries * rng.uniform(2.5, 5.0)),
        "receiving_yards": float(receptions * rng.uniform(6.0, 14.0)),
        "rushing_tds": int(scored and position in ("RB", "QB")),
        "receiving_tds": int(scored and position in ("WR", "TE")),
        "passing_yards": float(rng.uniform(180, 320)) if position == "QB" else 0.0,
        "passing_tds": int(rng.poisson(1.6)) if position == "QB" else 0,
    }

    common = {
        "game_id": game_id,
        "season": season,
        "week": week,
        "posteam": team,
        "defteam": opponent,
        "interception": 0,
        "complete_pass": 0,
        "pass": 0,
        "rush": 0,
        "touchdown": 0,
        "td_player_id": None,
        "receiver_player_id": None,
        "rusher_player_id": None,
    }

    plays = []
    for _ in range(carries):
        plays.append(
            {
                **common,
                "rush": 1,
                "rusher_player_id": player["gsis_id"],
                "yardline_100": float(rng.integers(1, 90)),
                "yards_gained": float(rng.integers(-2, 12)),
            }
        )
    for _ in range(targets):
        complete = int(rng.random() < 0.65)
        plays.append(
            {
                **common,
                "pass": 1,
                "complete_pass": complete,
                "receiver_player_id": player["gsis_id"],
                "yardline_100": float(rng.integers(1, 90)),
                "yards_gained": float(rng.integers(0, 20)) * complete,
            }
        )

    if scored and plays:
        plays[-1] = {**plays[-1], "touchdown": 1, "td_player_id": player["gsis_id"]}
    return stat, plays


class FixtureDataSource(NFLDataSource):
    """Serves the generated fixture in place of nflreadpy."""

    def __init__(self):
        self.stats, self.pbp, self.rosters, self.schedules = build_fixture()

    def _for(self, frame, seasons):
        return frame[frame["season"].isin(seasons)].copy()

    def load_player_stats(self, seasons):
        return self._for(self.stats, seasons)

    def load_pbp(self, seasons):
        return self._for(self.pbp, seasons)

    def load_rosters(self, seasons):
        return self._for(self.rosters, seasons)

    def load_schedules(self, seasons):
        return self._for(self.schedules, seasons)

    def load_teams(self):
        return pd.DataFrame({"team_abbr": TEAMS})

    def load_injuries(self):
        return pd.DataFrame()


def load_fixture_data():
    manager = NFLDataManager.__new__(NFLDataManager)
    manager.data_source = FixtureDataSource()
    manager._source_cache = {}
    return manager.load_and_process_data(TARGET_SEASON, TARGET_WEEK)


class UnitTests(TestCase):
    """Behaviour that can be checked without the full pipeline."""

    def test_ewma_only_sees_prior_games(self):
        series = pd.Series([10.0, 20.0, 30.0, 40.0])
        result = calculate_ewma_feature(series, alpha=0.5, winsorize=False)

        self.assertTrue(pd.isna(result.iloc[0]), "first game has no history to average")
        self.assertEqual(result.iloc[1], 10.0)
        self.assertEqual(result.iloc[2], 15.0)  # 0.5 * 20 + 0.5 * 10

    def test_winsorization_cannot_see_the_future(self):
        """A late outlier must not cap values that came before it."""
        calm = pd.Series([10.0] * 20)
        spike = pd.concat([calm, pd.Series([5000.0])], ignore_index=True)

        baseline = calculate_ewma_feature(calm, alpha=0.5)
        with_spike = calculate_ewma_feature(spike, alpha=0.5).iloc[: len(calm)]

        pd.testing.assert_series_equal(baseline, with_spike, check_names=False)

    def test_position_normalization_actually_divides(self):
        """A floor above the feature's average would make this a no-op."""
        df = pd.DataFrame(
            {
                "position": ["RB"] * 3 + ["WR"] * 3,
                "season": [2025] * 6,
                "week": [1] * 6,
                "total_touchdowns_ewma": [0.6, 0.3, 0.15, 0.4, 0.2, 0.1],
            }
        )
        result = add_position_normalized_features(
            df, POSITION_NORMALIZED_FEATURES, POSITION_NORMALIZED_CAP
        )
        normalized = result["total_touchdowns_ewma_position_normalized"]

        self.assertFalse(
            np.allclose(normalized, result["total_touchdowns_ewma"]),
            "normalized values are identical to the raw values",
        )
        # RB mean is 0.35, so the leading RB should sit above its position average.
        self.assertAlmostEqual(normalized.iloc[0], 0.6 / 0.35, places=6)
        self.assertLessEqual(normalized.max(), POSITION_NORMALIZED_CAP)

    def test_pipeline_captures_statistics_before_scaling(self):
        """Measuring after the transform yields mean 0 and std 1 for everything."""
        df = pd.DataFrame({"a": [1.0, 3.0, 5.0, 7.0], "b": [10.0, 20.0, 30.0, 40.0]})
        pipeline = FeaturePipeline(["a", "b"], ["a", "b"])
        matrix = pipeline.fit_transform(df)

        self.assertAlmostEqual(pipeline.means["a"], 4.0)
        self.assertNotAlmostEqual(pipeline.stds["a"], 1.0)
        self.assertAlmostEqual(matrix[:, 0].mean(), 0.0, places=9)

    def test_pipeline_transform_is_order_independent(self):
        """Reindexing protects against a reordered or incomplete frame."""
        pipeline = FeaturePipeline(["a", "b"], ["a", "b"]).fit(
            pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        )
        shuffled = pipeline.transform(pd.DataFrame({"b": [3.0], "a": [1.0]}))
        straight = pipeline.transform(pd.DataFrame({"a": [1.0], "b": [3.0]}))

        np.testing.assert_allclose(shuffled, straight)
        partial = pipeline.transform(pd.DataFrame({"a": [1.0]}))
        self.assertEqual(partial.shape[1], 2, "missing feature must not shift columns")
        self.assertTrue(np.isnan(partial[0, 1]))

    def test_precision_at_k_ranks_rather_than_thresholds(self):
        y_true = [0, 1, 1, 0, 1]
        y_prob = [0.1, 0.9, 0.8, 0.2, 0.7]
        self.assertEqual(precision_at_k(y_true, y_prob, 3), 1.0)
        self.assertEqual(precision_at_k(y_true, y_prob, 5), 0.6)


class PipelineTests(TestCase):
    """Full data build, training and prediction against the fixture."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data = load_fixture_data()

    def test_upcoming_week_has_no_fabricated_outcomes(self):
        """The regression that made every upcoming player a confirmed non-scorer."""
        current = self.data["current_week"]

        self.assertGreater(len(current), 0)
        self.assertTrue(
            (current["played"] == 0).all(),
            "players were marked as having played a game that has not kicked off",
        )
        self.assertTrue(
            current["touchdown"].isna().all(),
            "unplayed games were labelled as scoreless rather than unknown",
        )

    def test_training_rows_are_played_and_labelled(self):
        history = self.data["df"]

        self.assertGreater(len(history), 0)
        self.assertTrue((history["played"] == 1).all())
        self.assertTrue(history["touchdown"].notna().all())
        self.assertEqual(set(history["touchdown"].unique()), {0.0, 1.0})
        self.assertTrue(
            ((history["season"] < TARGET_SEASON) | (history["week"] < TARGET_WEEK)).all(),
            "training data must not include the target week",
        )

    def test_features_are_present_on_both_sides(self):
        for name, frame in (("history", self.data["df"]), ("current", self.data["current_week"])):
            missing = [f for f in FEATURES if f not in frame.columns]
            self.assertEqual(missing, [], f"{name} frame is missing {missing}")

            populated = [f for f in FEATURES if frame[f].notna().any()]
            self.assertGreater(
                len(populated), len(FEATURES) // 2, f"{name} frame is mostly empty"
            )

    def test_train_and_predict_round_trip(self):
        fast = {"optuna_trials": 3, "optuna_timeout": 60, "num_boost_round": 40,
                "early_stopping_rounds": 10, "precision_at_k": 5}

        with unittest.mock.patch.dict("predictions.config.MODEL_PARAMS", fast):
            model = NFLTouchdownModel(TARGET_SEASON, TARGET_WEEK)
            success, message = model.train(
                self.data["df"], FEATURES, NUMERIC_FEATURES
            )
            self.assertTrue(success, message)

            # The bug that shipped: statistics measured after scaling, so every
            # mean was 0 and every std 1, and prediction never scaled at all.
            self.assertFalse(
                all(abs(v) < 1e-9 for v in model.pipeline.means.values()),
                "scaler means are all zero, so they were captured after scaling",
            )
            self.assertFalse(
                all(abs(v - 1.0) < 1e-9 for v in model.pipeline.stds.values()),
                "scaler stds are all one, so they were captured after scaling",
            )

            reloaded = NFLTouchdownModel(TARGET_SEASON, TARGET_WEEK)
            self.assertTrue(reloaded.load())
            self.assertEqual(reloaded.pipeline.features, FEATURES)
            self.assertAlmostEqual(
                reloaded.pipeline.means[FEATURES[0]], model.pipeline.means[FEATURES[0]]
            )

            predictions, shap_values = reloaded.predict(
                self.data["current_week"], FEATURES, NUMERIC_FEATURES
            )

        self.assertGreater(len(predictions), 0)
        self.assertTrue(predictions["probability"].between(0, 1).all())
        self.assertGreater(
            predictions["probability"].nunique(), 1, "every player got the same score"
        )
        self.assertTrue(
            predictions["probability"].is_monotonic_decreasing, "output is not ranked"
        )
        self.assertTrue((predictions["played"] == "N").all())

        if shap_values:
            contributions = next(iter(shap_values.values()))["contributions"]
            self.assertEqual(set(contributions) - set(FEATURES), set())

    def test_stored_model_reports_ranking_metrics(self):
        fast = {"optuna_trials": 2, "optuna_timeout": 60, "num_boost_round": 30,
                "early_stopping_rounds": 10, "precision_at_k": 5}

        with unittest.mock.patch.dict("predictions.config.MODEL_PARAMS", fast):
            model = NFLTouchdownModel(TARGET_SEASON, TARGET_WEEK - 1)
            success, message = model.train(self.data["df"], FEATURES, NUMERIC_FEATURES)
            self.assertTrue(success, message)

        test_metrics = model.metrics["test"]
        for key in ("auc", "precision_at_5", "base_rate", "log_loss"):
            self.assertIn(key, test_metrics)
        self.assertGreater(test_metrics["base_rate"], 0.0)
