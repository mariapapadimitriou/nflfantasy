# Project brief: NFL touchdown prediction

Context for an agent building this from scratch. It states **what** to build and
what is already known to be true or to go wrong. It deliberately does not
prescribe a language, framework, model family, or host — those are open.

Everything factual below was measured against live data, not assumed.

---

## 1. The product

A weekly ranked shortlist of NFL players most likely to score a touchdown, for
fantasy football decisions. A user opens it, sees perhaps 20–40 names in
probability order with enough supporting detail to trust or dismiss each one,
and acts.

Two consequences follow, and they shape everything else:

- **It is a ranking product, not a classifier.** Nobody consumes a yes/no
  prediction for 300 players. Optimise and report ordering quality.
- **It refreshes weekly, not per-request.** The output for a given week is one
  small document. Nothing about the product requires model inference to happen
  while a user waits.

## 2. Data

**nflverse** is the right source and no serious alternative exists for free.
It is the open dataset behind nflfastR: no API key, updated within hours of
each game, and what the public NFL analytics community works from. Python
binding `nflreadpy`, R binding `nflreadr`; the underlying files are public
Parquet/CSV on GitHub, so any language can read them directly.

Tables that matter, with the columns that earn their place:

| Table | Use | Notes |
|---|---|---|
| `pbp` | Touchdown labels, red-zone usage, defensive stats | `td_player_id`, `yardline_100`, `rusher_player_id`, `receiver_player_id`, `posteam`, `defteam`. Large — ~100k rows for 3 seasons. |
| `player_stats` | Weekly stat lines | `carries`, `targets`, `receptions`, rushing/receiving yards and TDs. |
| `rosters_weekly` | Who was on which team that week | **Use this, not `rosters`.** Seasonal rosters list a traded player under both clubs. |
| `schedules` | Game context and betting lines | `spread_line`, `total_line`, `home_moneyline`, `roof`, `wind`, `home_rest`. Lines are present for upcoming games. |
| `snap_counts` | `offense_pct` — share of snaps played | Strongest single usage signal. Keys on **PFR ids**; join to gsis via `players.pfr_id`. |
| `injuries` | Official `report_status` | Out / Doubtful / Questionable. Published for upcoming weeks. |
| `players` | Id crosswalk | Takes **no** `seasons` argument, unlike every other loader. |

Join keys: `(player_id, season, week)` for players, `(team, season, week)` for
teams, `game_id` for games. Player ids are gsis ids (`00-00XXXXX`).

Also available and unexplored: `depth_charts`, `nextgen_stats`, `pfr_advstats`,
`ff_opportunity` (expected production), `participation`.

## 3. The prediction target

Binary, per player per game: **did they score any touchdown?**

- WR/TE/RB: rushing or receiving both count.
- QB: rushing only. A passing touchdown belongs to the receiver.
- `td_player_id` in play-by-play gives the scorer regardless of play type.

**Base rate is ~19–21%** among players with meaningful usage. Note what this
means: predicting "nobody scores" scores ~80% accuracy. Accuracy is worse than
useless here as a headline metric — it is actively misleading.

## 4. Reference numbers

Real results from this data, so you know what "working" looks like. Two
independent implementations, same week (2026 week 2), both gradient-boosted
trees:

| | Players | Base rate | Test AUC | Precision@10 |
|---|---|---|---|---|
| Implementation A | 352 | 0.186 | 0.737 | 0.60 |
| Implementation B | 312 | 0.210 | 0.694 | 0.60 |

**AUC in the 0.68–0.75 band is realistic.** Much above 0.80 on a chronological
holdout almost certainly means leakage — check that no feature reads the row's
own game.

**Precision@10 of 0.60 against a base rate of ~0.20** is the number that
describes the product: the top ten score at roughly 3× the rate of the eligible
pool. Report this.

Calibration matters as much as ordering, because users read the percentages as
percentages. A correct model's mean predicted probability equals the base rate,
and the probabilities across a week's slate sum to roughly the number of
players who will actually score (~60–80 league-wide among skill players).

## 5. Traps

Each of these was a real, shipped bug. They are subtle, they fail silently, and
several produced plausible-looking output while being badly wrong.

**Preprocessing statistics captured after the transform they describe.**
Standardise, *then* record the means and standard deviations, and you record 0
and 1 for every column. Prediction then applies `(x − 0) / 1` and feeds raw
values to a model fitted on z-scores. Symptom: stored means are all exactly
zero. Test for it.

**Training and inference with separate preprocessing code.** The above is one
instance of a general failure. Build the matrix once, in one place, used by
both paths. Reindex onto a fixed feature list at inference so a missing column
becomes NaN rather than shifting every later column one position left.

**Labelling unplayed games as zero.** The week being predicted has no outcome.
If the label is `0` rather than null, the upcoming week reads as "nobody
scored" — it corrupts training, evaluation, and the UI at once. Make the label
three-valued: scored / did not score / not yet played.

**Deriving "did they play" from placeholder rows.** Rolling features need a row
for the upcoming week, so a placeholder gets added — and if the played flag is
computed afterwards, every rostered player is marked as having played a game
that has not kicked off. Capture appearance from real stat lines *before*
adding placeholders.

**Normalising by a group average without checking the divisor.** Dividing a
feature by its position-week average is sound, and a floor on the divisor
guards against near-zero denominators — but a floor set above the feature's
typical average pins the divisor to a constant and the normalisation silently
becomes the identity. Three features were 99.7–100% identical to their raw
values this way. Assert that normalised ≠ raw.

**Normalising sparse features by a group average at all.** A breakout-game
indicator is zero for most players most weeks, so its position-week average is
also zero. No floor rescues this. Scale such features against an absolute
threshold instead.

**Outlier capping that sees the future.** Winsorising against a quantile of a
player's entire series lets his later games set the ceiling applied to his
earlier ones. Shift first, then cap against an expanding quantile of prior
games only.

**Random train/test splits on panel data.** Weekly rows for one player are not
independent. A random split puts his later games in training and his earlier
ones in test. Split chronologically.

**Calibrating with balanced class weights.** If you handle imbalance during
training (`scale_pos_weight` or similar) *and* fit the calibrator with balanced
weights, the scores are balanced twice. Measured effect: mean predicted
probability 0.466 against a base rate of 0.186 — every number inflated ~2.5×,
the top player shown at 74% when his real chance was ~40%. Fit the calibrator
unweighted. Ranking metrics cannot detect this (AUC was identical); log loss
and mean-probability-vs-base-rate both catch it immediately.

**Synthetic oversampling on a matrix containing NaN.** Gradient-boosted trees
handle missing values natively and informatively. SMOTE cannot, which forces a
sentinel-value workaround that then leaks sentinels into evaluation sets and
interpolates nonsense between them. Use class weighting instead and keep NaN.

**`dropna()` across a whole row.** Dropping rows with any null discards every
player whenever one column is unpopulated — e.g. a game whose betting line is
not yet posted. Subset it to the columns that genuinely must be present.

**Displaying data you never act on.** Injury status was fetched and shown but
never filtered on, so a player ruled out could top the shortlist. Either use it
or do not show it.

**Attributing explanations positionally.** If SHAP values are mapped to feature
names by index, any column reordering misattributes every explanation. Derive
both from the same ordered list.

## 6. Features that work

Ordered by measured gain in a trained model:

1. **Usage, position-normalised** — touches, targets, yards, recent scoring,
   red-zone touches, red-zone share. The bulk of the signal (~45% combined).
2. **Snap share** — cleanest single availability/role signal.
3. **Market-implied team total** — `total_line / 2 − spread_line / 2`. The
   market's own view of how many points this offense scores, and therefore how
   many touchdowns are on offer. Available before kickoff.
4. **Opposing defence** — touchdowns and red-zone touchdowns allowed, rolling.
5. **Team volume** — plays and red-zone plays, rolling.

Weak or unproven: weather, rest days, dome. Cheap to include, unclear value at
~10k training rows — measure before keeping.

Use exponentially weighted averages over prior games (α≈0.5). Require ~4 games
of history before a player is modelled. Filter to players with real individual
usage — a backup on a great offense should not rank — and apply that filter
identically to training and inference so the model is fitted on the population
it will be asked to rank.

## 7. Deployment constraint

**Serverless Python ML at request time does not fit on free tiers.** Measured:
xgboost alone is 239 MB installed; the runtime set with pandas/scipy/sklearn is
~546 MB. Vercel's Python functions cap at 250 MB. Add a read-only filesystem
and a 60-second function limit against 5–15 minute training, and the
interactive-training-in-a-browser shape is simply not viable there.

The architecture that does work, and is free:

```
scheduled job (CI, cron, anywhere)
  → fetch, train, predict
  → write one JSON document
  → static site reads it
```

No server, no database, no cold starts, no size limits. It also matches the
product: the data changes weekly, not per request. The whole output for a week
is a few hundred KB.

If you want live interactivity later, serve the precomputed JSON and do
filtering client-side.

## 8. Open choices

Everything here is yours to decide:

- **Language and stack.** Python is convenient for the modelling libraries and
  `nflreadpy`; the underlying nflverse files are plain Parquet, so R, JS or
  anything else can read them directly.
- **Model family.** Gradient-boosted trees are a strong baseline and handle
  missing values natively. Logistic regression with good features is a
  defensible, more interpretable alternative. Calibrated ensembles, monotonic
  constraints, or a hierarchical model over team-total × player-share are all
  reasonable. The traps in §5 apply regardless.
- **UI.** Anything. The data is one JSON document.
- **Hosting.** Any static host. The scheduled job can run anywhere with
  network access.
- **Explanations.** SHAP is one option; permutation importance or a simple
  "here's why" from the top contributing features also works.

## 9. Definition of done

- Produces a ranked list for the **next unplayed week**, not a past one.
- Probabilities are calibrated: mean ≈ base rate.
- Reports AUC and precision@k on a **chronological** holdout, alongside the
  base rate for comparison.
- Players ruled out by the injury report are excluded.
- Unplayed games show no outcome, rather than a zero.
- Refreshes weekly without manual intervention.
- Each prediction can be explained in terms a fantasy player understands
  ("high red-zone usage, good matchup, high team total") rather than raw
  feature names.
