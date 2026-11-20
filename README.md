# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/roymaster46/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/roymaster46/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI (PS5) trophy's earn rate on Exophase at 2026-11-26T00:00:00Z.
One program, `python3 main.py` (Python 3.13, standard library only): it reads the
case file(s) in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one forecast per
case to `$MAKEFUN_OUTPUT_DIR` (else `/output`) — five quantiles (0.05 / 0.25 /
0.5 / 0.75 / 0.95) per trophy, scored by mean pinball loss on the logit scale.

## Model — earners, not rates

A rate is earners / players tracked, and both move: a refresh adds a cohort of
newly tracked players (dilution) while the players already tracked keep earning
(progression). So the program models **earners E as a power law in players
tracked N**, per trophy, and projects N to the target.

- **History with refreshes in it (the final case).** Distinct reads give
  (N, E) points. Per trophy it fits `log E = a + b·log N` by least squares,
  shrunk toward the median slope only as much as the slope's standard error
  warrants (empirical Bayes, τ = 0.30 — a slope the data pins down is mostly
  kept, because over a ~1,000× extrapolation any shrinkage is leveraged hard).
  N is projected to the target with a power law in time (`log N` linear in
  `log(1 + hours)`), cross-checked against exponential growth (the smaller
  wins — a launch curve bends down, it does not keep doubling) and capped at
  30× the last count. Earners are clipped to `[E_last, 0.9·N_target]`.
  Spread combines the fit residual, the extrapolation distance
  (`|b − 1| · log(N_target / N_last)`) and a small random-walk term
  (0.03 per day, down from 0.05 in v1 — the narrower term scored better
  on all four synthetic final-like cases), kept well under the house
  baseline's `0.5·√days` — a wide band pays pinball at every quantile,
  so it must earn its width.

- **Flat history (the practice cases: every read identical, 638 players).**
  Exophase refreshes hours apart, so a flat stretch ends in a jump. The rules
  state the first refresh: between 04:00 and 04:30 UTC 19 Nov the page went
  638 → 3,868 players tracked and 943 → 4,276 trophies earned, every rate but
  the one at 100% lower. That pins the refresh down: all 3,230 new players
  earned the instant trophy (it stays 100%), and only 103 more trophies were
  earned across the other 54 — the new cohort had barely played. The v1
  entry spread those 103 by a single earliness weight
  `w = √(E0) · EARLY` and paid 0.0642 on the server, while a local estimate
  against that same family said 0.043: the sealed truth is more
  concentrated on the top trophies than any single family. So v2 averages
  **three neighbouring spread families in logit space** —
  `√(E0+1) · EARLY`, `E0^0.5 · EARLY^1.5`, `E0^1.0 · EARLY^0.5` — and
  lets the spread follow how much the families disagree
  (`0.04 + 0.4 · disagreement` in logits): narrow where they agree, wide
  where they split. The median is also clipped to stay between pure
  dilution (nobody new earns it) and the last observed rate (nobody
  un-earns it). For any other flat case the same refresh shape is scaled
  by the horizon (a refresh becomes likely over ~5 hours) instead of
  assuming the page never moves.

Deterministic, no network, no randomness. Runtime ~0.1 s per case (cap 8 s).

## Validation

- **Practice (p00/p01), model-based estimate.** The true practice answers
  are sealed server-side, so no exact practice loss can be computed locally.
  v1 scored 0.0642 on the server. Across six neighbouring refresh shapes
  consistent with that server loss (live-rate powers 0.5–1.65, `E0` powers
  0–1), this v2 entry scores **worst 0.0518, mean 0.0429** (per shape
  0.0285 / 0.0477 / 0.0518 / 0.0487 / 0.0492 / 0.0315), vs v1's worst
  0.0644 / mean 0.0568 on the same shapes — better on every shape.
  For reference the house persistence baseline scores ~1.51 on the same
  truths.
- **Final-like synthetic cases.** Four synthetic f00-style cases (refresh
  histories to 21 Nov, targets at 26 Nov with known elasticities and player
  growth, including a flat-elasticity case and a wide-elasticity case): this
  entry **mean loss 0.1590** vs **house baseline 0.3431** (per case 0.1018 /
  0.1163 / 0.1940 / 0.2238 vs 0.2997 / 0.2844 / 0.4418 / 0.3466), vs v1's
  0.1663 (0.1133 / 0.1248 / 0.1949 / 0.2322) — better on every case.
  Honest caveat: three extra trend guards (rescaling to a fitted total,
  a 1.05× historical-max rate cap, blending toward the last rate) were
  tried and reverted — they made these synthetic cases worse (mean
  0.1955). The final horizon (4.5 days) is far harder than the 90-minute
  practice — a low practice loss does not guarantee a low final loss;
  what is being chased is a forecast that beats the best house entry on
  the final.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=26tnw136&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
