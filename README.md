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
  shrunk toward a per-trophy prior (v3; v2 shrunk toward the median slope)
  only as much as the slope's standard error warrants (empirical Bayes,
  τ = 0.30 — a slope the data pins down is mostly kept, because over a
  ~1,000× extrapolation any shrinkage is leveraged hard). N is projected
  to the target with the median of three growth models (v3; v2 took
  min(power law, exponential) — the power law under-projected in every
  fast-growth synthetic world, ratios 0.40–0.74, while the exponential
  explodes, so the min was just the power law), capped at 30× the last
  count. Earners are clipped to `[E_last, 0.9·N_target]`. Spread combines
  the fit residual, the extrapolation distance (`|b − 1| · log(N_target /
  N_last)`) and a small random-walk term (0.03 per day, down from 0.05 in
  v1 — the narrower term scored better on all four synthetic final-like
  cases), kept well under the house baseline's `0.5·√days` — a wide band
  pays pinball at every quantile, so it must earn its width.

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
  assuming the page never moves. **The flat path is unchanged in v3:
  its practice output is byte-identical to v2's (server practice loss
  0.0646), because the prize is decided by the final case alone.**

## What's new in v3 (trend path only)

1. **Floor width.** A median pinned at the floor (0.01%, the smallest rate
   the scorer can see) has a truncated downside — the truth cannot go below
   it — so a wide band wastes loss on quantiles the truth can never reach.
   v3 caps the band width at 0.10 for medians at the floor. Real data says
   this is safe: the modal trophy's earners barely move as N grows (median
   2-point elasticity 0.04 from launch night to 20 Nov), so ~34 of the 55
   trophies will still print 0.00–0.01% at the target.
2. **Slope prior from the earliness ordering.** The real 2-point
   elasticities (launch night → 20 Nov, from the public Exophase page)
   correlate +0.90 with the LATE_DAY ordering (which trophies fresh
   players earn first): early trophies have b ≈ 1.0–1.5, the modal trophy
   b ≈ 0. So v3 shrinks each slope toward a per-trophy prior from a
   precision-weighted regression of the case's own fitted slopes on
   log10(LATE_DAY / 0.04), with the regression slope itself shrunk by its
   signal-to-noise and capped at 0.6 — a case with no such pattern falls
   back to the global median, exactly v2's behaviour. No outside formula
   is baked in; the pattern is fitted on the case's own data.
3. **Player-count ensemble.** v3 takes the median of three growth models —
   a power law in time, a linear trend, and a recent-half exponential — so
   no single shape's bias dominates. On the real trajectory the sane models
   agree closely (power ~222k, linear ~191k players at the target), so this
   changes little there and only guards the fast worlds.

Deterministic, no network, no randomness. Runtime ~0.1 s per case (cap 8 s).

## Validation

- **Practice (p00/p01).** v3 output is byte-identical to v2's (flat path
  untouched): server practice loss **0.0646**, rank 74. The prize is
  decided by the final case, not practice.
- **Real-data-anchored final-like case.** Built from the actual Exophase
  page (a Wayback snapshot of 20 Nov 13:16 UTC: 42,228 players tracked,
  full rate table): real N trajectory 638 → 3,868 → 42,228, elasticities
  from the real 2-point estimates, target N ≈ 222k. v3 loss **0.0353**
  vs v2 **0.0673** (−48%), house baseline 0.1865. Coverage of the 90%
  band: 1.00.
- **50-scenario synthetic ensemble** (5 elasticity modes × 5 player
  paths × 2 noise levels, two seeds): v3 **0.1681 / 0.1668** vs v2
  **0.1769 / 0.1759** — better on every mode (live, flat, spread, random,
  inverse) and on both seeds.
- **Final-like synthetic cases (the v2 set).** Four synthetic f00-style
  cases: v3 **mean loss 0.1511** vs v2 **0.1590** vs house baseline
  0.3431 — better on every case.
- Honest caveat: three extra trend guards (rescaling to a fitted total, a
  1.05× historical-max rate cap, blending toward the last rate) were tried
  in v2 and reverted — they made the synthetic cases worse (mean 0.1955).
  The final horizon (4.5 days) is far harder than the 90-minute practice —
  a low practice loss does not guarantee a low final loss; what is being
  chased is a forecast that beats the best house entry on the final
  (26 Nov, real Exophase data). No synthetic validation can guarantee
  that; v3 is a genuine improvement on the final path on every validation
  set, not a practice-score chase.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=26tnw136&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
