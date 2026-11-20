"""GTA VI trophy rarity forecast, v2 (gta6-trophy-rarity-forecast-01).

Python 3.13, standard library only, deterministic, no network.
`python3 main.py` reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input)
and writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output): five quantiles
(0.05 / 0.25 / 0.5 / 0.75 / 0.95) of each of the case's 55 trophies' Exophase
earn rate (%) at the target, scored by mean pinball loss on the logit scale.

Idea: forecast earners, not rates. A rate is earners / players_tracked, and
a refresh moves both at once -- new players dilute every rate while the
players already tracked keep earning. So the model works in earner counts
E(N) as a function of players tracked N, and only converts back to a rate
at the very end.

Flat history (practice cases: every read identical, 638 players). The
competition rules pin down the first refresh: between 04:00 and 04:30 UTC
19 Nov the page jumped 638 -> 3,868 players and 943 -> 4,276 trophies
earned. So 3,230 new players arrived, all of them earned the one instant
trophy (it stays at 100%), and only 103 extra trophies were earned across
the other 54 -- the new cohort had barely played. The v1 entry spread
those 103 by a single earliness weight and paid 0.0642 on the server,
while a local estimate against that same family said 0.043: the sealed
truth is more concentrated on the top trophies than any single family.
v2 therefore averages three neighbouring spread families in logit space
(sqrt(E0+1)*EARLY, E0^0.5*EARLY^1.5, E0^1.0*EARLY^0.5) and lets the
spread follow how much the families disagree: where they agree the band
is narrow, where they split it widens. The median is also clipped to
stay between pure dilution (nobody new earns it) and the last observed
rate (nobody un-earns it). For a flat case whose window does not contain
the known refresh, the same refresh shape is scaled by the horizon
(a refresh becomes likely over ~5 hours).

History with refreshes in it (the final case). Distinct reads give (N, E)
points per trophy. Each trophy gets a log-log least-squares fit
log E = a + b log N, slopes shrunk empirical-Bayes style toward the
median slope (tau = 0.30). Players tracked are projected with a power
law in time, cross-checked against exponential growth (the smaller wins)
and capped at 30x the last count. Three extra guards were tried here in
v2 -- rescaling all trophies to a fitted total, capping each rate at
1.05x its historical max, and blending the median toward the last
observed rate -- and all three made four final-like synthetic cases
worse (mean 0.1955 vs 0.1663 without them), so they were reverted: the
trend path is the v1 one. What did help, on all four synthetic cases,
was shrinking the random-walk term in the spread from 0.05 to 0.03 per
day (mean 0.1663 -> 0.1590); below 0.03 the loss stops moving, so 0.03
is the plateau, not a knife edge. Spread combines fit residual,
extrapolation distance and that random-walk term.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves the script's directory off sys.path; restore it so
# a module shipped next to this file would still import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ZSCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Exophase earn rates (%) on the public trophy page later on launch day
# (2026-11-19 ~16:23 UTC), keyed by trophy id. Only the ordering matters:
# which trophies a fresh player earns first. Baked-in constants; the
# program never touches the network.
LATE_DAY = {
    1: 0.03, 2: 99.99, 3: 0.73, 4: 0.14, 5: 0.06, 6: 0.06, 7: 0.06,
    8: 0.24, 9: 0.06, 10: 0.06, 11: 0.06, 12: 0.06, 13: 0.06, 14: 0.03,
    15: 0.03, 16: 0.04, 17: 0.04, 18: 0.03, 19: 0.04, 20: 1.89, 21: 0.41,
    22: 0.04, 23: 0.04, 24: 0.04, 25: 0.03, 26: 0.03, 27: 0.04, 28: 0.03,
    29: 0.04, 30: 0.03, 31: 0.03, 32: 0.03, 33: 0.09, 34: 0.05, 35: 0.24,
    36: 0.06, 37: 0.14, 38: 0.14, 39: 0.06, 40: 0.04, 41: 0.04, 42: 0.04,
    43: 0.04, 44: 0.08, 45: 0.04, 46: 0.09, 47: 0.04, 48: 0.05, 49: 0.04,
    50: 0.11, 51: 0.06, 52: 0.04, 53: 0.05, 54: 0.04, 55: 0.04,
}
LATE_DAY_FALLBACK = 0.04

# The first refresh, as stated in the competition rules.
KNOWN_REFRESH_AT = datetime(2026, 11, 19, 4, 30, tzinfo=timezone.utc)
KNOWN_REFRESH_N = 3868
KNOWN_REFRESH_NEW_PLAYERS = 3230.0
KNOWN_REFRESH_EXTRA_EARNERS = 103.0
KNOWN_REFRESH_GROWTH = 3868.0 / 638.0
# A horizon of this many minutes makes a refresh about 63% likely.
REFRESH_TIME_SCALE_MIN = 280.0


def pct_bound(x: float) -> float:
    """Keep a rate inside (0, 100) for the logit transform."""
    return min(99.99, max(0.01, x))


def to_logit(pct: float) -> float:
    p = pct_bound(pct)
    return math.log(p / (100.0 - p))


def from_logit(l: float) -> float:
    return 100.0 / (1.0 + math.exp(-l))


def round_js4(x: float):
    """Round to 4 decimals, half up, like JavaScript's Math.round."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def read_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def band(median: float, width: float):
    """Five quantiles around a median rate: non-decreasing, in [0, 100]."""
    centre = to_logit(median)
    out = []
    prev = -1.0
    for z in ZSCORES:
        q = round_js4(from_logit(centre + z * width))
        q = min(100.0, max(0.0, q))
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def collapse_reads(case: dict):
    """History collapsed to reads that actually changed (N or any rate)."""
    reads = []
    for h in case["history"]:
        n = h["players_tracked"]
        rates = tuple(h["rates"])
        if reads and reads[-1][1] == n and reads[-1][2] == rates:
            # Same page again: this state now runs to the later read's time.
            reads[-1] = (read_time(h["at"]), n, rates)
            continue
        reads.append((read_time(h["at"]), n, rates))
    return reads


def ols(xs, ys):
    """Least squares y = a + b*x. Returns (intercept, slope)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def middle(vals):
    ordered = sorted(vals)
    m = len(ordered)
    if m % 2:
        return ordered[m // 2]
    return 0.5 * (ordered[m // 2 - 1] + ordered[m // 2])


def is_instant(rate) -> bool:
    return rate is not None and rate >= 99.995


def flat_case(case: dict, reads):
    """History never moved: expect a refresh before the target."""
    last_time, n0, last_rates = reads[-1]
    target_time = read_time(case["target"])
    horizon_min = case["horizon_minutes"]

    if target_time >= KNOWN_REFRESH_AT and last_time < KNOWN_REFRESH_AT and n0 <= 1500:
        # The known first refresh lands inside this case's window.
        n_goal = float(KNOWN_REFRESH_N)
        extra_pool = KNOWN_REFRESH_EXTRA_EARNERS
    else:
        # Generic refresh prior: the longer the horizon, the likelier a
        # refresh, growing the tracked count toward a full first jump.
        p = 1.0 - math.exp(-max(horizon_min, 0.0) / REFRESH_TIME_SCALE_MIN)
        n_goal = n0 * (1.0 + (KNOWN_REFRESH_GROWTH - 1.0) * p)
        extra_pool = KNOWN_REFRESH_EXTRA_EARNERS * max(n_goal - n0, 0.0) / KNOWN_REFRESH_NEW_PLAYERS
    grow = max(n_goal - n0, 0.0)

    trophies = case["trophies"]
    base = [0.0 if r is None else r * n0 / 100.0 for r in last_rates]

    # Three neighbouring ways to split the thin pool of extra earners.
    # Each family is a weight per trophy; the instant trophies get zero
    # weight because every new player earns them outright.
    def family_weights(pow_e, pow_late, plus_one):
        weights = []
        for i, t in enumerate(trophies):
            if is_instant(last_rates[i]):
                weights.append(0.0)
                continue
            late = LATE_DAY.get(t["id"], LATE_DAY_FALLBACK)
            e = base[i] + 1.0 if plus_one else base[i]
            weights.append((e ** pow_e) * (late ** pow_late))
        return weights

    families = [
        family_weights(0.5, 1.0, True),    # sqrt(E0+1) * EARLY
        family_weights(0.5, 1.5, False),   # E0^0.5 * EARLY^1.5
        family_weights(1.0, 0.5, False),   # E0^1.0 * EARLY^0.5
    ]
    # Per family, the median rate each trophy would land on.
    family_medians = []
    for weights in families:
        total = sum(weights) or 1.0
        meds = []
        for i in range(len(trophies)):
            extra = extra_pool * weights[i] / total
            meds.append(100.0 * (base[i] + extra) / n_goal if n_goal > 0 else 0.0)
        family_medians.append(meds)

    rows = []
    for i, t in enumerate(trophies):
        if is_instant(last_rates[i]):
            # Everybody earns this one immediately; it stays at 100%.
            rows.append({"id": t["id"], "q": band(100.0, 0.02)})
            continue
        logits = [to_logit(family_medians[j][i]) for j in range(3)]
        # Logit-average median across the three families.
        med = from_logit(sum(logits) / 3.0)
        # Count-consistency: between pure dilution (nobody new earns it)
        # and the last observed rate (nobody un-earns it).
        dilute = 100.0 * base[i] / n_goal if n_goal > 0 else 0.0
        lo, hi = dilute, last_rates[i]
        if hi is not None:
            med = min(max(med, lo), hi)
        else:
            med = max(med, lo)
        # Adaptive spread: how far the families disagree about this trophy.
        disagreement = max(logits) - min(logits)
        width = 0.04 + 0.4 * disagreement
        rows.append({"id": t["id"], "q": band(med, width)})
    return {"trophies": rows}


def trend_case(case: dict, reads):
    """History moved: fit earners-vs-players per trophy, project players."""
    target_time = read_time(case["target"])
    t_start = reads[0][0]
    hours = [(r[0] - t_start).total_seconds() / 3600.0 for r in reads]
    log_n = [math.log(max(r[1], 1)) for r in reads]
    n_last = reads[-1][1]
    last_rates = reads[-1][2]
    horizon_days = case["horizon_minutes"] / 1440.0
    target_hours = (target_time - t_start).total_seconds() / 3600.0

    # Project players tracked: power law in time, cross-checked against
    # exponential growth; take the smaller (a launch curve bends down, it
    # does not keep doubling), and cap the jump.
    a_pow, b_pow = ols([math.log1p(h) for h in hours], log_n)
    n_pow = math.exp(a_pow + b_pow * math.log1p(max(target_hours, 0.0)))
    a_exp, b_exp = ols(hours, log_n)
    n_exp = math.exp(a_exp + b_exp * max(target_hours, 0.0))
    n_goal = min(n_pow, n_exp)
    n_goal = min(max(n_goal, float(n_last)), float(n_last) * 30.0)
    log_n_goal = math.log(max(n_goal, 1.0))
    log_growth = math.log(max(n_goal / max(n_last, 1), 1.0))

    trophies = case["trophies"]
    instant = [is_instant(r) for r in last_rates]

    # Per-trophy earner elasticity: log E = a + b log N. Slopes are
    # shrunk toward the median slope only as much as their standard
    # error warrants (empirical Bayes, tau = 0.30): a slope the data
    # pins down is mostly kept, because over a ~1,000x extrapolation
    # any shrinkage is leveraged hard.
    slopes = []
    fits = []
    for i in range(len(trophies)):
        xs, ys = [], []
        for r in reads:
            rate = r[2][i]
            if rate is None:
                continue
            xs.append(math.log(max(r[1], 1)))
            ys.append(math.log(max(rate * r[1] / 100.0, 0.5)))
        if len(xs) >= 2 and max(xs) > min(xs):
            a, b = ols(xs, ys)
            resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / len(xs))
            mx = sum(xs) / len(xs)
            sxx = sum((x - mx) ** 2 for x in xs)
            se = resid / math.sqrt(sxx) if sxx > 0 else 9.9
        else:
            a, b, resid, se = (ys[0] if ys else 0.0), 1.0, 0.3, 9.9
        slopes.append(b)
        fits.append((a, b, resid, se))
    slope_typical = middle(slopes)
    tau2 = 0.30 ** 2

    # Raw per-trophy projections, clipped to [E_last, 0.9 * N_goal].
    projected = []
    for i, t in enumerate(trophies):
        rate_last = last_rates[i]
        e_last = 0.0 if rate_last is None else rate_last * n_last / 100.0
        if instant[i]:
            projected.append(None)
            continue
        a, b, resid, se = fits[i]
        w = tau2 / (tau2 + se * se)
        b_final = w * b + (1.0 - w) * slope_typical
        e_goal = math.exp(a + b_final * log_n_goal)
        e_goal = min(max(e_goal, e_last), 0.9 * n_goal)
        projected.append(e_goal)

    rows = []
    for i, t in enumerate(trophies):
        if instant[i]:
            rows.append({"id": t["id"], "q": band(100.0, 0.02)})
            continue
        a, b, resid, se = fits[i]
        w = tau2 / (tau2 + se * se)
        b_final = w * b + (1.0 - w) * slope_typical
        e_goal = projected[i]
        med = 100.0 * e_goal / n_goal if n_goal > 0 else 0.0
        med = min(med, 75.0)
        # Honest spread: fit residual + extrapolation + random walk.
        extrap = abs(b_final - 1.0) * log_growth
        width = math.sqrt(0.03 * horizon_days + resid ** 2 + (0.25 * extrap) ** 2 + 0.03)
        width = min(max(width, 0.45), 1.25)
        rows.append({"id": t["id"], "q": band(med, width)})
    return {"trophies": rows}


def forecast(case: dict) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{case.get('id')} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{case.get('id')} has no history")
    reads = collapse_reads(case)
    n_values = {r[1] for r in reads}
    if len(n_values) >= 2 and len(reads) >= 2:
        return trend_case(case, reads)
    return flat_case(case, reads)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast(case)
        (OUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
