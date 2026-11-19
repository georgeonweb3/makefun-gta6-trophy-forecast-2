"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.

Reads every ``*.in`` case in the input directory and writes one
``<id>.out`` forecast per case. Standard library only, no network,
fully deterministic: the same input always produces the same bytes.

Approach, in brief (full reasoning in METHOD.md):

* Each trophy's earn rate follows a saturating logistic curve in hours
  since release. The curve's midpoint is picked by grid search, pulled
  toward a prior that depends on what kind of trophy it is (story
  chapter, side mission, activity, skill test, grind, 100% completion,
  platinum); the ceiling is the median ceiling implied by the reads.
* A damped log-time trend is blended in at 25% as insurance against a
  wrong curve shape.
* When a case's history shows no Exophase refresh at all and the
  horizon is short, the flat lines are not extrapolated. Instead a
  launch-night refresh prior is applied: the player base jumps about
  sixfold and the newcomers have earned almost nothing yet, so every
  non-intro rate falls by roughly a factor of five.
* Medians are then forced to respect what the trophies require of each
  other (platinum last, 100% completion second-last, story in order),
  and each trophy gets a logit-normal band whose width grows with fit
  error, horizon, player-base growth and model disagreement.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Midpoint candidates (hours) tried for every trophy's logistic curve.
GRID = [4, 6, 9, 13, 19, 28, 40, 58, 84, 120, 175, 250, 360]

# Launch-night refresh, disclosed in the rules: 638 -> 3,868 players
# tracked, and non-intro trophies earned falling from 305 to 408 in
# aggregate. F is the player-base jump; K is how fast the new cohort
# earns each kind of trophy relative to the old one.
REFRESH_JUMP = 3868.0 / 638.0          # ~6.06
REFRESH_K = {
    "story": 0.10, "side": 0.10,
    "activity": 0.08, "skill": 0.08,
    "grind": 0.03,
    "platinum": 0.01, "completion": 0.01,
}
REFRESH_SPREAD = 0.45

# (kind, midpoint prior in hours, ceiling prior as a fraction)
KIND_PRIORS = {
    "story": (48.0, 0.50),
    "side": (70.0, 0.16),
    "activity": (45.0, 0.10),
    "skill": (90.0, 0.07),
    "grind": (130.0, 0.03),
    "completion": (170.0, 0.015),
    "platinum": (220.0, 0.002),
}
# Extra band width (logit units) for the slowest kinds of trophy.
KIND_EXTRA = {"grind": 0.10, "platinum": 0.15, "completion": 0.15}

CHAPTER_H50 = {1: 20.0, 2: 32.0, 3: 48.0, 4: 64.0}
FINAL_H50 = 95.0

_NUMBER = re.compile(r"(\d[\d,]*)")
_CHAPTER = re.compile(r"chapter\s+(\d)", re.IGNORECASE)
_COUNT_COMPLETE = re.compile(r"complete\s+(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------- maths

def _clamp_pct(x):
    return min(99.99, max(0.01, x))


def _logit_pct(pct):
    p = _clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def _inv_logit_pct(lg):
    # Guard against overflow for extreme logits.
    if lg > 20.0:
        return 99.99
    if lg < -20.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-lg))


def _round4(x):
    """Round to 4 decimals, half up, returning an int for whole values."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def _norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


Z = [_norm_ppf(level) for level in LEVELS]


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _hours_since_release(when):
    return (when - RELEASE).total_seconds() / 3600.0


def _log1h(hours):
    return math.log(max(1.0 + hours, 0.01))


def _ols(xs, ys):
    """Least-squares slope/intercept; flat line when x never varies."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


# ------------------------------------------------------------- trophies

def classify(trophy):
    """Return (kind, h50 prior, ceiling prior) from a trophy's text.

    The opening mission is its own case (handled separately). Otherwise
    the description decides: story chapters and the finale first, then
    the 100% trophy and the platinum, then anything counted (stunts,
    card sets, vaults, vehicles, dollars) as a grind, then skill tests
    (medals, races won, duels beaten), then named side missions, and
    everything left over is an ordinary activity.
    """
    desc = trophy.get("description", "") or ""
    low = desc.lower()
    if "gone fishing" in low:
        return "intro", 4.0, 1.0
    if trophy.get("tier") == "platinum":
        return "platinum", 220.0, 0.002
    if "100% completion" in low:
        return "completion", 170.0, 0.015
    m = _CHAPTER.search(low)
    if m and int(m.group(1)) in CHAPTER_H50:
        return "story", CHAPTER_H50[int(m.group(1))], 0.50
    if "final mission" in low and " for " not in low:
        return "story", FINAL_H50, 0.50
    if _COUNT_COMPLETE.search(low):
        return "grind", 130.0, 0.03
    if re.search(r"\d[\d,]*\s+(locations|vehicles|bank vaults|races|shops)",
                 low):
        return "grind", 130.0, 0.03
    if "earn $" in low or "spend $" in low or "laundering" in low:
        return "grind", 130.0, 0.03
    if "gold medal" in low or low.startswith("win ") or "win a " in low \
            or low.startswith("beat "):
        return "skill", 90.0, 0.07
    if low.startswith("complete "):
        return "side", 70.0, 0.16
    return "activity", 45.0, 0.10


def distinct_points(history, index):
    """One (hours, rate) per distinct displayed rate, at first sighting.

    Exophase sits flat between its own refreshes, so repeated reads of
    the same printed rate carry no new information.
    """
    seen = {}
    for read in history:
        rates = read.get("rates") or []
        if index >= len(rates) or rates[index] is None:
            continue
        rate = float(rates[index])
        if rate not in seen:
            seen[rate] = _hours_since_release(_parse_time(read["at"]))
    return sorted(((hour, rate) for rate, hour in seen.items()))


def last_rate(history, index):
    for read in reversed(history):
        rates = read.get("rates") or []
        if index < len(rates) and rates[index] is not None:
            return float(rates[index])
    return 0.0


# -------------------------------------------------------------- the fit

def fit_curve(points, h50_prior):
    """Best logistic midpoint/ceiling for one trophy's distinct points.

    For each candidate midpoint the ceiling is the median of the
    ceilings each point implies; the winner minimises mean squared
    logit error plus a pull toward the prior midpoint. Returns
    (h50, ceiling fraction, residual std in logit units).
    """
    if not points:
        return h50_prior, KIND_PRIORS["activity"][1], 0.0
    frac = [(h, min(max(rate / 100.0, 0.0001), 0.9999))
            for h, rate in points]
    best = None
    for h50 in GRID:
        width = h50 / 3.0
        implied = []
        for h, p in frac:
            implied.append(p * (1.0 + math.exp(-(h - h50) / width)))
        implied.sort()
        mid = len(implied) // 2
        if len(implied) % 2:
            ceiling = implied[mid]
        else:
            ceiling = 0.5 * (implied[mid - 1] + implied[mid])
        ceiling = min(max(ceiling, 0.0001), 0.99)
        sq = 0.0
        for h, p in frac:
            pred = ceiling / (1.0 + math.exp(-(h - h50) / width))
            pred = min(max(pred, 0.0001), 0.9999)
            err = math.log(p / (1.0 - p)) - math.log(pred / (1.0 - pred))
            sq += err * err
        score = sq / len(frac) + 0.15 * (math.log(h50 / h50_prior) ** 2)
        if best is None or score < best[0]:
            best = (score, h50, ceiling)
    _, h50, ceiling = best
    width = h50 / 3.0
    residuals = []
    for h, p in frac:
        pred = ceiling / (1.0 + math.exp(-(h - h50) / width))
        pred = min(max(pred, 0.0001), 0.9999)
        residuals.append(math.log(p / (1.0 - p))
                         - math.log(pred / (1.0 - pred)))
    mean_res = sum(residuals) / len(residuals)
    var = sum((r - mean_res) ** 2 for r in residuals) / len(residuals)
    return h50, ceiling, math.sqrt(var)


def trend_rate(points, target_h, anchor_rate, ceiling):
    """Damped log-time trend, re-anchored at the last observed rate.

    A straight line is fitted to logit(rate) against ln(1 + hours);
    its slope is halved toward zero (the trend is insurance, not the
    forecast), the line is shifted to pass through the last observed
    rate, and the result is capped at the curve's ceiling.
    """
    if not points:
        return anchor_rate
    xs = [_log1h(h) for h, _ in points]
    ys = [_logit_pct(rate) for _, rate in points]
    slope, _ = _ols(xs, ys)
    slope *= 0.5
    last_h = points[-1][0]
    trend_logit = _logit_pct(anchor_rate) \
        + slope * (_log1h(target_h) - _log1h(last_h))
    return min(_inv_logit_pct(trend_logit), ceiling * 100.0)


def players_growth(history, target_h):
    """Forecast tracked-players growth factor to the target.

    ln(players) is regressed on ln(1 + hours); the fitted exponent is
    shrunk toward 1.0, the launch-phase prior, with weight
    span / (span + 12 h) on the data. Returns P(target) / P(last).
    """
    rows = []
    for read in history:
        players = read.get("players_tracked")
        if players:
            rows.append((_hours_since_release(_parse_time(read["at"])),
                         math.log(float(players))))
    if len(rows) < 2:
        return 1.0
    xs = [_log1h(h) for h, _ in rows]
    ys = [y for _, y in rows]
    slope, intercept = _ols(xs, ys)
    span = max(rows[-1][0] - rows[0][0], 0.0)
    weight = span / (span + 12.0) if span > 0 else 0.0
    slope = weight * slope + (1.0 - weight) * 1.0
    forecast = math.exp(intercept + slope * _log1h(target_h))
    last = rows[-1][1]
    return max(math.exp(math.log(forecast) - last), 1.0)


def history_is_flat(history):
    """True when no read differs from the first: no refresh yet."""
    if len(history) < 2:
        return True
    first = history[0]
    for read in history[1:]:
        if read.get("players_tracked") != first.get("players_tracked"):
            return False
        if read.get("rates") != first.get("rates"):
            return False
    return True


# ----------------------------------------------------------- forecasting

def intro_median(horizon_days):
    """The opening mission: nearly everyone who plays finishes it.

    Its share only falls as the tracked base broadens to include
    owners who have not played yet: 100% minus a small slice that
    grows with the horizon (0.13 points a day, at most 0.6).
    """
    return 100.0 - min(0.6, horizon_days * 0.13)


def forecast_one(case, trophy, index, target_h, horizon_days, growth):
    """Return (median percent, logit sigma) for one trophy."""
    kind, h50_prior, _ = classify(trophy)
    history = case.get("history") or []
    anchor = last_rate(history, index)
    extra = KIND_EXTRA.get(kind, 0.0)

    if kind == "intro":
        sigma = math.sqrt(0.26 ** 2 + (0.30 * math.sqrt(horizon_days)) ** 2
                          + (0.30 * math.log10(max(growth, 1.0))) ** 2)
        return intro_median(horizon_days), sigma

    flat = history_is_flat(history)
    short = case.get("horizon_minutes", 0) <= 720
    if flat and short:
        # No refresh in the history: do not extrapolate flat lines.
        # The new cohort earns at rate k of the old one, so the rate
        # is diluted by (1 + (F - 1) * k) / F.
        k = REFRESH_K.get(kind, 0.08)
        median = anchor * (1.0 + (REFRESH_JUMP - 1.0) * k) / REFRESH_JUMP
        sigma = math.sqrt(REFRESH_SPREAD ** 2 + extra ** 2)
        return _clamp_pct(median), sigma

    points = distinct_points(history, index)
    h50, ceiling, resid_std = fit_curve(points, h50_prior)
    width = h50 / 3.0
    curve = ceiling / (1.0 + math.exp(-(target_h - h50) / width))
    curve_pct = _clamp_pct(curve * 100.0)
    trend = trend_rate(points, target_h, anchor, ceiling)
    blended_logit = 0.75 * _logit_pct(curve_pct) + 0.25 * _logit_pct(trend)
    median = _inv_logit_pct(blended_logit)
    disagreement = min(0.5 * abs(_logit_pct(curve_pct) - _logit_pct(trend)),
                       1.2)
    sigma = math.sqrt(
        0.26 ** 2
        + (0.70 * min(resid_std, 0.8)) ** 2
        + (0.30 * math.sqrt(horizon_days)) ** 2
        + (0.30 * math.log10(max(growth, 1.0))) ** 2
        + extra ** 2
        + disagreement ** 2
    )
    return _clamp_pct(median), sigma


def apply_guards(case, medians):
    """Force medians to respect what the trophies require of each other.

    Platinum needs every other trophy; 100% completion needs every
    non-platinum one; the story chapters are played in order.
    """
    trophies = case["trophies"]
    by_id = {t["id"]: i for i, t in enumerate(trophies)}

    def kind_of(t):
        return classify(t)[0]

    plat = [i for i, t in enumerate(trophies) if kind_of(t) == "platinum"]
    if plat:
        cap = min(medians[i] for i in range(len(trophies)) if i not in plat)
        for i in plat:
            medians[i] = min(medians[i], cap)
    comp = [i for i, t in enumerate(trophies) if kind_of(t) == "completion"]
    if comp:
        others = [medians[i] for i, t in enumerate(trophies)
                  if kind_of(t) not in ("platinum", "completion")]
        if others:
            cap = min(others)
            for i in comp:
                medians[i] = min(medians[i], cap)
    chain = []
    for t in trophies:
        desc = (t.get("description", "") or "").lower()
        m = _CHAPTER.search(desc)
        if m:
            chain.append((int(m.group(1)), by_id[t["id"]]))
        elif "final mission" in desc and " for " not in desc:
            chain.append((99, by_id[t["id"]]))
    chain.sort()
    running = None
    for _, i in chain:
        if running is None:
            running = medians[i]
        else:
            running = min(running, medians[i])
            medians[i] = running
    return medians


def forecast_case(case):
    target = _parse_time(case["target"])
    target_h = _hours_since_release(target)
    horizon_days = case.get("horizon_minutes", 0) / 1440.0
    growth = players_growth(case.get("history") or [], target_h)

    medians, sigmas = [], []
    for i, trophy in enumerate(case["trophies"]):
        median, sigma = forecast_one(case, trophy, i, target_h,
                                     horizon_days, growth)
        medians.append(median)
        sigmas.append(sigma)
    medians = apply_guards(case, medians)

    rows = []
    for trophy, median, sigma in zip(case["trophies"], medians, sigmas):
        centre = _logit_pct(median)
        qs = []
        prev = None
        for z in Z:
            value = _round4(_clamp_pct(_inv_logit_pct(centre + z * sigma)))
            if prev is not None and value < prev:
                value = prev
            qs.append(value)
            prev = value
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
