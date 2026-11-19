# Method — GTA VI trophy rarity forecast

Entry: `main.py` for make.fun `gta6-trophy-rarity-forecast-01`.
Job: for each of the 55 GTA VI PlayStation 5 trophies, forecast its earn
rate on Exophase at 2026-11-26T00:00:00Z — a week after launch — from
every read the house made before the deadline, as five quantiles per
trophy. Scoring is mean pinball loss on the logit scale; lower is better.

## 1. What the numbers are

Exophase prints, for each trophy, the percentage of the players it
tracks who have earned it, to two decimals, plus how many players it
tracks. The house reads that page every 30 minutes on launch night and
every 10 minutes after that. The answer is the first valid read in the
hour from the target. The program runs offline, so it forecasts from
the case file alone.

Two facts about the data shape everything below:

* Exophase refreshes its counts on its own schedule, hours apart, not
  continuously. A read shows the rates as of the last refresh before it,
  so the history is flat stretches separated by jumps.
* The tracked base grows fast all week — it is a small self-selected
  slice of a launch selling tens of millions — so a trophy's rate can
  fall even while more people earn it, because newcomers dilute it.

## 2. The forecast

### 2.1 One saturating curve per trophy

An earn rate is cumulative progress: slow at first, steep while the
bulk of players reach the trophy, flat once nearly everyone who will
earn it has. So each trophy is fitted with a logistic curve in hours
since 2026-11-19T00:00:00Z:

```
rate(h) = S / (1 + exp(-(h - h50) / w)),   w = h50 / 3
```

The midpoint `h50` is picked by grid search over 4, 6, 9, 13, 19, 28,
40, 58, 84, 120, 175, 250 and 360 hours. For each candidate the ceiling
`S` is the median of the ceilings each observed point implies, and the
candidate is scored by mean squared logit error plus
`0.15 * (ln(h50 / h50_prior))^2` — a pull toward the midpoint the
trophy's kind suggests. The fit uses one point per distinct displayed
rate, taken at the hour that rate first appeared: between refreshes
Exophase repeats the same numbers, and repeated reads add no signal.

The kind of a trophy is read off its description. Story chapters get
early midpoints (Chapter 1 about 20 h, rising to about 95 h for the
final mission) and a high ceiling (about half of players); named side
missions sit near 70 h with a ceiling near 16%; ordinary activities
near 45 h and 10%; skill tests (gold medals, races won, duels) near
90 h and 7%; counted grinds (250 stunts, card sets, bank vaults,
vehicles found, dollars laundered) near 130 h and 3%; 100% completion
near 170 h and 1.5%; the platinum near 220 h and 0.2%. The priors
matter most where history is thin — the platinum and the grinds — and
fade as a trophy accumulates two days of real reads.

### 2.2 A damped trend, blended in

A second forecast fits a straight line to logit(rate) against
`ln(1 + h)`, halves its slope toward zero, shifts it to pass through
the last observed rate, and caps it at the curve's ceiling. The final
median blends the two on the logit scale, 75% curve and 25% trend:
the curve is the belief, the trend is insurance against a wrong curve
shape.

### 2.3 What trophies require of each other

Medians are adjusted after fitting, because some trophies logically
bound others:

* The platinum needs every other trophy, so its median cannot exceed
  the smallest other median.
* 100% completion needs every non-platinum trophy, so its median
  cannot exceed the smallest of those.
* The story chain (Chapters 1–4, then the final mission) is played in
  order, so its medians are forced non-increasing.

### 2.4 The opening mission

"Just Like Paradise" (Complete Gone Fishing) sat at 100% among the
first 638 tracked players: almost everyone who plays finishes the
first mission. Its median is `100 - min(0.6, days * 0.13)` percent,
where days is the horizon in days — 99.99% over 90 minutes, 99.41%
over the 4.5-day final horizon — with a band that widens with the
horizon, since the share of tracked owners who never play grows as
the base broadens.

### 2.5 How wide the bands are

Each trophy's quantiles come from a logit-normal band around its
median. The logit sigma combines, in quadrature:

* 0.26 base;
* 0.70 × the curve fit's residual standard deviation (capped at 0.8);
* 0.30 × sqrt(horizon in days);
* 0.30 × log10 of the forecast growth in tracked players — the
  dilution uncertainty;
* an extra 0.10 for grinds and 0.15 for the platinum and 100%
  completion;
* 0.50 × the gap between the curve and trend medians on the logit
  scale (capped at 1.2) — when the two models disagree, the band
  widens.

Quantiles use the standard normal quantiles at the case's levels,
rounded to 4 decimals half-up.

### 2.6 Tracked players, for the dilution term only

`ln(players)` is regressed on `ln(1 + h)` and the fitted exponent is
shrunk toward a launch-phase prior of 1.0, with weight
`span / (span + 12 h)` on the data. The prior comes from the disclosed
first refresh: 638 → 3,868 players in about 4.5 hours, i.e.
`(1 + 4.5)^b = 6.06`, so `b ≈ 1.06`; 1.0 is used.

## 3. The launch-night refresh prior

The rules disclose the launch-night sequence: the page was unchanged
(638 players, 943 trophies earned) from 2026-11-18T23:48Z through the
04:00Z read, then refreshed by 04:30Z to 3,868 players and 4,276
trophies earned, with every rate but the one at 100% lower.

Reconstruction: trophy 2 sits at 100% (all 638 earned it), so
non-intro trophies earned were 943 − 638 = 305. After the refresh,
trophy 2 still at 100% accounts for 3,868 of the 4,276, so non-intro
trophies earned were 4,276 − 3,868 = 408. In aggregate, non-intro
rates fell by `(408 / 3,868) / (305 / 638)` = **0.2206**. In cohort
terms the 3,230 new players earned non-intro trophies at a rate `k`
of the old cohort's, where `1 + (6.06 − 1) * k = 408 / 305`, so
`k = 0.067`: the newcomers had earned almost nothing beyond the
intro mission — launch night's first tracked players are the most
hardcore, and everyone after them is hours behind.

When a case's history contains **no refresh at all** and the horizon
is at most 12 hours — exactly the practice shape — the program does
not extrapolate the flat lines. It applies this prior: players
multiply by 6.06, each non-intro trophy's median is its last rate
times `(1 + (6.06 − 1) * k) / 6.06`, with `k` by kind (story and side
0.10, activity and skill 0.08, grind 0.03, platinum and 100% 0.01;
the rate-weighted mean is 0.067, matching the aggregate), and a wide
spread of 0.45 logit units (plus the kind extra, in quadrature),
because the per-trophy fall is uneven. Once the history contains
refreshes, the data takes over and this prior is not used.

## 4. Outside context for the priors (not fitted inputs)

* **GTA V trophy rarity (PSN-wide, long term):** the platinum is
  about 0.1% of all players (about 1.5% on PSNProfiles, whose users
  are self-selected hunters); the first story trophy is about 38.6%;
  mid-story trophies run about 15.6% / 9.9% / 7.0%; 70 gold medals
  is 0.2%. GTA V's platinum takes 150+ hours, over 100 of them in
  GTA Online — GTA VI launches single-player only, so its platinum
  (100% completion, no online) is more attainable than GTA V's, but
  still a multi-day grind nobody finishes in meaningful numbers in
  week one.
* **GTA VI launch scale:** Sensor Tower counted 7.71M digital
  preorders worldwide by Nov 3 and about 12.1M / $1.18B by Nov 13;
  Compound's base case is 14M copies in week one; DFC Intelligence
  put day one near 25M units; Newzoo's week-one range is 37–51M;
  PS5 took about 74% of preorders. So the PS5 player base at Nov 26
  is of order 10–25M people, while Exophase tracks only the small
  self-selected fraction who sync profiles — hence tracked players
  grow fast all week and dilution is a first-order effect, not a
  footnote.

## 5. Local backtests (run before the deadline)

### 5.1 Practice cases (real house data, reconstructed answers)

The kit ships no practice answers. They are reconstructed from the
aggregates the rules disclose, as in section 3: every non-intro
trophy's rate at the target is its last rate times 0.2206, trophy 2
stays 100%. That reconstruction reproduces the house baseline's
stated practice score (the local scorer gives the starter loss 1.402,
typical miss 0.677 pp, 90% coverage 0.0; the rules say about 1.51 /
about 0.68 / about 0), so it is consistent with the true answers up
to per-trophy unevenness.

| Forecast | Loss | Typical miss | 90% coverage |
|---|---|---|---|
| This entry (p00 and p01) | **0.109** | 0.019 pp | 0.98 |
| House baseline (starter) | 1.402 | 0.677 pp | 0.0 |

Sensitivity: if the true per-trophy falls are uneven rather than
uniform — lognormal factors around 0.2206, rescaled to the same
aggregate — the entry's practice loss is 0.26 on average at sigma 0.5
and 0.65 at sigma 1.0 (5 draws each), still well below the baseline's
~1.40. The practice pays nothing; its value is proving the refresh
prior works on real house data.

### 5.2 Synthetic final (pipeline and shape test, known truth)

A synthetic final-like case (119 reads, Nov 18 23:50Z – Nov 21
11:00Z, refreshes every 4 h, players 638 → 24,464 in history →
63,349 at the target; per-trophy logistic progression with
kind-based ceilings; truth = the last refresh before the target).
This tests the full program on final-shaped data, including the
story-chain guard and the platinum.

| Forecast | Loss | Typical miss | 90% coverage |
|---|---|---|---|
| This entry | **0.300** | 1.64 pp | 0.98 |
| House baseline (starter) | 0.630 | 4.24 pp | 0.98 |

Median spot checks (entry vs truth): Ch.1 68.9% vs 68.0%, Ch.2 50.5%
vs 55.0%, final mission 15.2% vs 20.3%, platinum 0.046% vs 0.060%,
100% completion 0.21% vs 0.63%, 250 Stunts 0.29% vs 0.38%.

### 5.3 Determinism and limits

Two runs on the same case write byte-identical outputs. Runtime
about 0.1 s per case (limit 8 s). Output 2,802 bytes per case
(limit 65,536 bytes). Python standard library only; no network; no
randomness.

## 6. Limitations, stated plainly

* The practice answers are a reconstruction, not the house's answers.
  The uniform-fall version is the neutral assumption; the uneven
  versions (loss 0.26–0.63) are the honest range.
* The synthetic final was built with the same logistic family the
  model uses, so its 0.300 is optimistic. The real final's shape
  errors will be larger; the bands are sized for that, and the
  practice range is the better guide to the real loss.
* The kind priors (midpoints, ceilings, first-refresh `k`) are
  judgement calls from GTA V analogues and the one disclosed refresh.
  They matter most where history is thin (the platinum, the grinds);
  the story trophies are driven by their own two days of data.
* The real final case (f00, cut at the deadline) will have about two
  days of reads at 10-minute cadence; this program was tested on the
  practice shape (7–8 reads, flat) and the synthetic final shape
  (119 reads, refresh steps), not on f00 itself, which does not exist
  yet.
* The house model forecast (an AI model at maximum reasoning effort,
  called at the deadline minus 2 h) is the unknown bar alongside the
  two baselines; beating it is not guaranteed by beating the
  baselines.
