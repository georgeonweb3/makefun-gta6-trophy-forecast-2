# GTA VI trophy rarity forecast

Entry for make.fun `gta6-trophy-rarity-forecast-01` ($250): forecast
every GTA VI PlayStation 5 trophy's earn rate on Exophase at
2026-11-26T00:00:00Z, as five quantiles per trophy, from the reads the
house made before the deadline.

## How it works

* Each trophy gets a saturating logistic curve in hours since release,
  fitted by grid search over the midpoint with a pull toward a prior
  set by the trophy's kind (story chapter, side mission, activity,
  skill test, grind, 100% completion, platinum).
* A damped log-time trend is blended in at 25% as insurance against a
  wrong curve shape.
* When a case's history shows no Exophase refresh yet and the horizon
  is short, a launch-night refresh prior is applied instead of
  extrapolating flat lines: the player base jumps ~6x and the newcomers
  have earned almost nothing, so non-intro rates fall by roughly a
  factor of five.
* Medians respect what trophies require of each other (platinum last,
  100% completion second-last, story in order), and each trophy gets a
  logit-normal band sized by fit error, horizon, player-base growth
  and model disagreement.

Full reasoning, calibration and backtests: [METHOD.md](METHOD.md).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in`
in the input directory and writes one `<id>.out` per case.

## Files

* `main.py` — the entrypoint (`python3 /bot/main.py`).
* `METHOD.md` — method, calibration, assumptions, sources, backtests.
* `LICENSE` — MIT.
