# XAUUSD PPO Signal Guide

## Recommended models

- Stable baseline: `train/ppo_xauusd_best.zip`
- Higher-return candidate with risk overlay: `train/ppo_xauusd_aggressive.zip`

The current preferred signal path is the aggressive model with the DXY momentum
risk filter:

```bash
python scripts/generate_signal.py
```

This is equivalent to:

```bash
python scripts/generate_signal.py \
  --model train/ppo_xauusd_aggressive.zip \
  --risk-filter dxy_mom20_q75_train
```

## Signal output

The script prints:

- the latest feature timestamp and close price
- the raw PPO signal
- whether the DXY risk filter is active
- the final signal after the risk overlay

To write a machine-readable signal file:

```bash
python scripts/generate_signal.py \
  --json \
  --output train/latest_signal.json
```

If the current live position is already long, pass it explicitly so the PPO
observation includes the correct position state:

```bash
python scripts/generate_signal.py --current-position long
```

## Risk filter

`dxy_mom20_q75_train` computes the 75th percentile of `dxy_mom_20` using only
data before `2022-01-01`. If the latest `dxy_mom_20` is above that threshold,
the final signal is forced to `FLAT`.

This filter is intended for the aggressive d12 model. It reduced the aggressive
model's validation drawdown and improved the worst walk-forward year in the
2018-2025 checks.

## Verification commands

Stable baseline:

```bash
python evaluate_model.py \
  --kind ppo \
  --model train/ppo_xauusd_best.zip \
  --period walk-forward \
  --walk-start-year 2018 \
  --walk-end-year 2025 \
  --device cpu
```

Filtered aggressive:

```bash
python evaluate_model.py \
  --kind ppo \
  --model train/ppo_xauusd_aggressive.zip \
  --period walk-forward \
  --walk-start-year 2018 \
  --walk-end-year 2025 \
  --risk-filter dxy_mom20_q75_train \
  --device cpu
```

