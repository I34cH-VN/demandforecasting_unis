# WORK9 — Demand Forecasting Clean Rerun

`work9` is the clean successor of `work8`. `work8` remains archive/lineage only and must not be used as a runtime dependency.

## Locked business rules
- Primary grain: **Base SKU × Branch × Month**.
- Branch × Month: diagnostic/aggregation view only; no independent production model in Work9 V1.
- Target: **gross-positive M²**.
- Bravo SKU → Base SKU: master mapping only; never infer from suffix/prefix/name.
- Known Pair + closed month + no sales record => **0 M² observed demand**, `target_available=true`, `zero_semantics=OBSERVED_ZERO_IMPLICIT`.
- Negative-only month remains return/adjustment evidence and is not a trainable zero target.
- Dense Pair history starts at first valid M² observation; do not zero-fill months before the Pair is known.
- Unseen Pair: no forecast.
- Current SKU/Branch status: publication/universe mask only, never a historical predictor.
- Monthly production concept: expanding-window retrain; at month T forecast T+1/T+2/T+3 and preserve every forecast vintage.

## V1 model architecture — FROZEN
Champion:

```text
soft_two_part_expected
= P(Y>0|X) × E(Y|Y>0,X)
```

- Occurrence: LightGBM binary.
- Positive quantity: LightGBM Tweedie.
- Direct H1/H2/H3.
- Same 54 Pair features from Feature Selection V04.
- No hard threshold, global bias scaling, or Pair-level rounding.
- `lightgbm_tweedie` remains the reference/fallback model.

Architecture freeze: `model_architecture_freeze_v01_20260815T152950Z`.

## Prospective Frozen Test vintage — LOCKED
Stage 06A accepted run:
`prospective_frozen_forecast_v01_20260815T155226Z`.

Reserved vintage:
- Origin Jun-2026.
- H1 Jul-2026, H2 Aug-2026, H3 Sep-2026.
- 17,253 known/current-active Pair; 51,759 Pair-horizon rows.
- Prediction SHA256: `3dd59e503f3bf59eaa542222a20995aca9af61732eaecb4702e61cb1161f33dc`.
- Post-lock source check: raw sales still max Jun-2026; Jul/Aug/Sep row counts all zero.

This forecast is immutable. Do not rerun 06A later and replace it.

## Completed run order
01 Dataset → 02 Features → 03 Feature Selection → 04 Pair Model → 05 Rolling → 05B Diagnosis → 05C Challenger → Architecture Freeze → 06A Prospective Forecast Lock.

## Next
Wait until Jul/Aug/Sep 2026 are all closed/loaded, then perform 06B Frozen Test scoring using the exact locked prediction SHA. No holdout-driven tuning is allowed.

Reconciliation and production publication remain closed.
