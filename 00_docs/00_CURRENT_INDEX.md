# WORK9 — Current Index

## State
- Workspace: CLEAN RERUN
- Stage 01 Dataset V012: **PASS** — `core_dataset_v012_20260815T122509Z`
- Stage 02 Feature V013: **PASS** — `feature_stage_v013_20260815T123431Z`
- Stage 03 Feature Selection V04: **PASS** — `feature_selection_v04_20260815T130048Z`
- Stage 04 Pair Modeling V02: **PASS** — `pair_modeling_v02_20260815T130724Z`
- Stage 05 Rolling Backtest V01: **PASS** — `rolling_backtest_v01_20260815T132319Z`
- Stage 05B Underforecast Diagnosis V01: **PASS** — `underforecast_diagnosis_v01_20260815T140720Z`
- Stage 05C Soft Two-Part Challenger V01: **PASS** — `soft_two_part_challenger_v01_20260815T145347Z`
- Model Architecture Freeze V01: **APPROVED** — `model_architecture_freeze_v01_20260815T152950Z`
- Stage 06A Prospective Frozen Forecast V01: **PASS / LOCKED PENDING ACTUALS** — `prospective_frozen_forecast_v01_20260815T155226Z`
- Champion V1: **`soft_two_part_expected`**.
- Reference/fallback: `lightgbm_tweedie`.

## Frozen holdout reservation
Accepted immutable forecast vintage:

```text
forecast origin = 2026-06-01
H1 = 2026-07-01
H2 = 2026-08-01
H3 = 2026-09-01
```

Post-lock SELECT-only source verification still found:

```text
raw.sales_monthly max month = 2026-06-01
Jul-2026 rows = 0
Aug-2026 rows = 0
Sep-2026 rows = 0
```

Therefore the prospective forecast was locked before the reserved holdout labels existed in the source.

## Stage 06A accepted evidence
- 17,253 unique Pair; 1,447 Base SKU; 58 Branch.
- 51,759 Pair-horizon rows.
- 54 frozen selected Pair features.
- Prediction SHA256: `3dd59e503f3bf59eaa542222a20995aca9af61732eaecb4702e61cb1161f33dc`.
- All six model binaries independently hash-verified against the manifest.
- Forecast file has no actual label columns; all rows are known Pair + current production mask; no duplicate/NaN/negative predictions.
- FIT target max = Mar-2026; CAL = Apr–Jun 2026 only.
- 06A safety PASS: no Supabase access, no future actual read, no holdout scoring, no architecture change, no hard threshold, no bias scaling, no Pair rounding, no production publish.

Frozen volumes:
- Jul H1: 469,830.323620 M².
- Aug H2: 396,152.269628 M².
- Sep H3: 318,297.385404 M².

Audit document: `00_docs/06A_PROSPECTIVE_FROZEN_FORECAST_AUDIT_V1.0.md`.

## Locked V1 business/model rules
- Grain: Base SKU × Branch × Month.
- Target: gross-positive M².
- Known Pair + closed month + no source row = observed implicit zero.
- Negative-only month is target-unavailable.
- Dense history starts at first valid M² observation; no pre-known zero backfill.
- Production universe = known Pair × current-active Base SKU × current-active Branch.
- Current status is mask only, never predictor.
- Soft Two-Part direct H1/H2/H3; no hard threshold, global bias scaling, or Pair-level rounding.

## Current pointers
- Dataset → `01_config/current_dataset_run.json`
- Feature → `01_config/current_feature_run.json`
- Feature Selection → `01_config/current_feature_selection_run.json`
- Model candidate → `01_config/current_model_candidate_run.json`
- Rolling backtest → `01_config/current_rolling_backtest_run.json`
- Diagnosis → `01_config/current_underforecast_diagnosis_run.json`
- 05C challenger → `01_config/current_soft_two_part_challenger_run.json`
- Architecture freeze → `01_config/current_model_architecture_freeze.json`
- Prospective frozen forecast → `01_config/current_prospective_frozen_forecast_run.json`

## NEXT
**WAIT FOR ACTUALS.** Do not rerun or replace the accepted June-origin frozen vintage.

06B Frozen Test scoring may be created/run only after Jul-2026, Aug-2026, and Sep-2026 are all closed/loaded. It must score the exact prediction SHA above without tuning from holdout labels.

## Closed
06B scoring until all three actual months are closed/loaded; reconciliation and production publication remain closed.
