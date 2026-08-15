# WORK9 — Current Index

## State
- Workspace: CLEAN RERUN
- Stage 01 Dataset V012: **PASS**
- Accepted dataset run: `core_dataset_v012_20260815T122509Z`
- Stage 02 Feature V013: **PASS**
- Accepted feature run: `feature_stage_v013_20260815T123431Z`
- Business zero rule: **known Pair + closed month + no sales row = 0 M²**
- Branch × Month: diagnostic only; Pair is the production forecast grain.
- Work8 remains archive only; no Work8 runtime dependency is allowed.

## Dataset audit summary
- Pair panel: 585,614 rows across 28,605 known Pairs.
- `actual_observed_rows`: 585,614; `missing_unknown_rows`: 0.
- Negative-only rows: 8,528 and remain target-unavailable.
- Positive/negative M² reconciliation: PASS.

## Feature audit summary
- Pair feature rows: 1,186,159.
- Branch feature rows: 3,468.
- Pair train rows: 1,107,295 total; 1,090,317 eligible.
- Pair validation rows: 78,864 total; 77,886 eligible.
- Branch train rows: 3,288 eligible.
- Branch validation rows: 180 eligible.
- Pair/Branch grain uniqueness: PASS.
- Origin-safe information checks: PASS.
- Target-horizon alignment: PASS.
- Negative-only target not trainable: PASS.
- Calendar V013 presence/completeness/range checks: PASS.
- Frozen Test touched: **false**.
- Feature Selection run: false; model training run: false; production published: false.

## Current pointers
- Dataset: `01_config/current_dataset_run.json` → `core_dataset_v012_20260815T122509Z`
- Feature: `01_config/current_feature_run.json` → `feature_stage_v013_20260815T123431Z`
- Feature Selection: not created yet.
- Model candidate: not created yet.
- Rolling backtest: not created yet.
- Diagnosis: not created yet.

## NEXT
Run `03_notebooks/03_feature_selection/03_FEATURE_SELECTION_COLAB_V04_CURRENT_ACTIVE_WORK9.ipynb`.

## Closed until rerun audit
Freeze, new holdout/Frozen Test, reconciliation, production publication.
