# Work9 Pipeline Run Order

Run one notebook at a time and only continue after PASS/pointer creation.

1. `01_dataset/01_BUILD_CORE_DATASETS_COLAB_V012_WORK9.ipynb` — PASS
2. `02_features/02_BUILD_FEATURES_COLAB_V013_CALENDAR_WORK9.ipynb` — PASS
3. `03_feature_selection/03_FEATURE_SELECTION_COLAB_V04_CURRENT_ACTIVE_WORK9.ipynb` — PASS
4. `04_modeling/04_TRAIN_PAIR_MODELS_COLAB_V02_CORRECTED_3M_GPU_WORK9.ipynb` — PASS
5. `05_backtest/05_ROLLING_ORIGIN_BACKTEST_COLAB_V01_GPU_WORK9.ipynb` — PASS
6. `05_backtest/05B_UNDERFORECAST_DIAGNOSIS_COLAB_V01_WORK9.ipynb` — PASS
7. `05_backtest/05C_SOFT_TWO_PART_CHALLENGER_COLAB_V01_GPU_WORK9.ipynb` — PASS
8. Architecture Freeze V01 — APPROVED; champion `soft_two_part_expected`.
9. `06_frozen_test/06A_LOCK_PROSPECTIVE_FROZEN_FORECAST_COLAB_V01_GPU_WORK9.ipynb` — PASS / IMMUTABLE VINTAGE LOCKED.
10. 06B Frozen Test scoring — **WAIT** until Jul/Aug/Sep 2026 are all closed/loaded.

## Locked prospective vintage
- Origin: 2026-06-01.
- H1/H2/H3: Jul/Aug/Sep 2026.
- Prediction SHA256: `3dd59e503f3bf59eaa542222a20995aca9af61732eaecb4702e61cb1161f33dc`.
- Do not rerun 06A later and replace this accepted vintage.

## 06B rule
06B must score only the exact accepted frozen forecast against actuals after all three target months are confirmed closed/loaded. Holdout labels may not alter features, thresholds, iterations, scaling, model family, architecture, or forecast values.

Until then, Frozen Test scoring, reconciliation and production publication remain closed.
