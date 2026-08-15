# WORK9 — 06A Prospective Frozen Forecast Audit V1.0

## Decision
Status: **PASS — IMMUTABLE FORECAST LOCKED, PENDING ACTUALS**

Accepted run:
`prospective_frozen_forecast_v01_20260815T155226Z`

Architecture freeze:
`model_architecture_freeze_v01_20260815T152950Z`

Champion:
`soft_two_part_expected = P(Y>0|X) × E(Y|Y>0,X)`

## Reserved prospective holdout
- Forecast origin: `2026-06-01`.
- H1 target: `2026-07-01`.
- H2 target: `2026-08-01`.
- H3 target: `2026-09-01`.
- Evaluation status: `LOCKED_PENDING_ACTUALS`.

A post-lock SELECT-only source check on 2026-08-15 confirmed `raw.sales_monthly` still had max month `2026-06-01` and zero rows for Jul/Aug/Sep 2026. Therefore the forecast was locked before holdout labels existed in the source.

## Frozen forecast universe
- 17,253 unique known/current-active Pair at origin.
- 1,447 Base SKU.
- 58 Branch.
- 51,759 Pair-horizon rows = 17,253 × 3 horizons.
- Every saved row has `known_pair_asof_origin=true` and `current_production_forecast_mask=true`.
- No duplicate Pair × origin × target month × horizon rows.
- Prediction file contains no actual/target label column.
- No NaN forecast; no negative forecast.

## Forecast volume locked
- H1 / Jul-2026: 469,830.323620 M²; mean p_positive 0.299377.
- H2 / Aug-2026: 396,152.269628 M²; mean p_positive 0.262922.
- H3 / Sep-2026: 318,297.385404 M²; mean p_positive 0.236269.

These are frozen outputs, not accuracy scores.

## Training / calibration cutoffs
Iteration selection uses only known historical labels:
- FIT target max: `2026-03-01` for H1/H2/H3.
- CAL target range: `2026-04-01` through `2026-06-01`.
- Final refit then produces the June-origin Jul/Aug/Sep forecasts.

Selected iterations:
- H1 occurrence 139; positive quantity 184.
- H2 occurrence 108; positive quantity 148.
- H3 occurrence 95; positive quantity 143.

All six LightGBM components ran on CPU in this run. Device choice affects runtime only; the architecture and evaluation protocol are unchanged.

## Immutable artifact hashes
Prediction parquet:
`3dd59e503f3bf59eaa542222a20995aca9af61732eaecb4702e61cb1161f33dc`

Prediction CSV:
`9d768aefef68fba29b1ab414a2d6be5941bbfeb11f36361aa3b099acebb62c02`

Models:
- occurrence_h1_lightgbm.txt: `7099c6d991dbd9a8b7eb03f8bb0030f20df8241516b312419c99d0f5a5e91693`
- occurrence_h2_lightgbm.txt: `be67fc97e2f1100c646153287696e857dc05a991950d7f98a369651074690183`
- occurrence_h3_lightgbm.txt: `47e84c9ff38235973e5bc173e96a242985c2020235886fbdeb088382ec08cc46`
- positive_quantity_h1_lightgbm_tweedie.txt: `151774aad2a85991a2d1579f568c642e7c5f352b916c753af3e80a5f3a066bbb`
- positive_quantity_h2_lightgbm_tweedie.txt: `5ccd2c552c1a1aa931cfad2f6b258a250e6931e4eb2ab5ea9b62d1523295942d`
- positive_quantity_h3_lightgbm_tweedie.txt: `a0d7a3c363ca1dfb470c587a6e21eee327e2e179d27c03b971b8856e19d60137`

The prediction SHA and all six model SHA values were independently recomputed from the Drive files during the post-run audit and match the manifest exactly.

## Safety audit
PASS:
- Supabase accessed by 06A: false.
- Future actual labels read: false.
- Future actual labels used for fit/calibration: false.
- Pair panel max month equals forecast origin: true.
- Production universe current-active known Pair only: true.
- Hard occurrence threshold: false.
- Global post-hoc bias scaling: false.
- Pair-level rounding: false.
- Architecture changed after freeze: false.
- Frozen Test evaluation run: false.
- Production published: false.

## Change control
Do **not** overwrite, edit, delete, or replace the accepted June-origin forecast/model artifacts.
Do **not** rerun 06A later and substitute a newer forecast for this Frozen Test vintage.

Formal 06B scoring may start only after Jul-2026, Aug-2026, and Sep-2026 are all confirmed closed/loaded. 06B must join those actuals to this exact frozen prediction SHA and must not tune features, model parameters, thresholds, scaling, or architecture from the holdout labels.

Until then: Frozen Test scoring, reconciliation, and production publication remain closed.
