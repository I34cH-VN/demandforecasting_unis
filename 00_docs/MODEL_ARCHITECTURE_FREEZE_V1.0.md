# WORK9 — Model Architecture Freeze V1.0

## Decision
Status: **APPROVED — ARCHITECTURE ONLY**

User-approved champion for Work9 V1:

```text
soft_two_part_expected
= P(Y > 0 | X) × E(Y | Y > 0, X)
```

Reference/fallback model:

```text
lightgbm_tweedie
```

This freeze locks the modeling architecture and evaluation policy. It does **not** freeze one fitted binary model, open a Frozen Test, perform reconciliation, or publish production forecasts.

## Locked lineage
- Dataset: `dataset_v012` — accepted run `core_dataset_v012_20260815T122509Z`.
- Pair features: `pair_feature_v013` — accepted run `feature_stage_v013_20260815T123431Z`.
- Feature Selection: `feature_selection_v04` — accepted run `feature_selection_v04_20260815T130048Z`.
- Selected Pair features: **54**.
- Reference model study: `pair_modeling_v02_20260815T130724Z`.
- Rolling backtest: `rolling_backtest_v01_20260815T132319Z`.
- Underforecast diagnosis: `underforecast_diagnosis_v01_20260815T140720Z`.
- Soft Two-Part challenger: `soft_two_part_challenger_v01_20260815T145347Z`.

## Locked business universe
Production forecast grain remains:

```text
Base SKU × Branch × Month
```

Production universe:
- known Pair as of forecast origin;
- Base SKU current ACTIVE;
- Branch current ACTIVE;
- unseen Pair is not forecast;
- current status is a universe/publication mask only and is never a model predictor.

Training keeps all historical eligible rows, including entities that are currently inactive.

Branch × Month remains an aggregation/diagnostic view only; no independent Branch production model is introduced by this freeze.

## Locked target semantics
- Target: gross-positive M².
- Known Pair + closed month + no source/sales row = observed implicit zero, target available.
- Explicit zero remains observed zero.
- Negative-only month remains target-unavailable and is not converted to a trainable zero.
- Dense Pair history begins at the Pair's first valid M² observation; no pre-known zero backfill.
- Canonical prediction stays decimal M²; no Pair-level ceil/rounding.

## Locked architecture
For each direct horizon H1/H2/H3:

1. Occurrence component:
   `P(Y > 0 | X)` using LightGBM binary classification.
2. Positive-quantity component:
   `E(Y | Y > 0, X)` using LightGBM Tweedie on positive targets only.
3. Final expected-demand forecast:
   `p_positive × q_positive`.

Rules:
- separate direct models for H1, H2 and H3;
- same 54 selected Pair features;
- no hard probability threshold;
- no global post-hoc bias multiplier;
- no Pair-level rounding;
- origin-safe features only;
- early-stopping/iteration choice may use trailing calibration history only;
- evaluation/Frozen-Test labels may never tune thresholds, iterations, scaling, features, or architecture.

## Monthly retrain policy
The architecture is frozen, not one month's fitted weights.

At closed month T:
1. refresh data through T;
2. rebuild origin-safe features;
3. retrain H1/H2/H3 occurrence + positive-quantity components using eligible history available at T;
4. forecast T+1/T+2/T+3;
5. save immutable forecast vintage;
6. repeat next month with expanding history.

## Evidence used for approval
Stage 05C compared the champion against the accepted direct LightGBM reference on exactly the same 420,846 rolling-primary Pair-month rows across 12 origins with feature parity PASS.

Complete 3M WAPE:
- Pair: 0.77771 → **0.76884**.
- Base SKU: 0.44758 → **0.44044**.
- Branch: 0.31000 → **0.29134**.
- Portfolio: 0.19332 → **0.18501**.
- 3M bias ratio: -18.85% → **-17.21%**.

Monthly WAPE improved from 1.00972 to **1.00309** and monthly bias from -19.19% to **-17.69%**. Soft Two-Part won 7/12 rolling origins versus 5/12 for the reference.

## Known limitation retained in V1
The architecture does not solve zero-month allocation completely:
- actual-zero forecast volume is essentially unchanged versus direct LightGBM;
- positive-demand conditional bias remains materially negative;
- exact-zero forecasts are not created because this architecture intentionally avoids a hard occurrence threshold.

This limitation is accepted for V1 because the champion improves the primary 3M business-planning metrics and aggregate bias while keeping a probabilistic expected-demand interpretation. It remains a monitored research item, not a reason to alter the frozen V1 architecture without a new versioned study.

## Change control
Any change to one of the following requires a new architecture version and a new origin-honest backtest before promotion:
- target/zero semantics;
- production universe;
- selected feature set;
- Soft Two-Part formula;
- component model families/objectives;
- hard thresholding;
- post-hoc bias scaling;
- direct H1/H2/H3 structure;
- evaluation universe or metric priority.

## Frozen Test policy
Frozen Test remains **CLOSED** after this architecture decision.

A true Frozen Test must use a period whose actual labels were not inspected, used for feature/model selection, architecture choice, calibration, or prior recovery analysis. No existing legacy Work8 holdout is automatically reclassified as untouched.
