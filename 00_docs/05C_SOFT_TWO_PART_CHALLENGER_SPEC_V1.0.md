# WORK9 — 05C Soft Two-Part Expected-Demand Challenger V1.0

## Purpose
05C is the final targeted research challenger after Stage 05B. It addresses the zero-heavy Pair-month target exposed by Dataset V012 without changing dataset semantics, selected features, evaluation universe, or the untouched Frozen Test policy.

## Locked formula
For each direct horizon H1/H2/H3:

```text
p_positive = P(Y > 0 | X)
q_positive = E(Y | Y > 0, X)
forecast   = p_positive × q_positive
```

There is **no hard probability threshold**. A low but non-zero occurrence probability remains a low expected-demand forecast rather than being forced to zero.

## Training policy
- Same 54 Pair features from Feature Selection V04.
- Same rolling origins and FIT/CAL/EVAL temporal split as Rolling Backtest V01.
- Occurrence component trains on all target-available historical rows with binary label `Y > 0`.
- Positive-quantity component trains only on rows with `Y > 0`.
- Early stopping uses the trailing calibration period only.
- After stopping iteration is selected, each component is refit on FIT + CALIBRATION and then predicts EVALUATION.
- Evaluation labels are never used to fit, select stopping iteration, scale predictions, or choose a threshold.

## Reference comparison
The reference is the already accepted Stage 05 `lightgbm_tweedie` prediction. 05C must match exactly the same evaluation keys:

```text
base_sku × branch_code × forecast_origin × target_month × horizon
```

Actual values must also match exactly. Base SKU / Branch / Portfolio 3M metrics are roll-ups of this same Pair forecast universe only.

## Required outputs
- Pair-month same-row predictions: reference LightGBM + occurrence probability + conditional positive quantity + soft expected demand.
- Monthly/H1/H2/H3 metrics.
- Zero-month forecast volume and positive-demand WAPE/bias.
- Probability/Brier diagnostics.
- Complete H1+H2+H3 cumulative metrics at Pair, Base SKU, Branch, Portfolio.
- H2→H1 and H3→H2 forecast revision stability.
- Origin audits and actual compute device.
- Decision review JSON.

## Decision policy
05C is research-only. It does not auto-promote itself even if it wins metrics. After PASS, audit must compare:

1. Base SKU 3M WAPE.
2. Branch 3M WAPE.
3. Pair 3M WAPE.
4. Portfolio absolute bias.
5. Forecast volume placed on actual-zero months.
6. WAPE/bias on actual-positive months.
7. Forecast revision stability.

Only after that audit may the user approve a model architecture freeze. A later untouched period is still required for a true Frozen Test.

## Safety
05C must not access Supabase, touch the Frozen Test period, perform reconciliation, write production outputs, use current status as a model feature, apply Pair-level ceil, or apply global post-hoc bias scaling.
