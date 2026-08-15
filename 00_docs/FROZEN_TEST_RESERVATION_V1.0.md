# WORK9 — Prospective Frozen Test Reservation V1.0

## Purpose
After Model Architecture Freeze V1, Work9 must create a truly prospective forecast vintage **before** the holdout actuals exist in the source.

At the reservation check on 2026-08-15 22:33 +07, `raw.sales_monthly` had maximum month `2026-06-01`, with 0 rows for July 2026 and 0 rows for August 2026. No future quantity values were queried.

Therefore the reserved immutable forecast vintage is:

```text
forecast origin: 2026-06-01
H1: 2026-07-01
H2: 2026-08-01
H3: 2026-09-01
```

## What Stage 06A does
Stage 06A trains the already-frozen V1 architecture using only the accepted Work9 Pair panel through June 2026, then writes predictions for Jul–Aug–Sep 2026.

Champion architecture:

```text
p_positive = P(Y > 0 | X)          [LightGBM binary]
q_positive = E(Y | Y > 0, X)       [LightGBM Tweedie; positive targets only]
forecast   = p_positive × q_positive
```

It does **not** query Supabase, read Jul/Aug/Sep actuals, score the holdout, change the 54 selected features, retune the architecture, apply a hard occurrence threshold, add global bias scaling, round Pair forecasts, reconcile, or publish production forecasts.

## Training policy for the frozen vintage
For each direct H1/H2/H3 model:
- FIT: all target-available historical eligible rows before Apr 2026;
- CAL: Apr–Jun 2026 target months, current-active known-Pair universe only, used only for early-stopping / iteration selection;
- final refit: FIT + CAL under the already-frozen architecture;
- future forecast universe at origin Jun 2026: known Pair × current-active Base SKU × current-active Branch.

Apr–Jun 2026 may be used for final fitting because architecture/features/model family are already frozen; they may not be used to revise the frozen V1 architecture.

## Holdout purity
The Jul–Aug–Sep forecast file and six fitted component models are hashed and stored as an immutable vintage.

The future actual labels must not be used to alter V1 architecture. The formal Frozen Test score should be computed only when all three reserved months are closed/loaded, using exactly the saved June-origin prediction file.

Do not replace the June-origin prediction with a later reforecast when evaluating this reservation.

## Evaluation levels
When Stage 06B is eventually opened, score the same forecasted Pair universe at:
- Pair × month and cumulative Pair 3M;
- Base SKU cumulative 3M;
- Branch cumulative 3M;
- Portfolio cumulative 3M;
- H1/H2/H3 bias and WAPE;
- zero-month and positive-demand diagnostics.

No unseen/new Pair actual may be silently added to the model-accuracy actual denominator. Demand coverage from unseen Pairs, if reported, must be a separate business coverage metric.

## Gate
After 06A PASS:

```text
architecture: FROZEN V1
forecast vintage: LOCKED
Frozen Test evaluation: PENDING ACTUALS
reconciliation: CLOSED
production publication: CLOSED
```
