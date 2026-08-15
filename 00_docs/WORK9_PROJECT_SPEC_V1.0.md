# WORK9 Project Spec V1.0

## Purpose
Clean rerun of the Base SKU × Branch × Month demand forecasting pipeline after correcting zero-demand semantics. Work8 is archive only.

## Target and data semantics
- Unit: M².
- `q > 0`: contributes to gross-positive demand.
- `q = 0`: explicit observed zero.
- `q < 0`: return/adjustment audit; excluded from gross-positive target.
- **Known Pair + closed month + no sales row: implicit observed zero (`y=0`)**.
- Source month/load failure is not silently converted to zero; it is a data-quality failure to investigate.

## Universe
- TRAIN: all structurally eligible historical observations, including entities inactive today.
- Validation/model selection: current-active Base SKU × current-active Branch × known Pair, with target available.
- Production: known Pair AND Base current active AND Branch current active.
- Current status is never a feature.

## Forecast scope
- Production forecast grain: Base SKU × Branch × Month.
- Branch × Month is retained for aggregation/diagnostics only; `branch_modeling=false`.

## Forecast architecture
Current candidate architecture remains direct H1/H2/H3 with LightGBM Tweedie as the main candidate, while Stage 04 reruns the accepted candidate comparison under Dataset V012. No global bias multiplier is pre-approved. Canonical forecast remains decimal M². Pair-level ceil is forbidden.

## Monthly retrain
At closed month T:
1. refresh Dataset V012 through T;
2. rebuild origin-safe features;
3. retrain using all eligible known labels through T;
4. forecast T+1, T+2, T+3;
5. save immutable forecast vintage;
6. next month repeat with the new actual.

Example: Mar -> Apr/May/Jun; Apr actual closes -> retrain -> May/Jun/Jul; May closes -> retrain -> Jun/Jul/Aug.

## Clean-run stages
01 Dataset V012 -> 02 Feature V013 -> 03 Feature Selection V04 -> 04 Pair Model V02 -> 05 Rolling Backtest V01 -> 05B Underforecast Diagnosis V01.

Freeze/Frozen-Test/reconciliation/production are intentionally not copied from Work8 because the target semantics changed. They require a new post-rerun decision.
