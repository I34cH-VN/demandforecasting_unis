# Dataset Build Spec V1.2 — Work9

## Grain
1 row = Base SKU × Branch × Month for each **known Pair** from its first valid M² observation through the closed panel end month.

## Authoritative mapping
Bravo SKU → Base SKU comes only from `raw.master_sku.bravo_sku -> raw.master_sku.base_sku`. No suffix/prefix/name/factory/sale/replacement inference.

## Gross-positive target
For valid M² source rows, aggregate positive quantities into `gross_positive_m2`; retain negatives separately for audit.

## Zero semantics V012
For a known Pair in a closed month:
- positive/mixed source observation -> gross-positive target;
- explicit zero-only source observation -> `0`, `OBSERVED_ZERO`;
- negative-only source observation -> target unavailable, `NEGATIVE_ONLY_GROSS_ZERO`;
- **no sales/source row -> `0`, `target_available=true`, `OBSERVED_ZERO_IMPLICIT`**.

`source_row_observed` is preserved so implicit zeros remain distinguishable from explicit source rows.

## Closed month
The Colab builder uses the maximum loaded M² month as `panel_end`. The project assumes loaded months through `panel_end` are closed/complete under the user-approved business rule. A source-load failure must be treated as a data-quality incident, not as a normal zero.

## Branch panel
Branch × Month may be built as an aggregate diagnostic artifact, but it is not an independent Work9 production forecast target.

## Safety
Supabase is SELECT-only. Current status does not rewrite history. Negative-only targets are not silently converted to zero.
