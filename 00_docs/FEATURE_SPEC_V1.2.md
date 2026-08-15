# Feature Spec V1.2 — Work9

Input dataset: `dataset_v012`. Pair feature code remains `pair_feature_v013`; branch diagnostic feature code remains `branch_feature_v013`.

All features must be origin-safe (`information_month <= forecast_origin`). Pair history now contains implicit observed zeros for known Pair closed months, so lag/rolling/ADI/CV² features are recomputed from V012 rather than reused from Work8. Calendar/Tết/public-holiday features remain deterministic target-calendar features. Current status and snapshot-risk fields are blocked as predictors.
