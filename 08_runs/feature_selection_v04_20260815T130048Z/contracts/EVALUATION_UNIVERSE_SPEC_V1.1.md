# Evaluation Universe Spec V1.1 — Work9

TRAIN uses all historical eligible rows and does not filter by current status. PRIMARY validation uses current-active Base SKU × current-active Branch × known Pair with target available. Production output uses known Pair + current-active Base + current-active Branch. Unseen Pair is not forecast. Current status is a universe/publication mask only.

Dataset V012 defines a known Pair with no sales row in a closed month as an observed zero, therefore such rows are target-available and must participate in training/evaluation. Negative-only months remain unavailable for the gross-positive target.
