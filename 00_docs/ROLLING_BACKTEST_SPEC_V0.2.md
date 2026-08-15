# Rolling Backtest Spec V0.2 — Work9

Re-run the same consecutive rolling-origin methodology using Dataset V012. Default origins remain Jan-2025 through Dec-2025, each predicting H1/H2/H3, with all development targets before Apr-2026. At each origin: FIT uses older known labels; CALIBRATION uses the last 3 known target months ending at the origin; EVALUATION uses future H1/H2/H3 only; evaluation labels never influence fit/early stopping/threshold/gate selection.

Report monthly/horizon metrics, cumulative 3M at Pair/Base SKU/Branch/Portfolio, forecast revisions, and origin stability. CatBoost remains optional/excluded in rolling according to the current contract.
