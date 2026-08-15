# Underforecast Diagnosis Spec V0.2 — Work9

Diagnosis runs only after the fresh Work9 rolling backtest PASS. It reads `current_rolling_backtest_run.json`; no Work8 run ID is accepted or hard-coded. Diagnose error attribution across actual zero, first positive known Pair, reactivation gaps, spikes, and ongoing positive demand. Rounding remains diagnostic only; canonical target/prediction stays decimal M². No retraining/calibration/freeze/production occurs in this stage.
