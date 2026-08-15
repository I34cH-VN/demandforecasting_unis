from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import pytest
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "02_src" / "modeling" / "soft_two_part_challenger_runner_v01.py"
CONTRACT = ROOT / "01_config" / "soft_two_part_challenger_contract_v01.yaml"
# Local build fallback; canonical Work9 layout uses the paths above.
if not SRC.exists():
    SRC = HERE / "soft_two_part_challenger_runner_v01.py"
if not CONTRACT.exists():
    CONTRACT = HERE / "soft_two_part_challenger_contract_v01.yaml"
spec = importlib.util.spec_from_file_location("c", SRC)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class Metrics:
    @staticmethod
    def wape(y, p):
        y = np.asarray(y, float); p = np.asarray(p, float)
        return float(np.abs(y-p).sum()/y.sum()) if y.sum() > 0 else np.nan
    @staticmethod
    def bias_ratio(y, p):
        y = np.asarray(y, float); p = np.asarray(p, float)
        return float((p.sum()-y.sum())/y.sum()) if y.sum() > 0 else np.nan
    @staticmethod
    def zero_false_positive_rate(y, p, threshold=1e-9):
        y=np.asarray(y,float); p=np.asarray(p,float); z=y==0
        return float((p[z]>threshold).mean()) if z.any() else np.nan


def test_soft_expected_is_probability_times_positive_quantity_no_threshold():
    out = c.soft_expected_demand([0.1, 0.7, 1.2, -1, np.nan], [100, 50, 20, 30, 40])
    assert np.allclose(out, [10, 35, 20, 0, 0])
    assert out[0] > 0  # low probability is NOT hard-thresholded to zero


def test_soft_expected_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        c.soft_expected_demand([0.5], [1, 2])


def _frames():
    keys = {
        "base_sku": ["A", "B"], "branch_code": ["X", "X"],
        "forecast_origin": ["2025-01-01", "2025-01-01"],
        "target_month": ["2025-02-01", "2025-02-01"], "horizon": [1, 1],
        c.PAIR_TARGET: [0.0, 10.0],
    }
    ch = pd.DataFrame(keys)
    ch["pred_soft_two_part_expected"] = [1.0, 8.0]
    ch["p_positive"] = [0.1, 0.8]
    ch["pred_positive_quantity"] = [10.0, 10.0]
    ref = pd.DataFrame(keys)
    ref["pred_lightgbm_tweedie"] = [3.0, 7.0]
    return ch, ref


def test_reference_alignment_is_exact_same_evaluation_rows():
    ch, ref = _frames()
    out = c.align_reference_predictions(ch, ref)
    assert len(out) == 2
    assert np.allclose(out["pred_lightgbm_reference"], [3, 7])


def test_reference_alignment_fails_if_actual_differs():
    ch, ref = _frames()
    ref.loc[0, c.PAIR_TARGET] = 1.0
    with pytest.raises(ValueError, match="actual target mismatch"):
        c.align_reference_predictions(ch, ref)


def test_reference_alignment_fails_on_missing_key():
    ch, ref = _frames()
    ref = ref.iloc[:1].copy()
    with pytest.raises(ValueError, match="row count mismatch"):
        c.align_reference_predictions(ch, ref)


def test_zero_positive_diagnostics_uses_same_rows():
    ch, ref = _frames()
    d = c.align_reference_predictions(ch, ref)
    z = c.zero_positive_diagnostics(d, {
        "lightgbm_reference": "pred_lightgbm_reference",
        "soft_two_part_expected": "pred_soft_two_part_expected",
    }, Metrics)
    overall = z[z["horizon"].isna()].set_index("model")
    assert overall.loc["lightgbm_reference", "zero_forecast_sum_m2"] == 3.0
    assert overall.loc["soft_two_part_expected", "zero_forecast_sum_m2"] == 1.0
    assert overall.loc["soft_two_part_expected", "actual_positive_rows"] == 1


def test_probability_diagnostics_brier_and_positive_quantity():
    ch, _ = _frames()
    p = c.probability_diagnostics(ch)
    overall = p[p["horizon"].isna()].iloc[0]
    assert 0 <= overall["probability_brier"] <= 1
    assert overall["positive_quantity_wape_on_positive_actual"] == 0.0


def test_contract_explicitly_forbids_hard_threshold_and_freeze():
    cfg = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert cfg["challenger"]["hard_zero_threshold_allowed"] is False
    assert cfg["challenger"]["posthoc_global_bias_scaling_allowed"] is False
    assert cfg["decision"]["auto_promote"] is False
    assert cfg["decision"]["auto_freeze"] is False
    assert cfg["safety"]["frozen_test_allowed"] is False
