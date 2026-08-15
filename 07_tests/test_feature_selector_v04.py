import numpy as np
import pandas as pd

from pathlib import Path
import sys

# Ensure pytest can import the Stage-6 module even when run in a fresh subprocess.
MODULE_DIR = Path(__file__).resolve().parents[1] / '02_src' / 'feature_selection'
if not MODULE_DIR.exists():
    MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from feature_selector_v04 import (
    SelectionConfig,
    static_screen,
    exact_duplicate_map,
    choose_features,
    validate_selection_result,
    wape,
)


def test_wape_zero_error():
    assert wape(np.array([1., 2.]), np.array([1., 2.])) == 0.0


def test_static_screen_drops_constant_and_high_missing():
    df = pd.DataFrame({
        "good": [1., 2., 3., 4.],
        "const": [1., 1., 1., 1.],
        "sparse": [np.nan, np.nan, np.nan, 1.],
    })
    cfg = SelectionConfig(missing_rate_drop=0.70)
    kept, report, dup = static_screen(df, ["good", "const", "sparse"], cfg)
    assert kept == ["good"]
    s = dict(zip(report.feature_name, report.static_status))
    assert s["const"] == "DROP_CONSTANT"
    assert s["sparse"] == "DROP_MISSING_RATE"


def test_exact_duplicate_map_detects_same_series():
    df = pd.DataFrame({"a": [1., np.nan, 3.], "b": [1., np.nan, 3.], "c": [1., 2., 3.]})
    dup = exact_duplicate_map(df, ["a", "b", "c"])
    assert dup == {"b": "a"}


def test_choose_features_keeps_horizon_for_pooled():
    cfg = SelectionConfig(min_perm_wape_gain=0.01, pair_min_selected=1)
    perm = pd.DataFrame([
        {"scope": "ALL", "feature_name": "x", "importance_wape_mean": 0.02, "importance_wape_std": 0.0},
        {"scope": "ALL", "feature_name": "horizon", "importance_wape_mean": -0.01, "importance_wape_std": 0.0},
        {"scope": "h1", "feature_name": "x", "importance_wape_mean": 0.02, "importance_wape_std": 0.0},
        {"scope": "h1", "feature_name": "horizon", "importance_wape_mean": 0.0, "importance_wape_std": 0.0},
        {"scope": "h2", "feature_name": "x", "importance_wape_mean": 0.0, "importance_wape_std": 0.0},
        {"scope": "h2", "feature_name": "horizon", "importance_wape_mean": 0.0, "importance_wape_std": 0.0},
        {"scope": "h3", "feature_name": "x", "importance_wape_mean": 0.0, "importance_wape_std": 0.0},
        {"scope": "h3", "feature_name": "horizon", "importance_wape_mean": 0.0, "importance_wape_std": 0.0},
    ])
    selected, decision = choose_features(["x", "horizon"], perm, {"calendar": ("horizon",)}, cfg, 1)
    assert "x" in selected and "horizon" in selected
    assert decision.loc[decision.feature_name.eq("horizon"), "selection_reason"].iloc[0] == "MANDATORY_FOR_POOLED"


def test_validate_selection_blocks_dropped_features():
    result = {
        "selected": ["x", "horizon"],
        "static_kept": ["x", "horizon"],
        "quality_report": pd.DataFrame({
            "feature_name": ["x", "horizon"],
            "selected": [True, True],
            "static_status": ["PASS_STATIC", "PASS_STATIC"],
        }),
        "full_metrics": {"wape": 0.5},
        "validation_rows_primary_current_active": 10,
        "validation_rows_secondary_all": 12,
        "candidate_features": ["x", "horizon"] + sorted(__import__("feature_selector_v04").CALENDAR_V013_CANDIDATES),
    }
    v = validate_selection_result(result, "PAIR", 1)
    assert v["status"] == "PASS"

def test_select_track_small_integration():
    from feature_selector_v04 import select_track, PAIR_FAMILIES
    rng = np.random.default_rng(42)
    n_train, n_val = 180, 60
    n = n_train + n_val
    horizon = np.tile([1,2,3], n//3 + 1)[:n]
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = np.maximum(10 + 4*x1 + rng.normal(scale=0.5, size=n), 0)
    df = pd.DataFrame({
        'pair_lag_1': x1,
        'pair_roll_3_mean': x2,
        'tile_size_code': np.where(np.arange(n)%2==0, '3060', '6060'),
        'horizon': horizon,
        'target_is_tet_month': np.isin(np.arange(n) % 12, [1]),
        'target_is_pre_tet_month': np.isin(np.arange(n) % 12, [0]),
        'target_is_post_tet_month': np.isin(np.arange(n) % 12, [2]),
        'target_actual_gross_m2': y,
        'target_available': True,
        'historical_train_mask': [True]*n_train + [False]*n_val,
        'official_validation_mask': [False]*n_train + [True]*n_val,
        'current_production_forecast_mask': [False]*n_train + [True]*n_val,
        'known_pair_asof_origin': [True]*n,
        'target_month': pd.to_datetime(['2025-12-01']*n_train + ['2026-01-01']*n_val),
    })
    inventory = pd.DataFrame({
        'track': ['PAIR']*7,
        'feature_name': ['pair_lag_1','pair_roll_3_mean','tile_size_code','horizon','target_is_tet_month','target_is_pre_tet_month','target_is_post_tet_month'],
        'initial_model_candidate': [True]*7,
    })
    cfg = SelectionConfig(permutation_repeats=1, pair_min_selected=2, screening_max_iter=20, pair_max_train_rows=180)
    result = select_track(df, inventory, 'PAIR', 'target_actual_gross_m2', PAIR_FAMILIES, cfg, 180, 2)
    assert len(result['selected']) >= 2
    assert 'horizon' in result['selected']
    assert np.isfinite(result['full_metrics']['wape'])


def test_calendar_v013_features_are_calendar_family_and_validation_candidates():
    from feature_selector_v04 import infer_family, PAIR_FAMILIES, BRANCH_FAMILIES, CALENDAR_V013_CANDIDATES
    assert len(CALENDAR_V013_CANDIDATES) == 20
    assert all(infer_family(c, PAIR_FAMILIES) == "calendar" for c in CALENDAR_V013_CANDIDATES)
    assert all(infer_family(c, BRANCH_FAMILIES) == "calendar" for c in CALENDAR_V013_CANDIDATES)


def test_validation_requires_all_calendar_v013_candidates():
    from feature_selector_v04 import validate_selection_result, CALENDAR_V013_CANDIDATES
    selected = ["x", "horizon"]
    result = {
        "selected": selected,
        "static_kept": selected,
        "quality_report": pd.DataFrame({
            "feature_name": selected, "selected": [True, True], "static_status": ["PASS_STATIC", "PASS_STATIC"]
        }),
        "full_metrics": {"wape": 0.5},
        "validation_rows_primary_current_active": 10,
        "validation_rows_secondary_all": 12,
        "candidate_features": selected + sorted(CALENDAR_V013_CANDIDATES),
    }
    assert validate_selection_result(result, "PAIR", 1)["status"] == "PASS"
    result["candidate_features"].remove("target_lunar_month_mid")
    assert validate_selection_result(result, "PAIR", 1)["status"] == "FAIL"


def test_primary_validation_excludes_current_inactive_pairs():
    from feature_selector_v04 import select_track, PAIR_FAMILIES
    rng = np.random.default_rng(7)
    n_train, n_val = 120, 30
    n = n_train + n_val
    x = rng.normal(size=n)
    y = np.maximum(5 + 2*x + rng.normal(scale=.2, size=n), 0)
    current_active = np.array([False]*n_train + [True]*20 + [False]*10)
    df = pd.DataFrame({
        'pair_lag_1': x,
        'horizon': np.tile([1,2,3], n//3 + 1)[:n],
        'target_actual_gross_m2': y,
        'target_available': True,
        'historical_train_mask': [True]*n_train + [False]*n_val,
        'official_validation_mask': [False]*n_train + [True]*n_val,
        'current_production_forecast_mask': current_active,
        'known_pair_asof_origin': True,
        'target_month': pd.to_datetime(['2025-12-01']*n_train + ['2026-01-01']*n_val),
    })
    # Selector validation requires all calendar candidates to be present in inventory/result; add constant-ish but varying placeholders.
    from feature_selector_v04 import CALENDAR_V013_CANDIDATES
    for j,c in enumerate(sorted(CALENDAR_V013_CANDIDATES)):
        if c not in df:
            df[c] = (np.arange(n) + j) % 7
    feats=['pair_lag_1','horizon'] + [c for c in sorted(CALENDAR_V013_CANDIDATES) if c!='horizon']
    inventory=pd.DataFrame({'track':['PAIR']*len(feats),'feature_name':feats,'initial_model_candidate':[True]*len(feats)})
    cfg=SelectionConfig(permutation_repeats=1,pair_min_selected=2,screening_max_iter=10,pair_max_train_rows=120)
    r=select_track(df, inventory, 'PAIR','target_actual_gross_m2',PAIR_FAMILIES,cfg,120,2)
    assert r['validation_rows_primary_current_active']==20
    assert r['validation_rows_secondary_all']==30
