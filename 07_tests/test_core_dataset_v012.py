import sys
from pathlib import Path
import pandas as pd

# In Colab/project test runs, add 02_src/data to sys.path before importing.
try:
    import core_dataset_v012 as c
except ImportError:
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / '02_src' / 'data'))
    import core_dataset_v012 as c


def fixtures():
    master = pd.DataFrame([
        {'bravo_sku':'01.L1.3060.A','base_sku':'01.L1.3060.X','master_status':'active','product_group':'GACH','brand':'UNIS','price_group':'3060','factory_code':'F1','pull_source':'P','branch_channel':'UNIS','factory_sku':'A','sale_sku':'01.X','source_file':'m'},
        {'bravo_sku':'01.L1.3060.B','base_sku':'01.L1.3060.X','master_status':'inactive','product_group':'GACH','brand':'UNIS','price_group':'3060','factory_code':'F1','pull_source':'P','branch_channel':'UNIS','factory_sku':'B','sale_sku':'01.X','source_file':'m'},
    ])
    branch = pd.DataFrame([{'source_file':'c','source_row_no':1,'branch_code':'001','branch_name':'B1','region':'R','brand':'UNIS','master_status':'Hoạt động','loaded_at':pd.Timestamp('2026-01-01',tz='UTC')}])
    sales = pd.DataFrame([
        {'source_file':'s','source_row_no':1,'bravo_sku':'01.L1.3060.A','sku_name':'A','branch_code':'001','unit':'M2','month':pd.Timestamp('2025-01-01'),'total_quantity':10,'total_amount':100,'line_count':1,'loaded_at':pd.Timestamp('2026-01-01',tz='UTC')},
        {'source_file':'s','source_row_no':2,'bravo_sku':'01.L1.3060.B','sku_name':'B','branch_code':'001','unit':'M2','month':pd.Timestamp('2025-01-01'),'total_quantity':-2,'total_amount':-20,'line_count':1,'loaded_at':pd.Timestamp('2026-01-01',tz='UTC')},
        {'source_file':'s','source_row_no':3,'bravo_sku':'01.L1.3060.A','sku_name':'A','branch_code':'001','unit':'M2','month':pd.Timestamp('2025-03-01'),'total_quantity':0,'total_amount':0,'line_count':1,'loaded_at':pd.Timestamp('2026-01-01',tz='UTC')},
    ])
    return master, branch, sales


def build_all():
    master, branch, sales = fixtures()
    bridge = c.build_bridge(master)
    dim_base = c.build_dim_base(bridge)
    dim_branch = c.build_dim_branch(branch)
    bravo, _ = c.build_bravo_observation(sales, bridge, dim_branch)
    fact = c.build_fact(bravo)
    pair = c.build_pair_panel(fact, dim_base, dim_branch, pd.Timestamp('2025-03-01'))
    return bridge, dim_base, dim_branch, bravo, fact, pair


def test_authoritative_many_bravo_to_one_base_aggregates():
    _, _, _, _, fact, _ = build_all()
    jan = fact.loc[fact.month.eq(pd.Timestamp('2025-01-01'))].iloc[0]
    assert jan.gross_positive_m2 == 10
    assert jan.negative_m2 == -2
    assert jan.net_m2 == 8
    assert jan.observed_bravo_count == 2


def test_current_inactive_variant_does_not_remove_history():
    _, _, _, _, fact, _ = build_all()
    assert len(fact) == 2


def test_missing_month_is_implicit_observed_zero():
    _, _, _, _, _, pair = build_all()
    feb = pair.loc[pair.month.eq(pd.Timestamp('2025-02-01'))].iloc[0]
    assert not bool(feb.source_row_observed)
    assert bool(feb.actual_observed)
    assert bool(feb.target_available)
    assert feb.actual_gross_m2 == 0
    assert feb.zero_semantics == 'OBSERVED_ZERO_IMPLICIT'


def test_explicit_zero_is_zero():
    _, _, _, _, _, pair = build_all()
    mar = pair.loc[pair.month.eq(pd.Timestamp('2025-03-01'))].iloc[0]
    assert bool(mar.actual_observed)
    assert mar.actual_gross_m2 == 0
    assert mar.zero_semantics == 'OBSERVED_ZERO'


def test_branch_status_normalization():
    _, _, dim_branch, _, _, _ = build_all()
    assert dim_branch.branch_current_status.iloc[0] == 'active'


def test_channel_relation_is_snapshot_metadata():
    _, _, _, bravo, _, _ = build_all()
    assert set(bravo.channel_relation_snapshot) == {'EXACT_MATCH'}


def test_negative_only_is_not_trainable_zero_and_first_positive_is_origin_safe():
    master = pd.DataFrame([
        {'bravo_sku':'01.L1.3060.C','base_sku':'01.L1.3060.Y','master_status':'active','product_group':'GACH','brand':'UNIS','price_group':'3060','factory_code':'F1','pull_source':'P','branch_channel':'UNIS','factory_sku':'C','sale_sku':'01.Y','source_file':'m'},
    ])
    branch = pd.DataFrame([{'source_file':'c','source_row_no':1,'branch_code':'001','branch_name':'B1','region':'R','brand':'UNIS','master_status':'Hoạt động','loaded_at':pd.Timestamp('2026-01-01',tz='UTC')}])
    sales = pd.DataFrame([
        {'source_file':'s','source_row_no':1,'bravo_sku':'01.L1.3060.C','sku_name':'C','branch_code':'001','unit':'M2','month':pd.Timestamp('2025-01-01'),'total_quantity':-5,'total_amount':-50,'line_count':1,'loaded_at':pd.Timestamp('2026-01-01',tz='UTC')},
        {'source_file':'s','source_row_no':2,'bravo_sku':'01.L1.3060.C','sku_name':'C','branch_code':'001','unit':'M2','month':pd.Timestamp('2025-03-01'),'total_quantity':7,'total_amount':70,'line_count':1,'loaded_at':pd.Timestamp('2026-01-01',tz='UTC')},
    ])
    bridge = c.build_bridge(master)
    dim_base = c.build_dim_base(bridge)
    dim_branch = c.build_dim_branch(branch)
    bravo, _ = c.build_bravo_observation(sales, bridge, dim_branch)
    fact = c.build_fact(bravo)
    pair = c.build_pair_panel(fact, dim_base, dim_branch, pd.Timestamp('2025-03-01'))

    jan = pair.loc[pair.month.eq(pd.Timestamp('2025-01-01'))].iloc[0]
    feb = pair.loc[pair.month.eq(pd.Timestamp('2025-02-01'))].iloc[0]
    mar = pair.loc[pair.month.eq(pd.Timestamp('2025-03-01'))].iloc[0]

    assert bool(jan.actual_negative_only)
    assert not bool(jan.target_available)
    assert pd.isna(jan.actual_gross_m2)
    assert jan.zero_semantics == 'NEGATIVE_ONLY_GROSS_ZERO'
    assert pd.isna(jan.first_positive_month_to_date)
    assert pd.isna(feb.first_positive_month_to_date)
    assert mar.first_positive_month_to_date == pd.Timestamp('2025-03-01')
    assert mar.months_since_first_positive == 0


def test_validation_checks_origin_safety_and_negative_only_target():
    master = pd.DataFrame([
        {'bravo_sku':'01.L1.3060.C','base_sku':'01.L1.3060.Y','master_status':'active','product_group':'GACH','brand':'UNIS','price_group':'3060','factory_code':'F1','pull_source':'P','branch_channel':'UNIS','factory_sku':'C','sale_sku':'01.Y','source_file':'m'},
    ])
    branch_raw = pd.DataFrame([{'source_file':'c','source_row_no':1,'branch_code':'001','branch_name':'B1','region':'R','brand':'UNIS','master_status':'Hoạt động','loaded_at':pd.Timestamp('2026-01-01',tz='UTC')}])
    sales = pd.DataFrame([
        {'source_file':'s','source_row_no':1,'bravo_sku':'01.L1.3060.C','sku_name':'C','branch_code':'001','unit':'M2','month':pd.Timestamp('2025-01-01'),'total_quantity':-5,'total_amount':-50,'line_count':1,'loaded_at':pd.Timestamp('2026-01-01',tz='UTC')},
        {'source_file':'s','source_row_no':2,'bravo_sku':'01.L1.3060.C','sku_name':'C','branch_code':'001','unit':'M2','month':pd.Timestamp('2025-03-01'),'total_quantity':7,'total_amount':70,'line_count':1,'loaded_at':pd.Timestamp('2026-01-01',tz='UTC')},
    ])
    bridge = c.build_bridge(master)
    dim_base = c.build_dim_base(bridge)
    dim_branch = c.build_dim_branch(branch_raw)
    bravo, _ = c.build_bravo_observation(sales, bridge, dim_branch)
    fact = c.build_fact(bravo)
    pair = c.build_pair_panel(fact, dim_base, dim_branch, pd.Timestamp('2025-03-01'))
    branch = c.build_branch_panel(fact, dim_branch, pd.Timestamp('2025-03-01'))
    calendar = c.build_calendar(pd.Timestamp('2025-01-01'), pd.Timestamp('2025-03-01'))
    v = c.validate_core(sales, bravo, bridge, dim_base, dim_branch, fact, pair, branch, calendar)
    assert v['status'] == 'PASS'
    assert v['checks']['negative_only_target_unavailable']['pass']
    assert v['checks']['first_positive_origin_safe']['pass']


def test_branch_month_without_sales_is_observed_zero():
    master, branch_raw, sales = fixtures()
    bridge = c.build_bridge(master)
    dim_base = c.build_dim_base(bridge)
    dim_branch = c.build_dim_branch(branch_raw)
    bravo, _ = c.build_bravo_observation(sales, bridge, dim_branch)
    fact = c.build_fact(bravo)
    branch = c.build_branch_panel(fact, dim_branch, pd.Timestamp('2025-04-01'))
    apr = branch.loc[branch.month.eq(pd.Timestamp('2025-04-01'))].iloc[0]
    assert bool(apr.branch_observed)
    assert not bool(apr.branch_source_row_observed)
    assert apr.branch_gross_m2 == 0
