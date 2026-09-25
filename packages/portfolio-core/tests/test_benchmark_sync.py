"""
Regression tests for benchmark creation and automatic transaction synchronization.
Verifies that:
1. Benchmark creation (single and linear combinations) correctly validates weights and fallback names.
2. check_benchmark_transactions_coverage accurately flags transactions missing benchmark shadow trades.
3. sync_benchmark_transactions_and_risk regenerates benchmark shadow transactions, benchmark values,
   and portfolio risk numbers (PORTFOLIO_VAR and PORTFOLIO_RISK_CONTRIBUTIONS) for all subsequent days.
"""

import pytest
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

from portfolio_core.db import (
    create_all_tables,
    add_benchmark,
    delete_benchmark,
    fetch_benchmarks_info,
    parse_benchmark_constituents,
    generate_benchmark_fallback_name,
    record_transaction,
    fetch_all_transactions,
    check_benchmark_transactions_coverage,
    generate_and_store_benchmark_transactions,
    calculate_and_store_daily_benchmark_values,
    fetch_benchmark_values_history,
    fetch_benchmark_transactions,
    regenerate_risk_numbers_from_date,
    sync_benchmark_transactions_and_risk,
    fetch_stored_var_metrics,
)


# =============================================================================
# 1. Benchmark Creation Regression Tests
# =============================================================================

def test_benchmark_creation_single_and_linear_combination(sqlite_test_engine):
    """
    Regression test: Verifies benchmark creation for single-ticker and weighted multi-ticker combinations,
    checking constituent weight normalization and automatic fallback name generation.
    """
    engine = sqlite_test_engine

    # Test parser
    c_single = parse_benchmark_constituents("CSP1.L: 100")
    assert c_single == {"CSP1.L": 1.0}

    c_multi = parse_benchmark_constituents("CSP1.L: 60, VUKE.L: 40")
    assert np.isclose(c_multi["CSP1.L"], 0.6)
    assert np.isclose(c_multi["VUKE.L"], 0.4)
    assert np.isclose(sum(c_multi.values()), 1.0)

    # Test fallback name generator
    name_single = generate_benchmark_fallback_name({"CSP1.L": 1.0})
    assert name_single == "CSP1.L_100"

    name_multi = generate_benchmark_fallback_name({"CSP1.L": 0.7, "VUKE.L": 0.3})
    assert name_multi == "CSP1.L_70_VUKE.L_30"

    # Register single benchmark
    res1 = add_benchmark(constituents="CSP1.L: 100", name="S&P 500 GBP", description="Core S&P 500", engine=engine)
    assert res1["status"] == "success"
    assert res1["benchmark_code"] == "CSP1.L_100"

    # Register multi-constituent linear combination with empty name (tests fallback name generation)
    res2 = add_benchmark(constituents="CSP1.L: 70, VUKE.L: 30", name="", description="70/30 US/UK Equity", engine=engine)
    assert res2["status"] == "success"
    assert res2["benchmark_code"] == "CSP1.L_70_VUKE.L_30"
    assert res2["name"] == "CSP1.L_70_VUKE.L_30"

    # Verify presence in BENCHMARKS table
    bm_info = fetch_benchmarks_info(engine=engine)
    assert not bm_info.empty
    codes = bm_info["BENCHMARK_CODE"].tolist()
    assert "CSP1.L_100" in codes
    assert "CSP1.L_70_VUKE.L_30" in codes


def test_benchmark_creation_validation_errors(sqlite_test_engine):
    """
    Regression test: Verifies that invalid benchmark constituent inputs are properly rejected.
    """
    engine = sqlite_test_engine

    # Empty constituents string
    with pytest.raises(ValueError, match="cannot be empty"):
        parse_benchmark_constituents("")

    # Whitespace only
    with pytest.raises(ValueError, match="cannot be empty"):
        parse_benchmark_constituents("   ")

    # Negative weight
    with pytest.raises(ValueError, match="strictly positive"):
        parse_benchmark_constituents("CSP1.L: 120, VUKE.L: -20")

    # Zero weight
    with pytest.raises(ValueError, match="strictly positive"):
        parse_benchmark_constituents("CSP1.L: 0")


# =============================================================================
# 2. Benchmark Transactions Coverage Detection Tests
# =============================================================================

def test_check_benchmark_transactions_coverage(sqlite_test_engine):
    """
    Regression test: Verifies that check_benchmark_transactions_coverage correctly identifies
    when all transactions have corresponding benchmark transactions, and detects missing ones.
    """
    engine = sqlite_test_engine

    # Case 1: Empty DB -> should report synced
    cov_empty = check_benchmark_transactions_coverage(engine=engine)
    assert cov_empty["is_synced"] is True
    assert cov_empty["missing_count"] == 0
    assert cov_empty["earliest_missing_date"] is None

    # Case 2: Register benchmark and seed a transaction without benchmark shadow trades
    add_benchmark(constituents="CSP1.L: 100", engine=engine)
    tx1 = record_transaction(ticker="NVDA", transaction_date="2026-08-01", quantity=10.0, engine=engine)

    cov_unmatched = check_benchmark_transactions_coverage(engine=engine)
    assert cov_unmatched["is_synced"] is False
    assert cov_unmatched["missing_count"] == 1
    assert tx1["id"] in cov_unmatched["missing_tx_ids"]
    assert cov_unmatched["earliest_missing_date"] == "2026-08-01"

    # Seed mock prices so shadow trade pricing succeeds
    mock_prices = pd.DataFrame([
        {"DATE": "2026-08-01", "TICKER": "NVDA", "CLOSE": 100.0, "CURRENCY": "GBP"},
        {"DATE": "2026-08-01", "TICKER": "CSP1.L", "CLOSE": 500.0, "CURRENCY": "GBP"},
        {"DATE": "2026-08-01", "TICKER": "VWRL.L", "CLOSE": 80.0, "CURRENCY": "GBP"},
        {"DATE": "2026-08-01", "TICKER": "VUKE.L", "CLOSE": 30.0, "CURRENCY": "GBP"},
    ])
    mock_prices.to_sql("ASSET_PRICES", con=engine, if_exists="append", index=False)

    # Generate benchmark shadow transactions
    generate_and_store_benchmark_transactions(engine=engine)

    # Case 3: Now it should be fully synced
    cov_synced = check_benchmark_transactions_coverage(engine=engine)
    assert cov_synced["is_synced"] is True
    assert cov_synced["missing_count"] == 0
    assert cov_synced["earliest_missing_date"] is None

    # Case 4: Add a SECOND transaction on a later date
    tx2 = record_transaction(ticker="AAPL", transaction_date="2026-08-05", quantity=20.0, engine=engine)
    cov_after_new_tx = check_benchmark_transactions_coverage(engine=engine)
    assert cov_after_new_tx["is_synced"] is False
    assert cov_after_new_tx["missing_count"] == 1
    assert tx2["id"] in cov_after_new_tx["missing_tx_ids"]
    assert cov_after_new_tx["earliest_missing_date"] == "2026-08-05"


# =============================================================================
# 3. End-to-End Benchmark & Subsequent Risk Regeneration Tests
# =============================================================================

def test_sync_benchmark_transactions_and_risk_end_to_end(sqlite_test_engine):
    """
    Regression test: Verifies that when a new transaction is added:
    1. check_benchmark_transactions_coverage flags it.
    2. sync_benchmark_transactions_and_risk regenerates benchmark shadow transactions,
       daily benchmark values, and risk numbers (PORTFOLIO_VAR and PORTFOLIO_RISK_CONTRIBUTIONS)
       for all subsequent dates.
    3. The updated risk metrics reflect the changed holdings.
    4. Coverage check returns is_synced=True afterwards.
    """
    engine = sqlite_test_engine

    # Seed price history across 5 trading days for NVDA, AAPL, and benchmark constituents
    dates = ["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05"]
    price_records = []
    for i, d in enumerate(dates):
        price_records.extend([
            {"DATE": d, "TICKER": "NVDA", "CLOSE": 100.0 + (i * 2.0), "CURRENCY": "GBP"},
            {"DATE": d, "TICKER": "AAPL", "CLOSE": 150.0 + (i * 1.5), "CURRENCY": "GBP"},
            {"DATE": d, "TICKER": "CSP1.L", "CLOSE": 500.0 + (i * 5.0), "CURRENCY": "GBP"},
            {"DATE": d, "TICKER": "VWRL.L", "CLOSE": 80.0 + (i * 0.8), "CURRENCY": "GBP"},
            {"DATE": d, "TICKER": "VUKE.L", "CLOSE": 30.0 + (i * 0.3), "CURRENCY": "GBP"},
        ])
    pd.DataFrame(price_records).to_sql("ASSET_PRICES", con=engine, if_exists="append", index=False)

    # Record initial trade on Day 1 (2026-08-01): 10 shares of NVDA
    record_transaction(ticker="NVDA", transaction_date="2026-08-01", quantity=10.0, engine=engine)

    # Run initial synchronization (with min_lookback=2 for mock timeline)
    res_sync1 = sync_benchmark_transactions_and_risk(engine=engine, min_lookback=2, lookback_days=5)
    assert res_sync1["synced"] is True
    assert res_sync1["regenerated"] is True
    assert res_sync1["benchmark_values_stored"] > 0

    # Verify benchmark transactions and values were created
    bm_tx = fetch_benchmark_transactions(benchmark_code="CSP1.L_100", engine=engine)
    assert not bm_tx.empty
    assert len(bm_tx) == 1
    assert bm_tx.iloc[0]["ORIGINAL_TX_ID"] == 1

    bm_vals = fetch_benchmark_values_history(benchmark_code="CSP1.L_100", engine=engine)
    assert not bm_vals.empty
    assert len(bm_vals) >= 5

    # Check risk records were populated for eligible dates (Day 3, 4, 5)
    var_day5_initial = fetch_stored_var_metrics(asof_date="2026-08-05", engine=engine)
    assert not var_day5_initial.empty
    # Portfolio on Day 5 initially has only NVDA (10 shares @ 108 = £1080)
    initial_pv = float(var_day5_initial.iloc[0]["PORTFOLIO_VALUE_GBP"])
    assert np.isclose(initial_pv, 1080.0, atol=1.0)

    # Verify coverage check is currently clean
    assert check_benchmark_transactions_coverage(engine=engine)["is_synced"] is True

    # -------------------------------------------------------------------------
    # Now simulate user adding a NEW transaction on Day 3 (2026-08-03): Buy 10 AAPL
    # -------------------------------------------------------------------------
    record_transaction(ticker="AAPL", transaction_date="2026-08-03", quantity=10.0, engine=engine)

    # Coverage check MUST immediately detect that benchmark shadow trade is missing
    cov = check_benchmark_transactions_coverage(engine=engine)
    assert cov["is_synced"] is False
    assert cov["missing_count"] == 1
    assert cov["earliest_missing_date"] == "2026-08-03"

    # Trigger synchronization
    res_sync2 = sync_benchmark_transactions_and_risk(engine=engine, min_lookback=2, lookback_days=5)
    assert res_sync2["synced"] is True
    assert res_sync2["regenerated"] is True
    assert res_sync2["earliest_missing_date"] == "2026-08-03"

    # 1. Benchmark shadow transactions for CSP1.L_100 now include both transactions
    bm_tx_updated = fetch_benchmark_transactions(benchmark_code="CSP1.L_100", engine=engine)
    assert len(bm_tx_updated) == 2

    # 2. Risk numbers for subsequent days (Day 3, 4, 5) were regenerated with the new holdings
    var_day5_updated = fetch_stored_var_metrics(asof_date="2026-08-05", engine=engine)
    assert not var_day5_updated.empty
    # Day 5 now has NVDA (10 @ 108 = £1080) + AAPL (10 @ 156 = £1560) = £2640 total
    updated_pv = float(var_day5_updated.iloc[0]["PORTFOLIO_VALUE_GBP"])
    assert np.isclose(updated_pv, 2640.0, atol=5.0)
    assert updated_pv > initial_pv

    # 3. Coverage check is synced again
    assert check_benchmark_transactions_coverage(engine=engine)["is_synced"] is True

    # 4. If sync is invoked again without new transactions, it does NOT regenerate
    res_sync3 = sync_benchmark_transactions_and_risk(engine=engine, min_lookback=2, lookback_days=5)
    assert res_sync3["synced"] is True
    assert res_sync3["regenerated"] is False


def test_benchmark_values_creation_bounded_to_last_successful_date(sqlite_test_engine):
    """
    Regression test: Verifies that benchmark generation and calculation strictly creates
    benchmark values up to the last successful processed date, never creating more recent data.
    """
    from portfolio_core.db import (
        record_processed_date,
        get_latest_processed_date,
        fetch_benchmark_values_history,
        calculate_and_store_daily_benchmark_values
    )

    engine = sqlite_test_engine
    create_all_tables(engine)

    dates = ["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"]

    # Insert transactions and market prices across all 7 days
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO TRANSACTIONS (ID, TICKER, TRANSACTION_DATE, QUANTITY) VALUES (1, 'NVDA', '2026-08-01', 10.0)"))
        for i, d in enumerate(dates):
            conn.execute(text(f"INSERT INTO ASSET_PRICES (DATE, TICKER, CURRENCY, OPEN, HIGH, LOW, CLOSE, VOLUME) VALUES ('{d}', 'NVDA', 'USD', 100, 105, 95, 100, 1000)"))
            conn.execute(text(f"INSERT INTO ASSET_PRICES (DATE, TICKER, CURRENCY, OPEN, HIGH, LOW, CLOSE, VOLUME) VALUES ('{d}', 'CSP1.L', 'GBP', 50, 55, 45, 50, 500)"))
            conn.execute(text(f"INSERT INTO FX_RATES (DATE, FROM_CURRENCY, TO_CURRENCY, RATE) VALUES ('{d}', 'USD', 'GBP', 0.8)"))

    # 1. Set the last successful processed date to 2026-08-04
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM `PROCESSED_DATES`"))
    record_processed_date("2026-08-04", status="SUCCESS", engine=engine)
    assert get_latest_processed_date(engine=engine) == "2026-08-04"

    # 2. Add benchmark and calculate daily values
    add_benchmark(constituents="CSP1.L: 100", benchmark_code="CSP1.L_100", engine=engine)
    res = calculate_and_store_daily_benchmark_values(engine=engine)
    assert res["records_stored"] > 0

    # 3. Verify BENCHMARK_VALUES contains NO records beyond 2026-08-04
    bm_vals = fetch_benchmark_values_history(benchmark_code="CSP1.L_100", engine=engine)
    assert not bm_vals.empty
    max_bm_date = str(bm_vals["DATE"].max())[:10]
    assert max_bm_date == "2026-08-04"

    # 4. Verify explicit asof_date cutoff works as well (e.g. 2026-08-02)
    res_cutoff = calculate_and_store_daily_benchmark_values(engine=engine, asof_date="2026-08-02")
    bm_vals_cutoff = fetch_benchmark_values_history(benchmark_code="CSP1.L_100", engine=engine)
    max_cutoff_date = str(bm_vals_cutoff["DATE"].max())[:10]
    assert max_cutoff_date == "2026-08-02"

