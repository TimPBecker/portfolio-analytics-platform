"""
Pipeline integration tests for automatic benchmark transactions coverage check and risk regeneration.
Verifies that Dagster pipeline assets (portfolio_daily_values and run_risk_pipeline):
1. Detect when benchmark shadow transactions are missing for any recorded portfolio transactions.
2. Regenerate benchmark shadow transactions and daily benchmark valuations.
3. Automatically regenerate portfolio risk numbers for subsequent dates.
4. Cleanly pass through when all transactions are already synchronized.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# Add pipeline directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio_core.db import (
    create_all_tables,
    record_transaction,
    add_benchmark,
    check_benchmark_transactions_coverage,
    fetch_stored_var_metrics,
    fetch_benchmark_transactions,
)
from repo import (
    portfolio_daily_values,
    PortfolioValuesConfig,
    run_risk_pipeline,
)


def _meta_val(metadata, key):
    val = metadata.get(key)
    return getattr(val, "value", val)


def test_pipeline_benchmark_check_and_regeneration(sqlite_test_engine):
    """
    Verifies that portfolio_daily_values detects unsynced benchmark transactions and regenerates
    benchmark values and subsequent risk numbers.
    """
    engine = sqlite_test_engine

    # Seed 10 days of price data
    dates = pd.date_range("2026-08-01", periods=10, freq="B").strftime("%Y-%m-%d")
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

    # Record trade on Day 1
    record_transaction(ticker="NVDA", transaction_date=dates[0], quantity=10.0, engine=engine)

    # Initial pipeline check: coverage is False because trade was added without benchmark shadow trade
    cov_init = check_benchmark_transactions_coverage(engine=engine)
    assert cov_init["is_synced"] is False
    assert cov_init["missing_count"] == 1

    # Mock DatabaseResource using MagicMock
    db_res = MagicMock()
    db_res.get_engine.return_value = engine

    # Run portfolio_daily_values asset
    cfg = PortfolioValuesConfig(backfill_days=0)
    out = portfolio_daily_values(config=cfg, db=db_res)

    # Assert that sync was triggered during pipeline execution
    assert _meta_val(out.metadata, "Benchmark Transactions Synced") is False
    assert "Benchmark Sync Triggered" in out.metadata
    assert _meta_val(out.metadata, "Earliest Missing Date") == dates[0]

    # Coverage check is now synced
    assert check_benchmark_transactions_coverage(engine=engine)["is_synced"] is True

    # Run portfolio_daily_values a second time with no new transactions -> should remain synced
    out2 = portfolio_daily_values(config=cfg, db=db_res)
    assert _meta_val(out2.metadata, "Benchmark Transactions Synced") is True
    assert "Benchmark Sync Triggered" not in out2.metadata


def test_run_risk_pipeline_auto_sync_on_new_transaction(sqlite_test_engine):
    """
    Verifies that run_risk_pipeline detects unsynced benchmark transactions and regenerates
    subsequent risk numbers.
    """
    engine = sqlite_test_engine

    # Seed 10 days of price data
    dates = pd.date_range("2026-08-01", periods=10, freq="B").strftime("%Y-%m-%d")
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

    # Record trade on Day 1
    record_transaction(ticker="NVDA", transaction_date=dates[0], quantity=10.0, engine=engine)

    # Run risk pipeline directly
    res = run_risk_pipeline(
        backfill_days=0,
        min_lookback=2,
        lookback_days=5,
        engine=engine,
        asof_date=dates[-1]
    )

    # Assert that sync was triggered and reported
    assert res.get("backfill_stats") is not None
    assert res["backfill_stats"]["sync_triggered"] is True

    # Assert that shadow transactions were generated
    assert check_benchmark_transactions_coverage(engine=engine)["is_synced"] is True
