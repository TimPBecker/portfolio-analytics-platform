"""
Integration tests for Streamlit dashboard UI components and portfolio_core wiring.
"""

import pytest
import pandas as pd
import numpy as np

from portfolio_core.db import get_connection_string
from portfolio_core.analytics.volatility import calculate_ewma_volatility, calculate_sample_volatility
from portfolio_core.analytics.var import compute_multi_model_var_spectrum, compute_shapley_risk_contributions
from portfolio_core.analytics.statistics import compute_distribution_metrics
try:
    from src.ui.theme import get_plotly_layout_defaults, PALETTE
    from src.ui.tab_portfolio import render_tab_portfolio
except ImportError:
    from apps.dashboard.src.ui.theme import get_plotly_layout_defaults, PALETTE
    from apps.dashboard.src.ui.tab_portfolio import render_tab_portfolio
from portfolio_core.analytics.statistics import compute_top_position_movers


def test_theme_layout_defaults():
    layout = get_plotly_layout_defaults()
    assert layout["plot_bgcolor"] == "#FFFFFF"
    assert layout["paper_bgcolor"] == "#FFFFFF"
    assert "primary" in PALETTE


def test_top_movers_computation():
    dates = pd.date_range("2026-08-01", periods=5, freq="B")
    prices_df = pd.DataFrame({
        "NVDA": [100.0, 102.0, 105.0, 108.0, 115.0],
        "STAN.L": [20.0, 20.5, 20.0, 19.5, 18.0],
        "AAPL": [150.0, 151.0, 152.0, 153.0, 154.0],
        "MSFT": [300.0, 301.0, 302.0, 303.0, 304.0]
    }, index=dates)
    positions = {"NVDA": 100.0, "STAN.L": 500.0, "AAPL": 50.0, "MSFT": 20.0}

    movers = compute_top_position_movers(prices_df, positions, top_n=10)
    assert len(movers) == 4
    # NVDA day diff: 100 * (115 - 108) = +700 GBP (|Δ| = 700)
    # STAN.L day diff: 500 * (18.0 - 19.5) = -750 GBP (|Δ| = 750)
    # MSFT day diff: 20 * (304 - 303) = +20 GBP (|Δ| = 20)
    # AAPL day diff: 50 * (154 - 153) = +50 GBP (|Δ| = 50)
    assert movers["TICKER"].iloc[0] == "STAN.L"
    assert movers["ABS_DIFF_GBP"].iloc[0] == 750.0
    assert movers["DIFF_GBP"].iloc[0] == -750.0
    assert movers["TICKER"].iloc[1] == "NVDA"
    assert movers["ABS_DIFF_GBP"].iloc[1] == 700.0
    assert movers["DIFF_GBP"].iloc[1] == 700.0


def test_end_to_end_analytics_pipeline():
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    np.random.seed(42)
    prices_df = pd.DataFrame({
        "NVDA": 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100))),
        "STAN.L": 20.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, 100)))
    }, index=dates)
    positions = {"NVDA": 10.0, "STAN.L": 50.0}

    # 1. Volatility
    log_rets = np.log(prices_df / prices_df.shift(1)).dropna()
    ewma_vol = calculate_ewma_volatility(log_rets)
    assert not ewma_vol.empty

    # 2. VaR Spectrum
    spectrum_df, scenarios_df, pos_values = compute_multi_model_var_spectrum(
        price_history=prices_df,
        positions=positions,
        lookback_days=90,
        confidence_levels=[0.90, 0.95, 0.99]
    )
    assert not spectrum_df.empty
    assert "Historical Simulation" in spectrum_df["METHOD"].values

    # 3. Shapley Risk Contributions
    shapley_df = compute_shapley_risk_contributions(
        price_history=prices_df,
        positions=positions,
        lookback_days=90,
        confidence_level=0.95,
        num_permutations=20
    )
    assert not shapley_df.empty
    assert len(shapley_df) == 2


def test_transaction_id_and_record():
    from sqlalchemy import create_engine
    from portfolio_core.db import create_all_tables, get_next_transaction_id, record_transaction, fetch_all_transactions

    engine = create_engine("sqlite:///:memory:")
    create_all_tables(engine)

    # Initial table is empty -> next ID should be 1
    next_id = get_next_transaction_id(engine=engine)
    assert next_id == 1

    # Record first transaction
    rec1 = record_transaction(ticker="NVDA", transaction_date="2026-08-20", quantity=10.0, engine=engine)
    assert rec1["id"] == 1
    assert rec1["ticker"] == "NVDA"
    assert rec1["quantity"] == 10.0

    # Next ID should now be 2
    assert get_next_transaction_id(engine=engine) == 2

    # Record second transaction
    rec2 = record_transaction(ticker="STAN.L", transaction_date="2026-08-21", quantity=-5.0, engine=engine)
    assert rec2["id"] == 2

    tx_df = fetch_all_transactions(engine=engine)
    assert len(tx_df) == 2
    assert tx_df["ID"].iloc[0] == 2  # Ordered by ID DESC


def test_yahoo_close_price_lookup():
    from portfolio_core.db import query_yahoo_close_price

    res = query_yahoo_close_price(ticker="NVDA", target_date="2026-08-20")
    assert res["status"] == "success"
    assert res["ticker"] == "NVDA"
    assert res["close_price"] > 0
    assert "currency" in res
    assert "fx_rate_to_gbp" in res
    assert "close_price_gbp" in res
    assert res["close_price_gbp"] > 0
    assert np.isclose(res["close_price_gbp"], res["close_price"] * res["fx_rate_to_gbp"])

