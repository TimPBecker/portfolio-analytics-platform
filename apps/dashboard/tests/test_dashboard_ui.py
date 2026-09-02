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
    from src.ui.tab_benchmarks import render_tab_benchmarks
except ImportError:
    from apps.dashboard.src.ui.theme import get_plotly_layout_defaults, PALETTE
    from apps.dashboard.src.ui.tab_portfolio import render_tab_portfolio
    from apps.dashboard.src.ui.tab_benchmarks import render_tab_benchmarks
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


def test_benchmark_tables_and_shadow_calculations():
    from sqlalchemy import create_engine
    from portfolio_core.db import (
        create_all_tables,
        fetch_benchmarks_info,
        add_benchmark,
        delete_benchmark,
        record_transaction,
        generate_benchmark_fallback_name,
        parse_benchmark_constituents,
        generate_and_store_benchmark_transactions,
        calculate_and_store_daily_benchmark_values,
        fetch_benchmark_values_history,
        fetch_benchmark_transactions
    )

    engine = create_engine("sqlite:///:memory:")
    create_all_tables(engine)

    # 1. Test fallback name generation
    assert generate_benchmark_fallback_name({"CSP1.L": 0.6, "VUKE.L": 0.4}) == "CSP1.L_60_VUKE.L_40"
    assert generate_benchmark_fallback_name({"VWRL.L": 1.0}) == "VWRL.L_100"

    # 2. Test constituent parser
    parsed = parse_benchmark_constituents("CSP1.L: 60, VUKE.L: 40")
    assert parsed["CSP1.L"] == 0.6
    assert parsed["VUKE.L"] == 0.4

    # 3. Default benchmarks seeded
    bm_info = fetch_benchmarks_info(engine=engine)
    assert not bm_info.empty
    assert "CSP1.L_100" in bm_info["BENCHMARK_CODE"].values

    # 4. Add custom linear combination benchmark with fallback naming
    add_res = add_benchmark(constituents="CSP1.L: 70, VUKE.L: 30", name="", description="70/30 US/UK Blend", engine=engine)
    assert add_res["status"] == "success"
    assert add_res["name"] == "CSP1.L_70_VUKE.L_30"
    assert add_res["benchmark_code"] == "CSP1.L_70_VUKE.L_30"

    bm_info2 = fetch_benchmarks_info(engine=engine)
    assert "CSP1.L_70_VUKE.L_30" in bm_info2["BENCHMARK_CODE"].values

    # 5. Record trade and generate shadow transactions for linear combinations
    record_transaction(ticker="NVDA", transaction_date="2026-08-01", quantity=10.0, engine=engine)

    # Write mock price matrix rows into ASSET_PRICES
    mock_prices = pd.DataFrame([
        {"DATE": "2026-08-01", "TICKER": "NVDA", "CLOSE": 100.0, "CURRENCY": "GBP"},
        {"DATE": "2026-08-01", "TICKER": "CSP1.L", "CLOSE": 500.0, "CURRENCY": "GBP"},
        {"DATE": "2026-08-01", "TICKER": "VUKE.L", "CLOSE": 50.0, "CURRENCY": "GBP"},
        {"DATE": "2026-08-02", "TICKER": "NVDA", "CLOSE": 110.0, "CURRENCY": "GBP"},
        {"DATE": "2026-08-02", "TICKER": "CSP1.L", "CLOSE": 550.0, "CURRENCY": "GBP"},
        {"DATE": "2026-08-02", "TICKER": "VUKE.L", "CLOSE": 55.0, "CURRENCY": "GBP"},
    ])
    mock_prices.to_sql("ASSET_PRICES", con=engine, if_exists="append", index=False)

    bm_tx_df = generate_and_store_benchmark_transactions(engine=engine)
    assert not bm_tx_df.empty

    # Trade: NVDA 10 @ £100 = £1,000 GBP.
    # For CSP1.L_70_VUKE.L_30:
    # CSP1.L (70%): £700 / £500 = 1.4 shares
    # VUKE.L (30%): £300 / £50 = 6.0 shares
    combo_tx = bm_tx_df[bm_tx_df["BENCHMARK_CODE"] == "CSP1.L_70_VUKE.L_30"]
    assert not combo_tx.empty
    csp1_row = combo_tx[combo_tx["TICKER"] == "CSP1.L"].iloc[0]
    vuke_row = combo_tx[combo_tx["TICKER"] == "VUKE.L"].iloc[0]
    assert csp1_row["QUANTITY"] == 1.4
    assert csp1_row["GBP_VALUE"] == 700.0
    assert vuke_row["QUANTITY"] == 6.0
    assert vuke_row["GBP_VALUE"] == 300.0

    # 6. Daily valuation calculation across constituents
    val_res = calculate_and_store_daily_benchmark_values(engine=engine)
    assert val_res["records_stored"] > 0

    hist_bms = fetch_benchmark_values_history(benchmark_code="CSP1.L_70_VUKE.L_30", engine=engine)
    assert not hist_bms.empty
    trade_vals = hist_bms[hist_bms["DATE"] >= "2026-08-01"]
    assert not trade_vals.empty
    # Day 1 (2026-08-01): 1.4 * 500 + 6.0 * 50 = 700 + 300 = £1000
    day1_row = trade_vals[trade_vals["DATE"].dt.strftime("%Y-%m-%d") == "2026-08-01"].iloc[0]
    assert float(day1_row["TOTAL_VALUE"]) == 1000.0
    assert float(day1_row["STOCKS"]) == 1000.0
    assert float(day1_row["CASH"]) == 0.0

    # Day 2 (2026-08-02): 1.4 * 550 + 6.0 * 55 = 770 + 330 = £1100
    day2_row = trade_vals[trade_vals["DATE"].dt.strftime("%Y-%m-%d") == "2026-08-02"].iloc[0]
    assert float(day2_row["TOTAL_VALUE"]) == 1100.0

    # 7. Add dividend payout for constituent and verify CASH account collection
    mock_div = pd.DataFrame([
        {"DATE": "2026-08-02", "TICKER": "VUKE.L", "CLOSE": 55.0, "DIVIDENDS": 5.0, "CURRENCY": "GBP"}
    ])
    mock_div.to_sql("ASSET_PRICES", con=engine, if_exists="append", index=False)

    calculate_and_store_daily_benchmark_values(engine=engine)
    hist_div = fetch_benchmark_values_history(benchmark_code="CSP1.L_70_VUKE.L_30", engine=engine)
    day2_div_row = hist_div[hist_div["DATE"].dt.strftime("%Y-%m-%d") == "2026-08-02"].iloc[0]
    # Dividend for VUKE.L: 6.0 shares * £5.00 = £30.00 cash
    assert float(day2_div_row["CASH"]) == 30.0
    assert float(day2_div_row["STOCKS"]) == 1100.0
    assert float(day2_div_row["TOTAL_VALUE"]) == 1130.0

    # 8. Delete benchmark from BENCHMARKS table and verify removal
    del_res = delete_benchmark(benchmark_code="CSP1.L_70_VUKE.L_30", engine=engine)
    assert del_res is True
    bm_info_after_del = fetch_benchmarks_info(engine=engine)
    assert "CSP1.L_70_VUKE.L_30" not in bm_info_after_del["BENCHMARK_CODE"].values


def test_quarto_report_template_discovery():
    try:
        from src.services.report_generator import find_report_template
    except ImportError:
        from apps.dashboard.src.services.report_generator import find_report_template

    template = find_report_template()
    assert template is not None
    assert template.exists()
    assert template.name == "portfolio_report.qmd"


def test_quarto_typst_pdf_report_generation():
    import shutil
    if not shutil.which("quarto"):
        pytest.skip("Quarto CLI not installed in current test runner environment")

    try:
        from src.services.report_generator import generate_portfolio_pdf_report
    except ImportError:
        from apps.dashboard.src.services.report_generator import generate_portfolio_pdf_report

    success, pdf_path, pdf_bytes, err = generate_portfolio_pdf_report()
    assert success is True
    assert err is None
    assert pdf_path is not None
    assert pdf_bytes is not None
    assert len(pdf_bytes) > 1000
    assert pdf_bytes[:5] == b"%PDF-"


def test_dashboard_databases_config_and_options():
    """Verify that dashboard config defines databases defaulting to stocks and allows multi-db options."""
    from portfolio_core.config import load_config, get_db_config

    cfg = load_config()
    db_cfg = get_db_config(cfg, is_test=False, is_dev=False)
    assert "databases" in db_cfg
    assert "stocks" in db_cfg["databases"]

    # Verify custom databases list parsing
    custom_cfg = {
        "db": {
            "type": "mariadb",
            "database": "stocks",
            "databases": ["stocks", "stocks_dev", "stocks_sim"]
        }
    }
    resolved = get_db_config(custom_cfg, is_test=False, is_dev=False)
    assert resolved["databases"] == ["stocks", "stocks_dev", "stocks_sim"]


