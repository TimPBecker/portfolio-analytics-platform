"""Tests for distribution statistics, returns, and diagnostics."""

import numpy as np
import pandas as pd
import pytest

from portfolio_core.analytics.statistics import (
    compute_asset_returns,
    compute_distribution_metrics,
    generate_density_curves,
    compute_qq_plot_data,
    compute_top_position_movers
)


def test_compute_asset_returns():
    prices = pd.Series([100.0, 105.0, 102.0, 110.0])
    log_rets = compute_asset_returns(prices, method="log")
    simple_rets = compute_asset_returns(prices, method="simple")

    assert len(log_rets) == 3
    assert len(simple_rets) == 3
    assert np.isclose(simple_rets.iloc[0], 0.05)
    assert np.isclose(log_rets.iloc[0], np.log(1.05))


def test_compute_distribution_metrics():
    np.random.seed(42)
    normal_rets = pd.Series(np.random.normal(0.0005, 0.015, 250))
    metrics = compute_distribution_metrics(normal_rets)

    assert metrics["count"] == 250
    assert "vol_annualized_pct" in metrics
    assert "skewness" in metrics
    assert "kurtosis_excess" in metrics
    assert "jb_pvalue" in metrics


def test_generate_density_curves():
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0, 0.02, 100))
    x_grid, kde_y, norm_y, t_y = generate_density_curves(rets, num_points=100)

    assert len(x_grid) == 100
    assert len(kde_y) == 100
    assert len(norm_y) == 100
    assert (kde_y >= 0).all()
    assert (norm_y >= 0).all()


def test_compute_qq_plot_data():
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0, 0.02, 100))
    osm, osr, slope, intercept = compute_qq_plot_data(rets)

    assert len(osm) == 100
    assert len(osr) == 100
    assert slope > 0


def test_compute_top_position_movers():
    dates = pd.date_range("2026-08-01", periods=3, freq="B")
    prices_df = pd.DataFrame({
        "NVDA": [100.0, 100.0, 110.0],
        "STAN.L": [20.0, 20.0, 18.0],
        "AAPL": [50.0, 50.0, 51.0]
    }, index=dates)
    positions = {"NVDA": 100.0, "STAN.L": 500.0, "AAPL": 200.0}

    movers = compute_top_position_movers(prices_df, positions, top_n=2)
    assert len(movers) == 2
    # STAN.L: 500 * (18 - 20) = -1000 GBP (|Δ| = 1000)
    # NVDA: 100 * (110 - 100) = +1000 GBP (|Δ| = 1000)
    # AAPL: 200 * (51 - 50) = +200 GBP (|Δ| = 200)
    assert movers["ABS_DIFF_GBP"].iloc[0] == 1000.0
    assert movers["ABS_DIFF_GBP"].iloc[1] == 1000.0
    assert "DIFF_GBP" in movers.columns
    assert "DIFF_PCT" in movers.columns
    assert "PRICE_TODAY_GBP" in movers.columns
    assert "PRICE_PREV_GBP" in movers.columns


def test_compute_top_position_movers_empty():
    empty_df = pd.DataFrame()
    movers = compute_top_position_movers(empty_df, {"NVDA": 10.0})
    assert movers.empty
    assert "TICKER" in movers.columns
