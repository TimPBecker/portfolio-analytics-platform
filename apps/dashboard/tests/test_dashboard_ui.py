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
from apps.dashboard.src.ui.theme import get_plotly_layout_defaults, PALETTE


def test_theme_layout_defaults():
    layout = get_plotly_layout_defaults()
    assert layout["plot_bgcolor"] == "#FFFFFF"
    assert layout["paper_bgcolor"] == "#FFFFFF"
    assert "primary" in PALETTE


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
