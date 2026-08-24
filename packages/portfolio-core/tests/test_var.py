"""Tests for Value-at-Risk, CVaR, and Shapley risk attribution models."""

import numpy as np
import pandas as pd
import pytest

from portfolio_core.analytics.var import (
    calculate_var_cvar_partition,
    compute_multi_model_var_spectrum,
    compute_standalone_asset_var,
    compute_shapley_risk_contributions
)


def test_calculate_var_cvar_partition():
    # 100 scenario PnLs from -100 to +100
    pnl = np.linspace(-100, 100, 101)
    # alpha = 0.05 -> 5th percentile
    var_5, cvar_5 = calculate_var_cvar_partition(pnl, 0.05)
    assert var_5 < 0
    assert cvar_5 <= var_5
    assert np.isclose(var_5, -90.0, atol=1.0)


def test_compute_multi_model_var_spectrum():
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    np.random.seed(42)
    p_nvda = 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
    p_stan = 20.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, 100)))

    prices_df = pd.DataFrame({"NVDA": p_nvda, "STAN.L": p_stan}, index=dates)
    positions = {"NVDA": 10.0, "STAN.L": 50.0}

    cls = [0.90, 0.95, 0.99]
    spectrum_df, scenarios_df, pos_values = compute_multi_model_var_spectrum(
        price_history=prices_df,
        positions=positions,
        lookback_days=90,
        confidence_levels=cls
    )

    assert not spectrum_df.empty
    assert len(pos_values) == 2
    assert "Historical Simulation" in spectrum_df["METHOD"].values
    assert "Vol-Scaled VaR (EWMA λ=0.94)" in spectrum_df["METHOD"].values
    assert set(spectrum_df["CONFIDENCE_LEVEL"].unique()) == set(cls)


def test_compute_standalone_asset_var():
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    np.random.seed(42)
    prices_df = pd.DataFrame({
        "NVDA": 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.03, 100))),
        "STAN.L": 20.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, 100)))
    }, index=dates)
    positions = {"NVDA": 10.0, "STAN.L": 50.0}

    standalone_df = compute_standalone_asset_var(
        price_history=prices_df,
        positions=positions,
        lookback_days=90,
        confidence_level=0.95
    )

    assert not standalone_df.empty
    assert len(standalone_df) == 2
    assert "HIST_VAR_GBP" in standalone_df.columns
    assert "VOL_SCALED_VAR_GBP" in standalone_df.columns


def test_compute_shapley_risk_contributions():
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    np.random.seed(42)
    prices_df = pd.DataFrame({
        "NVDA": 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.03, 100))),
        "STAN.L": 20.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, 100)))
    }, index=dates)
    positions = {"NVDA": 10.0, "STAN.L": 50.0}

    shapley_df = compute_shapley_risk_contributions(
        price_history=prices_df,
        positions=positions,
        lookback_days=90,
        confidence_level=0.95,
        num_permutations=50
    )

    assert not shapley_df.empty
    assert len(shapley_df) == 2
    assert "HIST_SHAPLEY_VAR_GBP" in shapley_df.columns
    assert "VOL_SCALED_SHAPLEY_VAR_GBP" in shapley_df.columns

    # Test exact Euler additivity: sum(Shapley) == Total Portfolio VaR
    tot_hist_var = float(shapley_df["PORTFOLIO_HIST_VAR_GBP"].iloc[0])
    sum_hist_shapley = float(shapley_df["HIST_SHAPLEY_VAR_GBP"].sum())
    assert np.isclose(tot_hist_var, sum_hist_shapley, atol=1e-2)
