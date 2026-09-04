"""Tests for empiricalCDF, clean P&L calculation, and backtesting empirical diagnostics."""

import numpy as np
import pandas as pd
import pytest

from portfolio_core.analytics.var import (
    EmpiricalCDFResult,
    calculate_clean_pnl,
    calculate_hypothetical_pnl,
    empirical_cdf,
    empiricalCDF,
    compute_portfolio_empirical_cdf,
    HistoricalVaR,
    VolatilityScaledVaR
)
from portfolio_core.analytics.statistics import empiricalCDF as empiricalCDF_stats
from portfolio_core.analytics import empiricalCDF as empiricalCDF_top


def test_empirical_cdf_basic_scalar():
    # 5 simulated PPLs: [-100, -50, 0, 50, 100]
    ppls = [-100.0, -50.0, 0.0, 50.0, 100.0]

    # actual P&L = -50: values <= -50 are -100 and -50 (2 out of 5)
    cdf_val = empiricalCDF(ppls, -50.0)
    assert isinstance(cdf_val, float)
    assert np.isclose(cdf_val, 0.40)

    # actual P&L = 0: values <= 0 are -100, -50, 0 (3 out of 5)
    assert np.isclose(empiricalCDF(ppls, 0.0), 0.60)

    # actual P&L = 100: all 5 <= 100
    assert np.isclose(empiricalCDF(ppls, 100.0), 1.00)

    # actual P&L = 200: all 5 <= 200
    assert np.isclose(empiricalCDF(ppls, 200.0), 1.00)

    # actual P&L = -200: 0 <= -200
    assert np.isclose(empiricalCDF(ppls, -200.0), 0.00)


def test_empirical_cdf_import_equivalences():
    assert empiricalCDF is empirical_cdf
    assert empiricalCDF_stats is empiricalCDF
    assert empiricalCDF_top is empiricalCDF
    assert calculate_hypothetical_pnl is calculate_clean_pnl


def test_empirical_cdf_sides():
    ppls = [-100.0, -50.0, 0.0, 50.0, 100.0]

    # Less-than or equal (default): values <= -50 -> [-100, -50] -> 2/5 = 0.40
    assert np.isclose(empiricalCDF(ppls, -50.0, side="less_equal"), 0.40)
    assert np.isclose(empiricalCDF(ppls, -50.0, side="right"), 0.40)

    # Strict less-than: values < -50 -> [-100] -> 1/5 = 0.20
    assert np.isclose(empiricalCDF(ppls, -50.0, side="less"), 0.20)
    assert np.isclose(empiricalCDF(ppls, -50.0, side="left"), 0.20)

    # Midpoint: (1 + 2) / (2 * 5) = 1.5 / 5 = 0.30
    assert np.isclose(empiricalCDF(ppls, -50.0, side="midpoint"), 0.30)

    # Greater-than or equal (survival function): values >= -50 -> [-50, 0, 50, 100] -> 4/5 = 0.80
    assert np.isclose(empiricalCDF(ppls, -50.0, side="greater_equal"), 0.80)

    # Strict greater-than: values > -50 -> [0, 50, 100] -> 3/5 = 0.60
    assert np.isclose(empiricalCDF(ppls, -50.0, side="greater"), 0.60)


def test_empirical_cdf_vectorized():
    ppls = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
    actual_pnls = [-150.0, -50.0, 0.0, 75.0, 150.0]

    # List input
    res_list = empiricalCDF(ppls, actual_pnls)
    assert isinstance(res_list, np.ndarray)
    assert np.allclose(res_list, [0.0, 0.4, 0.6, 0.8, 1.0])

    # Numpy array input
    res_np = empiricalCDF(ppls, np.array(actual_pnls))
    assert isinstance(res_np, np.ndarray)
    assert np.allclose(res_np, [0.0, 0.4, 0.6, 0.8, 1.0])

    # Pandas Series input
    dates = pd.date_range("2026-08-01", periods=5, freq="B")
    pnl_series = pd.Series(actual_pnls, index=dates)
    res_series = empiricalCDF(ppls, pnl_series)
    assert isinstance(res_series, pd.Series)
    assert (res_series.index == dates).all()
    assert np.allclose(res_series.values, [0.0, 0.4, 0.6, 0.8, 1.0])


def test_empirical_cdf_return_details():
    np.random.seed(42)
    simulated_ppls = np.random.normal(10.0, 100.0, 1000)
    actual_pnl = -180.0
    pv = 50000.0

    result = empiricalCDF(
        simulated_ppls=simulated_ppls,
        actual_pnl=actual_pnl,
        portfolio_value=pv,
        return_details=True
    )

    assert isinstance(result, EmpiricalCDFResult)
    assert float(result) == result.cdf_value
    assert 0.0 <= result.cdf_value <= 1.0
    assert result.actual_pnl == actual_pnl
    assert result.portfolio_value == pv
    assert np.isclose(result.actual_return_pct, (-180.0 / 50000.0) * 100.0)
    assert result.simulated_count == 1000
    assert result.count_satisfying == int(np.sum(simulated_ppls <= actual_pnl))
    assert np.isclose(result.percentile, result.cdf_value * 100.0)
    assert result.pnl_mean is not None
    assert result.pnl_std is not None
    assert result.var_95_gbp is not None
    assert result.var_99_gbp is not None
    assert isinstance(result.is_var_95_breach, bool)
    assert isinstance(result.is_var_99_breach, bool)
    assert "CDF Value (Probability)" in result.to_dict()
    assert "Actual Clean P&L" in result.summary_markdown()


def test_empirical_cdf_input_validation():
    # Empty simulated PPLs
    with pytest.raises(ValueError, match="simulated_ppls must contain at least one valid"):
        empiricalCDF([], 10.0)

    # All NaN simulated PPLs
    with pytest.raises(ValueError, match="simulated_ppls must contain at least one valid"):
        empiricalCDF([np.nan, np.nan], 10.0)

    # Invalid side
    with pytest.raises(ValueError, match="Invalid side"):
        empiricalCDF([10.0, 20.0], 15.0, side="unknown_side")

    # Non-positive portfolio value
    with pytest.raises(ValueError, match="portfolio_value must be positive"):
        empiricalCDF([10.0, 20.0], 15.0, portfolio_value=-1000.0)


def test_calculate_clean_pnl_zero_position_impact():
    """
    Tests that clean P&L evaluates price change strictly on base positions (zero position impact),
    independent of any intraday trading or rebalancing.
    """
    # Portfolio held at t-1: 100 NVDA, 500 STAN.L
    positions = {"NVDA": 100.0, "STAN.L": 500.0}

    # Prices at t-1
    start_prices = pd.Series({"NVDA": 120.0, "STAN.L": 8.0})
    # Prices at t
    end_prices = pd.Series({"NVDA": 125.0, "STAN.L": 7.5})

    clean_pnl, pv = calculate_clean_pnl(positions, start_prices, end_prices)

    # Expected PV at t-1: 100 * 120 + 500 * 8 = 12,000 + 4,000 = 16,000 GBP
    assert pv == 16000.0

    # Expected Clean P&L: 100 * (125 - 120) + 500 * (7.5 - 8.0) = 500 - 250 = +250 GBP
    # ZERO position impact: uses frozen positions without regard to intraday executions
    assert clean_pnl == 250.0

    # Empty positions
    pnl_zero, pv_zero = calculate_clean_pnl({}, start_prices, end_prices)
    assert pnl_zero == 0.0
    assert pv_zero == 0.0


def test_compute_portfolio_empirical_cdf_integration():
    """
    End-to-end institutional workflow test:
    - Verifies PV and simulated PPLs are generated on the exact same portfolio.
    - Verifies actual P&L has no position impact (clean hypothetical P&L).
    - Verifies eCDF evaluation.
    """
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    np.random.seed(42)

    # Generate synthetic price series
    p_nvda = 100.0 * np.exp(np.cumsum(np.random.normal(0.0008, 0.02, 100)))
    p_stan = 20.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, 100)))
    prices_df = pd.DataFrame({"NVDA": p_nvda, "STAN.L": p_stan}, index=dates)

    positions = {"NVDA": 50.0, "STAN.L": 200.0}

    # Evaluate penultimate day (t-1) to last day (t)
    cdf_val = compute_portfolio_empirical_cdf(
        positions=positions,
        prices_gbp=prices_df,
        lookback_days=60
    )

    assert isinstance(cdf_val, float)
    assert 0.0 <= cdf_val <= 1.0

    # Test with return_details=True
    detailed = compute_portfolio_empirical_cdf(
        positions=positions,
        prices_gbp=prices_df,
        lookback_days=60,
        return_details=True
    )

    assert isinstance(detailed, EmpiricalCDFResult)
    assert np.isclose(detailed.cdf_value, cdf_val)
    assert detailed.portfolio_value > 0
    assert detailed.simulated_count == 60

    # Manual verification:
    # Baseline date (t-1) is dates[-2]
    asof_date = dates[-2]
    eval_date = dates[-1]
    expected_pnl, expected_pv = calculate_clean_pnl(positions, prices_df.loc[asof_date], prices_df.loc[eval_date])

    assert np.isclose(detailed.actual_pnl, expected_pnl)
    assert np.isclose(detailed.portfolio_value, expected_pv)


def test_compute_portfolio_empirical_cdf_with_vol_scaled_model():
    """Integration test using VolatilityScaledVaR as the simulation engine."""
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    np.random.seed(42)

    p_nvda = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 100)))
    p_stan = 20.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.012, 100)))
    prices_df = pd.DataFrame({"NVDA": p_nvda, "STAN.L": p_stan}, index=dates)
    positions = {"NVDA": 25.0, "STAN.L": 150.0}

    model = VolatilityScaledVaR(confidence_level=0.95, horizon_days=1, lookback_days=50)
    res = compute_portfolio_empirical_cdf(
        positions=positions,
        prices_gbp=prices_df,
        model=model,
        lookback_days=50,
        return_details=True
    )

    assert isinstance(res, EmpiricalCDFResult)
    assert 0.0 <= res.cdf_value <= 1.0
    assert res.simulated_count == 50
