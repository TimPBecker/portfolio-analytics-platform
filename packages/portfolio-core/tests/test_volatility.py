"""Tests for volatility estimation analytics."""

import numpy as np
import pandas as pd
import pytest

from portfolio_core.analytics.volatility import (
    calculate_sample_volatility,
    calculate_ewma_volatility,
    calculate_scaling_factors,
    compute_volatility_summary_metrics
)


def test_calculate_sample_volatility():
    # Constant non-zero returns series -> zero standard deviation
    rets = pd.Series([0.01] * 50)
    vol = calculate_sample_volatility(rets, window=20, annualize=False)
    assert len(vol) == 50
    assert np.allclose(vol.iloc[-1], 0.0, atol=1e-6)

    # Random returns
    np.random.seed(42)
    random_rets = pd.Series(np.random.normal(0, 0.02, 100))
    vol_ann = calculate_sample_volatility(random_rets, window=30, annualize=True)
    assert len(vol_ann) == 100
    assert not vol_ann.isna().any()
    assert 0.10 < vol_ann.iloc[-1] < 0.60


def test_calculate_ewma_volatility():
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0, 0.015, 100))
    ewma_vol = calculate_ewma_volatility(rets, decay_factor=0.94, annualize=True)
    assert len(ewma_vol) == 100
    assert not ewma_vol.isna().any()
    assert (ewma_vol > 0).all()


def test_calculate_scaling_factors():
    vol = pd.Series([0.20, 0.25, 0.30, 0.15])
    # Latest vol is 0.15
    # scaling = 0.15 / vol -> [0.75, 0.60, 0.50, 1.00]
    scales = calculate_scaling_factors(vol)
    assert len(scales) == 4
    assert np.isclose(scales.iloc[-1], 1.0)
    assert np.isclose(scales.iloc[0], 0.75)
    assert np.isclose(scales.iloc[2], 0.50)


def test_compute_volatility_summary_metrics():
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0, 0.02, 100))
    summary = compute_volatility_summary_metrics(rets, ewma_lambda=0.94)
    assert "latest_ewma_vol_pct" in summary
    assert "mean_ewma_vol_pct" in summary
    assert "current_vol_percentile_rank" in summary
    assert 0.0 <= summary["current_vol_percentile_rank"] <= 100.0
