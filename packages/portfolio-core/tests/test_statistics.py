"""Tests for distribution statistics, returns, and diagnostics."""

import numpy as np
import pandas as pd
import pytest

from portfolio_core.analytics.statistics import (
    compute_asset_returns,
    compute_distribution_metrics,
    generate_density_curves,
    compute_qq_plot_data
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
