"""Unit tests for VaR Backtesting Analytics: PIT uniformity, Binomial outliers, and Kupiec tests."""

import math
import numpy as np
import pandas as pd
import pytest

from portfolio_core.analytics.backtesting import (
    UniformityTestResult,
    BinomialOutlierResult,
    KupiecPOFResult,
    IndependenceTestResult,
    ConditionalCoverageResult,
    evaluate_ecdf_uniformity,
    evaluate_binomial_outliers,
    kupiec_pof_test,
    kupiec_independence_test,
    christoffersen_conditional_coverage_test,
    run_backtest_diagnostics,
    generate_portfolio_backtest_timeline
)


def test_ecdf_uniformity_well_calibrated():
    """Tests that a sample generated from Uniform(0, 1) passes the Kolmogorov-Smirnov uniformity test."""
    np.random.seed(42)
    uniform_ecdfs = np.random.uniform(0.0, 1.0, 500)

    res = evaluate_ecdf_uniformity(uniform_ecdfs)
    assert isinstance(res, UniformityTestResult)
    assert res.sample_size == 500
    assert res.p_value > 0.05
    assert res.is_uniform_5pct is True
    assert np.isclose(res.empirical_mean, 0.5, atol=0.05)
    assert "KS Statistic (D)" in res.to_dict()
    assert res.calibration_quality in ("Excellent Calibration", "Acceptable Calibration")
    assert res.to_dict()["Calibration Assessment"] == res.calibration_quality


def test_ecdf_uniformity_miscalibrated():
    """Tests that a heavily biased distribution rejects the Uniformity hypothesis."""
    # Clustered near zero (heavy losses exceeding model forecast)
    biased_ecdfs = np.random.beta(a=0.5, b=5.0, size=250)

    res = evaluate_ecdf_uniformity(biased_ecdfs)
    assert res.is_uniform_5pct is False
    assert res.p_value < 0.01
    assert "Miscalibration" in res.calibration_quality
    assert res.to_dict()["Calibration Assessment"] == res.calibration_quality



def test_evaluate_binomial_outliers_acceptable():
    """Tests Binomial outlier evaluation with expected failure rate."""
    # 250 trading days, 95% VaR (expected ~12.5 outliers)
    # Suppose we observe exactly 12 outliers
    exceptions = np.zeros(250, dtype=int)
    exceptions[:12] = 1

    res = evaluate_binomial_outliers(exceptions, confidence_level=0.95, is_exception_indicator=True)
    assert isinstance(res, BinomialOutlierResult)
    assert res.total_observations == 250
    assert res.observed_outliers == 12
    assert np.isclose(res.expected_outliers, 12.5)
    assert res.is_acceptable_5pct is True
    assert res.basel_zone == "Green"
    assert res.ci_lower < 0.05 < res.ci_upper


def test_evaluate_binomial_outliers_excessive():
    """Tests Binomial outlier evaluation with severe tail exceptions (rejects model)."""
    # 250 trading days, 99% VaR (expected 2.5 outliers)
    # Suppose we observe 15 outliers (severe breach)
    exceptions = np.zeros(250, dtype=int)
    exceptions[:15] = 1

    res = evaluate_binomial_outliers(exceptions, confidence_level=0.99, is_exception_indicator=True)
    assert res.observed_outliers == 15
    assert res.p_value < 0.001
    assert res.is_acceptable_5pct is False
    assert res.basel_zone == "Red"


def test_evaluate_binomial_outliers_with_ecdf_input():
    """Tests Binomial outlier evaluation directly from eCDF values."""
    # 100 eCDF values: 5 values <= 0.05 (breaches for 95% VaR)
    ecdfs = [0.01, 0.02, 0.03, 0.04, 0.05] + [0.50] * 95

    res = evaluate_binomial_outliers(ecdfs, confidence_level=0.95, is_exception_indicator=False)
    assert res.observed_outliers == 5
    assert res.total_observations == 100
    assert np.isclose(res.expected_outliers, 5.0)
    assert res.is_acceptable_5pct is True


def test_kupiec_pof_test():
    """
    Tests Kupiec Proportion of Failures (POF / LR_uc) against standard theoretical values.
    For T=255, p=0.05, if observed exceptions = 13 (rate ~ 5.1%), LR_uc should be ~0.007.
    """
    t_obs = 255
    x_obs = 13
    exceptions = [1] * x_obs + [0] * (t_obs - x_obs)

    res = kupiec_pof_test(exceptions, confidence_level=0.95)
    assert isinstance(res, KupiecPOFResult)
    assert res.total_observations == 255
    assert res.observed_outliers == 13
    assert res.lr_statistic < 0.1  # Almost perfect match to expected
    assert res.p_value > 0.80
    assert res.is_accepted_5pct is True

    # Test failure case: 35 exceptions out of 255 at 95% confidence
    bad_exceptions = [1] * 35 + [0] * (t_obs - 35)
    bad_res = kupiec_pof_test(bad_exceptions, confidence_level=0.95)
    assert bad_res.lr_statistic > 3.841
    assert bad_res.p_value < 0.001
    assert bad_res.is_accepted_5pct is False


def test_kupiec_independence_test_independent():
    """Tests exception independence when exceptions are spaced apart (no clustering)."""
    # 200 days, 10 exceptions evenly spaced every 20 days
    pattern = ([0] * 19 + [1]) * 10
    res = kupiec_independence_test(pattern)
    assert isinstance(res, IndependenceTestResult)
    # Since exceptions never follow exceptions, T11 == 0
    assert res.contingency_matrix["T11"] == 0
    assert res.is_independent_5pct is True
    assert res.lr_statistic < 3.841


def test_kupiec_independence_test_clustered():
    """Tests exception independence when exceptions exhibit severe temporal clustering."""
    # 200 days with consecutive clustered runs of exceptions:
    # 10 consecutive exceptions (volatility shock clustering)
    clustered = [0] * 100 + [1] * 10 + [0] * 90
    res = kupiec_independence_test(clustered)
    # T11 is 9 (9 consecutive 1->1 transitions)
    assert res.contingency_matrix["T11"] == 9
    assert res.p11 > res.p01
    # Should reject independence due to significant clustering
    assert res.lr_statistic > 3.841
    assert res.p_value < 0.05
    assert res.is_independent_5pct is False


def test_christoffersen_conditional_coverage_test():
    """Tests the joint conditional coverage test (LR_cc = LR_uc + LR_ind)."""
    pattern = ([0] * 19 + [1]) * 10  # 200 days, 10 exceptions, unclustered
    pof = kupiec_pof_test(pattern, confidence_level=0.95)
    ind = kupiec_independence_test(pattern)
    cc = christoffersen_conditional_coverage_test(pof, ind)

    assert isinstance(cc, ConditionalCoverageResult)
    assert np.isclose(cc.lr_cc, pof.lr_statistic + ind.lr_statistic)
    assert cc.critical_value_5pct == pytest.approx(5.991, rel=1e-3)
    assert cc.is_accepted_5pct is True


def test_generate_portfolio_backtest_timeline():
    """Tests the 1-year timeline generation engine holding positions constant."""
    dates = pd.date_range("2025-01-01", periods=120, freq="B")
    np.random.seed(42)
    p_nvda = 100.0 * np.exp(np.cumsum(np.random.normal(0.0008, 0.02, 120)))
    p_stan = 20.0 * np.exp(np.cumsum(np.random.normal(0.0004, 0.015, 120)))
    prices_df = pd.DataFrame({"NVDA": p_nvda, "STAN.L": p_stan}, index=dates)

    positions = {"NVDA": 25.0, "STAN.L": 100.0}

    timeline = generate_portfolio_backtest_timeline(
        positions=positions,
        prices_gbp=prices_df,
        backtest_days=50,
        lookback_days=40,
        confidence_level=0.95
    )

    assert not timeline.empty
    assert len(timeline) == 50
    assert "CLEAN_PNL_GBP" in timeline.columns
    assert "HIST_VAR_GBP" in timeline.columns
    assert "HIST_ECDF" in timeline.columns
    assert "VOL_SCALED_VAR_GBP" in timeline.columns
    assert "VOL_SCALED_ECDF" in timeline.columns

    # Test eCDF values bounded in [0, 1]
    assert (timeline["HIST_ECDF"] >= 0.0).all()
    assert (timeline["HIST_ECDF"] <= 1.0).all()
    assert (timeline["VOL_SCALED_ECDF"] >= 0.0).all()
    assert (timeline["VOL_SCALED_ECDF"] <= 1.0).all()
    assert "HIST_PPLS" in timeline.columns
    assert "VOL_SCALED_PPLS" in timeline.columns

    # Diagnostics suite on timeline at 95%
    diag_95 = run_backtest_diagnostics(timeline, confidence_level=0.95, model_column_prefix="HIST")
    assert "uniformity" in diag_95
    assert "binomial" in diag_95
    assert "kupiec_pof" in diag_95
    assert "independence" in diag_95
    assert "conditional_coverage" in diag_95
    assert diag_95["dynamic_var"] is not None

    # Verify dynamic recalculation when changing confidence level:
    diag_90 = run_backtest_diagnostics(timeline, confidence_level=0.90, model_column_prefix="HIST")
    diag_99 = run_backtest_diagnostics(timeline, confidence_level=0.99, model_column_prefix="HIST")

    # Higher confidence level (more extreme tail) must produce fewer or equal outliers
    outliers_90 = diag_90["binomial"].observed_outliers
    outliers_95 = diag_95["binomial"].observed_outliers
    outliers_99 = diag_99["binomial"].observed_outliers
    assert outliers_90 >= outliers_95 >= outliers_99

    # Verify fallback behavior when PPLS column is dropped (e.g. legacy cached session state)
    timeline_no_ppls = timeline.drop(columns=["HIST_PPLS", "VOL_SCALED_PPLS"])
    fallback_90 = run_backtest_diagnostics(timeline_no_ppls, confidence_level=0.90, model_column_prefix="HIST")
    fallback_99 = run_backtest_diagnostics(timeline_no_ppls, confidence_level=0.99, model_column_prefix="HIST")
    assert fallback_90["binomial"].observed_outliers >= fallback_99["binomial"].observed_outliers


def test_two_tailed_backtesting_diagnostics():
    """Tests that run_backtest_diagnostics evaluates both the loss tail (alpha) and gain tail (1-alpha)."""
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    np.random.seed(123)
    p_nvda = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 100)))
    p_stan = 20.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.015, 100)))
    prices_df = pd.DataFrame({"NVDA": p_nvda, "STAN.L": p_stan}, index=dates)

    positions = {"NVDA": 20.0, "STAN.L": 80.0}

    timeline = generate_portfolio_backtest_timeline(
        positions=positions,
        prices_gbp=prices_df,
        backtest_days=40,
        lookback_days=30,
        confidence_level=0.95
    )

    assert "HIST_GAIN_VAR_GBP" in timeline.columns
    assert "VOL_SCALED_GAIN_VAR_GBP" in timeline.columns

    diag = run_backtest_diagnostics(timeline, confidence_level=0.95, model_column_prefix="HIST")

    # Both tails must be present
    assert "loss_tail" in diag
    assert "gain_tail" in diag
    loss = diag["loss_tail"]
    gain = diag["gain_tail"]

    # Check loss tail structures
    assert isinstance(loss["binomial"], BinomialOutlierResult)
    assert isinstance(loss["kupiec_pof"], KupiecPOFResult)
    assert isinstance(loss["independence"], IndependenceTestResult)
    assert isinstance(loss["conditional_coverage"], ConditionalCoverageResult)
    assert loss["tail_probability"] == pytest.approx(0.05)

    # Check gain tail structures
    assert isinstance(gain["binomial"], BinomialOutlierResult)
    assert isinstance(gain["kupiec_pof"], KupiecPOFResult)
    assert isinstance(gain["independence"], IndependenceTestResult)
    assert isinstance(gain["conditional_coverage"], ConditionalCoverageResult)
    assert gain["tail_probability"] == pytest.approx(0.05)

    # Loss VaR is typically negative; Gain VaR is typically positive
    assert (loss["dynamic_var"] < 0).mean() > 0.8
    assert (gain["dynamic_var"] > 0).mean() > 0.8

    # Dynamic recalculation check across tails
    diag_99 = run_backtest_diagnostics(timeline, confidence_level=0.99, model_column_prefix="HIST")
    assert diag_99["loss_tail"]["binomial"].observed_outliers <= loss["binomial"].observed_outliers
    assert diag_99["gain_tail"]["binomial"].observed_outliers <= gain["binomial"].observed_outliers

    # Verify separate PIT uniformity tests for both models
    assert isinstance(diag["uniformity_hist"], UniformityTestResult)
    assert isinstance(diag["uniformity_vol"], UniformityTestResult)
    assert diag["uniformity_hist"].sample_size == 40
    assert diag["uniformity_vol"].sample_size == 40
    assert 0.0 <= diag["uniformity_hist"].p_value <= 1.0
    assert 0.0 <= diag["uniformity_vol"].p_value <= 1.0
