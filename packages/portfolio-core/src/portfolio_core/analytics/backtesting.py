"""
VaR Backtesting, Probability Integral Transform (PIT), and Regulatory Diagnostics Module.

Implements institutional statistical testing for market risk models:
1. Probability Integral Transform (PIT) Uniformity Test:
   Tests if empirical cumulative distribution function (eCDF) values follow Uniform(0, 1)
   via the two-sided Kolmogorov-Smirnov test.
2. Binomial Distribution Outlier Test:
   Tests exception frequency under Binomial(T, p) distribution, exact two-sided p-values,
   Clopper-Pearson confidence intervals, and Basel Traffic Light classification.
3. Kupiec Proportion of Failures (POF) Test (Unconditional Coverage, LR_uc):
   Likelihood ratio test for whether the observed exception rate matches the expected rate.
4. Kupiec / Christoffersen Independence Test (LR_ind):
   First-order two-state Markov chain likelihood ratio test for exception clustering.
5. Christoffersen Conditional Coverage Test (LR_cc):
   Joint test combining unconditional coverage and exception independence.
6. Vectorized 1-Year Historical Backtesting Engine:
   Rolls a frozen portfolio across 1 year of market dates to generate daily Clean P&L,
   simulated PPL vectors, multi-model VaRs, and daily eCDF PIT values with zero position impact.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import numpy as np
import pandas as pd
from scipy import stats

from portfolio_core.analytics.var import (
    HistoricalVaR,
    VolatilityScaledVaR,
    EWMAVolatility,
    calculate_clean_pnl,
    empirical_cdf,
    empiricalCDF
)


@dataclass
class UniformityTestResult:
    """Outcomes of the Kolmogorov-Smirnov test for eCDF Uniformity over (0, 1)."""
    statistic: float
    p_value: float
    sample_size: int
    empirical_mean: float
    empirical_std: float
    expected_mean: float = 0.50
    expected_std: float = 0.288675  # sqrt(1/12)
    is_uniform_5pct: bool = True
    test_name: str = "Kolmogorov-Smirnov Test (Uniform[0,1])"

    @property
    def calibration_quality(self) -> str:
        """Categorical interpretation of model calibration based on the KS p-value."""
        if self.p_value >= 0.20:
            return "Excellent Calibration"
        elif self.p_value >= 0.05:
            return "Acceptable Calibration"
        elif self.p_value >= 0.01:
            return "Moderate Miscalibration"
        else:
            return "Severe Miscalibration"

    def to_dict(self) -> dict:
        return {
            "Test": self.test_name,
            "Sample Size": self.sample_size,
            "KS Statistic (D)": round(self.statistic, 4),
            "p-value": round(self.p_value, 5),
            "Empirical Mean": round(self.empirical_mean, 4),
            "Expected Mean": self.expected_mean,
            "Empirical Std": round(self.empirical_std, 4),
            "Expected Std": round(self.expected_std, 4),
            "Uniformity Null Hypothesis (5%)": "ACCEPTED (Well-Calibrated)" if self.is_uniform_5pct else "REJECTED (Miscalibrated)",
            "Calibration Assessment": self.calibration_quality,
        }


@dataclass
class BinomialOutlierResult:
    """Outcomes of the Binomial Distribution outlier count and coverage evaluation."""
    total_observations: int
    confidence_level: float
    expected_failure_rate: float
    expected_outliers: float
    observed_outliers: int
    observed_failure_rate: float
    p_value: float
    ci_lower: float
    ci_upper: float
    basel_zone: str  # "Green", "Yellow", "Red"
    is_acceptable_5pct: bool

    def to_dict(self) -> dict:
        return {
            "Total Observations (T)": self.total_observations,
            "Confidence Level": f"{self.confidence_level * 100:.2f}%",
            "Expected Outliers": round(self.expected_outliers, 2),
            "Observed Outliers": self.observed_outliers,
            "Expected Rate": f"{self.expected_failure_rate * 100:.2f}%",
            "Observed Rate": f"{self.observed_failure_rate * 100:.2f}%",
            "Binomial Test p-value": round(self.p_value, 5),
            "95% CI for Failure Rate": f"[{self.ci_lower * 100:.2f}%, {self.ci_upper * 100:.2f}%]",
            "Basel Traffic Light Zone": self.basel_zone,
            "Status (5% Significance)": "PASS" if self.is_acceptable_5pct else "FAIL"
        }


@dataclass
class KupiecPOFResult:
    """Outcomes of Kupiec's Proportion of Failures (Unconditional Coverage) Test."""
    total_observations: int
    observed_outliers: int
    expected_failure_rate: float
    observed_failure_rate: float
    lr_statistic: float
    p_value: float
    critical_value_5pct: float = 3.841459  # Chi-squared 1 df at 95%
    is_accepted_5pct: bool = True
    test_name: str = "Kupiec Proportion of Failures (LR_uc)"

    def to_dict(self) -> dict:
        return {
            "Test": self.test_name,
            "Sample Size (T)": self.total_observations,
            "Observed Failures": self.observed_outliers,
            "Expected Failures": round(self.total_observations * self.expected_failure_rate, 2),
            "LR Statistic (LR_uc)": round(self.lr_statistic, 4),
            "Critical Value (95%)": round(self.critical_value_5pct, 4),
            "p-value": round(self.p_value, 5),
            "Unconditional Coverage": "ACCEPTED" if self.is_accepted_5pct else "REJECTED"
        }


@dataclass
class IndependenceTestResult:
    """Outcomes of the Kupiec / Christoffersen Exception Independence Test."""
    contingency_matrix: Dict[str, int]  # T00, T01, T10, T11
    p01: float  # P(Exception | No Exception)
    p11: float  # P(Exception | Exception) - clustering rate
    pi_unconditional: float
    lr_statistic: float
    p_value: float
    critical_value_5pct: float = 3.841459  # Chi-squared 1 df at 95%
    is_independent_5pct: bool = True
    test_name: str = "Kupiec / Christoffersen Independence Test (LR_ind)"

    def to_dict(self) -> dict:
        return {
            "Test": self.test_name,
            "Non-Exception -> Non-Exception (T00)": self.contingency_matrix.get("T00", 0),
            "Non-Exception -> Exception (T01)": self.contingency_matrix.get("T01", 0),
            "Exception -> Non-Exception (T10)": self.contingency_matrix.get("T10", 0),
            "Exception -> Exception (T11, Clustered)": self.contingency_matrix.get("T11", 0),
            "P(Exception | Prior Non-Exception)": f"{self.p01 * 100:.2f}%",
            "P(Exception | Prior Exception)": f"{self.p11 * 100:.2f}%",
            "LR Statistic (LR_ind)": round(self.lr_statistic, 4),
            "Critical Value (95%)": round(self.critical_value_5pct, 4),
            "p-value": round(self.p_value, 5),
            "Independence Null Hypothesis": "ACCEPTED (No Clustering)" if self.is_independent_5pct else "REJECTED (Clustering Detected)"
        }


@dataclass
class ConditionalCoverageResult:
    """Outcomes of the joint Christoffersen Conditional Coverage Test (LR_cc = LR_uc + LR_ind)."""
    lr_uc: float
    lr_ind: float
    lr_cc: float
    p_value: float
    critical_value_5pct: float = 5.991465  # Chi-squared 2 df at 95%
    is_accepted_5pct: bool = True
    test_name: str = "Christoffersen Conditional Coverage (LR_cc)"

    def to_dict(self) -> dict:
        return {
            "Test": self.test_name,
            "LR_uc (Coverage)": round(self.lr_uc, 4),
            "LR_ind (Independence)": round(self.lr_ind, 4),
            "Joint Statistic (LR_cc)": round(self.lr_cc, 4),
            "Critical Value (95%, 2 df)": round(self.critical_value_5pct, 4),
            "p-value": round(self.p_value, 5),
            "Conditional Coverage": "ACCEPTED" if self.is_accepted_5pct else "REJECTED"
        }


# =====================================================================
# Statistical Test Implementations
# =====================================================================

def evaluate_ecdf_uniformity(ecdf_series: Union[pd.Series, np.ndarray, List[float]]) -> UniformityTestResult:
    """
    Tests if the empirical cumulative distribution function (eCDF / PIT) values are uniformly
    distributed over (0, 1) using the two-sided Kolmogorov-Smirnov goodness-of-fit test.

    Under the null hypothesis of a correctly specified risk model, the probability integral
    transform values U_t = F_t(Clean P&L_t) are independent and identically distributed Uniform(0, 1).
    """
    arr = np.asarray(ecdf_series, dtype=float)
    clean_arr = arr[~np.isnan(arr)]
    n = len(clean_arr)

    if n < 5:
        return UniformityTestResult(
            statistic=0.0,
            p_value=1.0,
            sample_size=n,
            empirical_mean=float(np.mean(clean_arr)) if n > 0 else 0.5,
            empirical_std=float(np.std(clean_arr)) if n > 1 else 0.0,
            is_uniform_5pct=True
        )

    # Two-sided Kolmogorov-Smirnov test against standard uniform distribution
    ks_res = stats.kstest(clean_arr, "uniform")
    stat = float(ks_res.statistic)
    p_val = float(ks_res.pvalue)

    emp_mean = float(np.mean(clean_arr))
    emp_std = float(np.std(clean_arr, ddof=1)) if n > 1 else 0.0

    return UniformityTestResult(
        statistic=stat,
        p_value=p_val,
        sample_size=n,
        empirical_mean=emp_mean,
        empirical_std=emp_std,
        is_uniform_5pct=bool(p_val > 0.05)
    )


# Alias
test_ecdf_uniformity = evaluate_ecdf_uniformity



def evaluate_binomial_outliers(
    exceptions_or_ecdfs: Union[pd.Series, np.ndarray, List[float], List[int], List[bool]],
    confidence_level: float = 0.95,
    is_exception_indicator: bool = False
) -> BinomialOutlierResult:
    """
    Evaluates the count and statistical significance of VaR outliers under the Binomial distribution.
    Allows changing the confidence interval/level dynamically to re-compute coverage diagnostics.

    Parameters:
    -----------
    exceptions_or_ecdfs : array-like
        Either a 1D boolean/integer indicator of exceptions (1 = loss breached VaR),
        or a continuous array of eCDF PIT values in [0, 1].
    confidence_level : float
        Target VaR confidence level (e.g., 0.95 for 95% VaR, 0.99 for 99% VaR).
    is_exception_indicator : bool
        If True, treats input directly as 0/1 exception flags.
        If False, identifies exceptions where eCDF <= (1.0 - confidence_level).
    """
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"Confidence level must be in (0, 1), got {confidence_level}")

    raw_arr = np.asarray(exceptions_or_ecdfs, dtype=float)
    clean_vals = raw_arr[~np.isnan(raw_arr)]
    t_obs = len(clean_vals)

    if t_obs == 0:
        raise ValueError("Input array must contain at least one valid observation.")

    p_expected = 1.0 - confidence_level
    expected_count = t_obs * p_expected

    if is_exception_indicator:
        k_observed = int(np.sum(clean_vals > 0))
    else:
        # eCDF value represents P(PPL <= Clean P&L). A tail breach occurs when eCDF <= alpha
        k_observed = int(np.sum(clean_vals <= (p_expected + 1e-9)))

    observed_p = k_observed / t_obs

    # Exact two-sided Binomial test
    binom_res = stats.binomtest(k=k_observed, n=t_obs, p=p_expected, alternative="two-sided")
    p_val = float(binom_res.pvalue)

    # Clopper-Pearson exact confidence interval for observed failure rate
    ci = binom_res.proportion_ci(confidence_level=0.95, method="exact")
    ci_lower = float(ci.low)
    ci_upper = float(ci.high)

    # Basel Traffic Light Classification (Cumulative binomial probability bounds)
    # Green: cumulative probability < 95%
    # Yellow: 95% <= cumulative probability < 99.99%
    # Red: cumulative probability >= 99.99%
    cum_prob = float(stats.binom.cdf(k_observed, t_obs, p_expected))
    if cum_prob < 0.95:
        basel_zone = "Green"
    elif cum_prob < 0.9999:
        basel_zone = "Yellow"
    else:
        basel_zone = "Red"

    return BinomialOutlierResult(
        total_observations=t_obs,
        confidence_level=float(confidence_level),
        expected_failure_rate=float(p_expected),
        expected_outliers=float(expected_count),
        observed_outliers=k_observed,
        observed_failure_rate=float(observed_p),
        p_value=p_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        basel_zone=basel_zone,
        is_acceptable_5pct=bool(p_val > 0.05)
    )


def kupiec_pof_test(
    exceptions: Union[pd.Series, np.ndarray, List[Union[int, bool]]],
    confidence_level: float = 0.95
) -> KupiecPOFResult:
    """
    Computes Kupiec's Proportion of Failures (POF / Unconditional Coverage) Likelihood Ratio test.
    Tests the null hypothesis H0: p = 1 - confidence_level against H1: p != 1 - confidence_level.

    Formula:
        LR_uc = 2 * [ x * ln(p_hat / p) + (T - x) * ln((1 - p_hat) / (1 - p)) ] ~ Chi2(1)
    """
    arr = np.asarray(exceptions, dtype=float)
    clean_exc = (arr[~np.isnan(arr)] > 0).astype(int)
    t_obs = len(clean_exc)

    if t_obs == 0:
        raise ValueError("Exceptions array cannot be empty.")

    p_exp = 1.0 - confidence_level
    x_obs = int(np.sum(clean_exc))
    p_hat = x_obs / t_obs

    # Handle boundary conditions for log-likelihood ratio
    if x_obs == 0:
        # As x -> 0, x * ln(p_hat / p) -> 0
        lr_stat = -2.0 * t_obs * math.log(1.0 - p_exp)
    elif x_obs == t_obs:
        # As x -> T, (T - x) * ln((1 - p_hat) / (1 - p)) -> 0
        lr_stat = -2.0 * t_obs * math.log(p_exp)
    else:
        term1 = x_obs * math.log(p_hat / p_exp)
        term2 = (t_obs - x_obs) * math.log((1.0 - p_hat) / (1.0 - p_exp))
        lr_stat = 2.0 * (term1 + term2)

    lr_stat = max(0.0, float(lr_stat))
    p_val = float(1.0 - stats.chi2.cdf(lr_stat, df=1))
    crit_val = float(stats.chi2.ppf(0.95, df=1))  # 3.841

    return KupiecPOFResult(
        total_observations=t_obs,
        observed_outliers=x_obs,
        expected_failure_rate=float(p_exp),
        observed_failure_rate=float(p_hat),
        lr_statistic=lr_stat,
        p_value=p_val,
        critical_value_5pct=crit_val,
        is_accepted_5pct=bool(lr_stat < crit_val and p_val > 0.05)
    )


def kupiec_independence_test(
    exceptions: Union[pd.Series, np.ndarray, List[Union[int, bool]]]
) -> IndependenceTestResult:
    """
    Computes Kupiec / Christoffersen Exception Independence Likelihood Ratio test (LR_ind).
    Uses a first-order two-state Markov chain to test if exceptions occur independently
    or exhibit clustering over time.

    Contingency Matrix:
        T00: Non-exception followed by non-exception
        T01: Non-exception followed by exception
        T10: Exception followed by non-exception
        T11: Exception followed by exception (consecutive exceptions / clustering)

    Formula:
        LR_ind = 2 * [ ln L1 - ln L0 ] ~ Chi2(1)
        where L0 assumes pi_01 = pi_11 = pi, and L1 estimates pi_01 and pi_11 separately.
    """
    arr = np.asarray(exceptions, dtype=float)
    clean_exc = (arr[~np.isnan(arr)] > 0).astype(int)
    t_obs = len(clean_exc)

    if t_obs < 2:
        return IndependenceTestResult(
            contingency_matrix={"T00": 0, "T01": 0, "T10": 0, "T11": 0},
            p01=0.0,
            p11=0.0,
            pi_unconditional=0.0,
            lr_statistic=0.0,
            p_value=1.0,
            is_independent_5pct=True
        )

    t00 = 0
    t01 = 0
    t10 = 0
    t11 = 0

    for i in range(1, t_obs):
        prev = clean_exc[i - 1]
        curr = clean_exc[i]
        if prev == 0 and curr == 0:
            t00 += 1
        elif prev == 0 and curr == 1:
            t01 += 1
        elif prev == 1 and curr == 0:
            t10 += 1
        elif prev == 1 and curr == 1:
            t11 += 1

    total_transitions = t00 + t01 + t10 + t11
    pi = (t01 + t11) / total_transitions if total_transitions > 0 else 0.0

    p01 = t01 / (t00 + t01) if (t00 + t01) > 0 else 0.0
    p11 = t11 / (t10 + t11) if (t10 + t11) > 0 else 0.0

    # If there are zero exceptions or all exceptions, independence cannot be rejected
    if (t01 + t11) == 0 or (t00 + t10) == 0:
        lr_stat = 0.0
        p_val = 1.0
    else:
        # Log-likelihood under null hypothesis (independence: pi_01 = pi_11 = pi)
        ln_l0 = (t00 + t10) * math.log(1.0 - pi) + (t01 + t11) * math.log(pi)

        # Log-likelihood under alternative hypothesis (Markov dependence)
        ln_l1 = 0.0
        if t00 > 0 and (1.0 - p01) > 0:
            ln_l1 += t00 * math.log(1.0 - p01)
        if t01 > 0 and p01 > 0:
            ln_l1 += t01 * math.log(p01)
        if t10 > 0 and (1.0 - p11) > 0:
            ln_l1 += t10 * math.log(1.0 - p11)
        if t11 > 0 and p11 > 0:
            ln_l1 += t11 * math.log(p11)

        lr_stat = 2.0 * (ln_l1 - ln_l0)

    lr_stat = max(0.0, float(lr_stat))
    p_val = float(1.0 - stats.chi2.cdf(lr_stat, df=1))
    crit_val = float(stats.chi2.ppf(0.95, df=1))

    return IndependenceTestResult(
        contingency_matrix={"T00": t00, "T01": t01, "T10": t10, "T11": t11},
        p01=float(p01),
        p11=float(p11),
        pi_unconditional=float(pi),
        lr_statistic=lr_stat,
        p_value=p_val,
        critical_value_5pct=crit_val,
        is_independent_5pct=bool(lr_stat < crit_val and p_val > 0.05)
    )


def christoffersen_conditional_coverage_test(
    pof_result: KupiecPOFResult,
    ind_result: IndependenceTestResult
) -> ConditionalCoverageResult:
    """
    Computes Christoffersen's Joint Conditional Coverage Test:
        LR_cc = LR_uc + LR_ind ~ Chi2(2 df)
    Combines unconditional coverage accuracy and exception independence into an omnibus test.
    """
    lr_cc = float(pof_result.lr_statistic + ind_result.lr_statistic)
    p_val = float(1.0 - stats.chi2.cdf(lr_cc, df=2))
    crit_val = float(stats.chi2.ppf(0.95, df=2))  # 5.991

    return ConditionalCoverageResult(
        lr_uc=pof_result.lr_statistic,
        lr_ind=ind_result.lr_statistic,
        lr_cc=lr_cc,
        p_value=p_val,
        critical_value_5pct=crit_val,
        is_accepted_5pct=bool(lr_cc < crit_val and p_val > 0.05)
    )


def run_backtest_diagnostics(
    timeline_df: pd.DataFrame,
    confidence_level: float = 0.95,
    model_column_prefix: str = "HIST"
) -> Dict[str, Any]:
    """
    Runs the complete two-tailed diagnostic suite for a backtested model:
    1. Loss Tail (Lower Tail: Quantile 1 - confidence_level, e.g. 5% downside risk):
       - Dynamic Loss VaR threshold (or eCDF <= 1 - confidence_level)
       - Loss Tail Outliers, Binomial Test, Kupiec POF, Independence, Conditional Coverage
    2. Gain Tail (Upper Tail: Quantile confidence_level, e.g. 95% profit surge threshold):
       - Dynamic Gain VaR threshold (or eCDF >= confidence_level)
       - Gain Tail Outliers, Binomial Test, Kupiec POF, Independence, Conditional Coverage
    3. Distribution Uniformity:
       - PIT eCDF Uniformity Test (Kolmogorov-Smirnov) against Uniform(0, 1)
    """
    if timeline_df.empty:
        return {}

    ecdf_col = f"{model_column_prefix}_ECDF"
    ppls_col = f"{model_column_prefix}_PPLS"
    clean_pnl_col = "CLEAN_PNL_GBP"

    if ecdf_col not in timeline_df.columns:
        raise ValueError(f"Column '{ecdf_col}' not found in timeline dataframe.")

    clean_ecdfs = timeline_df[ecdf_col].dropna()
    p_expected = 1.0 - confidence_level

    # --- 1. Dynamic VaR & Exception Series for Both Tails ---
    if ppls_col in timeline_df.columns and clean_pnl_col in timeline_df.columns:
        def _calc_dynamic_loss_var(ppls):
            if ppls is None or len(ppls) == 0:
                return np.nan
            k = max(0, min(int(p_expected * len(ppls)), len(ppls) - 1))
            return float(ppls[k])

        def _calc_dynamic_gain_var(ppls):
            if ppls is None or len(ppls) == 0:
                return np.nan
            k = max(0, min(int(confidence_level * len(ppls)), len(ppls) - 1))
            return float(ppls[k])

        dynamic_loss_var = timeline_df[ppls_col].apply(_calc_dynamic_loss_var)
        dynamic_gain_var = timeline_df[ppls_col].apply(_calc_dynamic_gain_var)

        exceptions_loss = (timeline_df[clean_pnl_col] < dynamic_loss_var).astype(int)
        exceptions_gain = (timeline_df[clean_pnl_col] > dynamic_gain_var).astype(int)
    else:
        # Mathematical equivalence using PIT eCDF:
        # Loss breach: eCDF <= 1 - confidence_level
        # Gain surge:  eCDF >= confidence_level
        exceptions_loss = (timeline_df[ecdf_col] <= (p_expected + 1e-9)).astype(int)
        exceptions_gain = (timeline_df[ecdf_col] >= (confidence_level - 1e-9)).astype(int)
        dynamic_loss_var = None
        dynamic_gain_var = None

    # --- 2. PIT Uniformity Tests (Computed for both models when present) ---
    uniformity_res = test_ecdf_uniformity(clean_ecdfs)

    uniformity_hist = None
    if "HIST_ECDF" in timeline_df.columns:
        uniformity_hist = test_ecdf_uniformity(timeline_df["HIST_ECDF"].dropna())

    uniformity_vol = None
    if "VOL_SCALED_ECDF" in timeline_df.columns:
        uniformity_vol = test_ecdf_uniformity(timeline_df["VOL_SCALED_ECDF"].dropna())

    # --- 3. Loss Tail Diagnostics ---
    binom_loss = evaluate_binomial_outliers(
        exceptions_or_ecdfs=exceptions_loss,
        confidence_level=confidence_level,
        is_exception_indicator=True
    )
    pof_loss = kupiec_pof_test(exceptions=exceptions_loss, confidence_level=confidence_level)
    ind_loss = kupiec_independence_test(exceptions=exceptions_loss)
    cc_loss = christoffersen_conditional_coverage_test(pof_loss, ind_loss)

    # --- 4. Gain Tail Diagnostics ---
    binom_gain = evaluate_binomial_outliers(
        exceptions_or_ecdfs=exceptions_gain,
        confidence_level=confidence_level,
        is_exception_indicator=True
    )
    pof_gain = kupiec_pof_test(exceptions=exceptions_gain, confidence_level=confidence_level)
    ind_gain = kupiec_independence_test(exceptions=exceptions_gain)
    cc_gain = christoffersen_conditional_coverage_test(pof_gain, ind_gain)

    loss_stats = {
        "tail_name": "Loss Tail (Downside)",
        "confidence_level": confidence_level,
        "tail_probability": p_expected,
        "dynamic_var": dynamic_loss_var,
        "binomial": binom_loss,
        "kupiec_pof": pof_loss,
        "independence": ind_loss,
        "conditional_coverage": cc_loss,
        "exceptions_series": exceptions_loss,
    }

    gain_stats = {
        "tail_name": "Gain Tail (Upside)",
        "confidence_level": confidence_level,
        "tail_probability": p_expected,
        "dynamic_var": dynamic_gain_var,
        "binomial": binom_gain,
        "kupiec_pof": pof_gain,
        "independence": ind_gain,
        "conditional_coverage": cc_gain,
        "exceptions_series": exceptions_gain,
    }

    return {
        "confidence_level": confidence_level,
        "model_prefix": model_column_prefix,
        "uniformity": uniformity_res,
        "uniformity_hist": uniformity_hist,
        "uniformity_vol": uniformity_vol,
        "loss_tail": loss_stats,
        "gain_tail": gain_stats,
        # Backwards compatibility shortcuts
        "dynamic_var": dynamic_loss_var,
        "dynamic_loss_var": dynamic_loss_var,
        "dynamic_gain_var": dynamic_gain_var,
        "binomial": binom_loss,
        "kupiec_pof": pof_loss,
        "independence": ind_loss,
        "conditional_coverage": cc_loss,
        "exceptions_series": exceptions_loss,
        "exceptions_series_gain": exceptions_gain,
    }


# =====================================================================
# 1-Year Portfolio Backtesting Timeline Engine
# =====================================================================

def generate_portfolio_backtest_timeline(
    positions: Dict[str, float],
    prices_gbp: pd.DataFrame,
    asof_date: Optional[str] = None,
    backtest_days: int = 252,
    lookback_days: int = 260,
    confidence_level: float = 0.95,
    ewma_lambda: float = 0.94
) -> pd.DataFrame:
    """
    Executes a rolling 1-year (252 trading days) backtest on a single baseline portfolio snapshot:
    1. Holds the target portfolio composition strictly constant (ZERO position impact).
    2. Computes the realized Clean / Hypothetical P&L between t-1 and t for each day.
    3. Simulates the 1-day scenario PPL distribution holding the exact same portfolio constant.
    4. Computes daily Value-at-Risk (95% & 99%) for both Historical Simulation and Vol-Scaled VaR.
    5. Evaluates daily empirical CDF (eCDF) Probability Integral Transform values.

    Parameters:
    -----------
    positions : Dict[str, float]
        Dictionary of {ticker: shares} locked at asof_date.
    prices_gbp : pd.DataFrame
        Historical market price matrix in GBP.
    asof_date : Optional[str]
        Target valuation date (defaults to the latest date in prices_gbp).
    backtest_days : int
        Number of daily backtesting points to generate (default: 252 trading days = 1 year).
    lookback_days : int
        Rolling simulation scenario window (default: 260 market observations).
    confidence_level : float
        Primary confidence level for VaR threshold tracking (default: 0.95).
    ewma_lambda : float
        Decay factor for EWMA Volatility-Scaled VaR (default: 0.94).

    Returns:
    --------
    pd.DataFrame:
        DataFrame containing:
        - DATE: Evaluation date t
        - PREV_DATE: Baseline date t-1
        - PORTFOLIO_VALUE_GBP: Portfolio value at t-1 (PV)
        - CLEAN_PNL_GBP: Realized clean P&L (zero position impact)
        - CLEAN_RETURN_PCT: Clean P&L as a percentage of PV
        - HIST_VAR_GBP: Historical Simulation VaR
        - HIST_ECDF: Historical Simulation empirical CDF value in [0, 1]
        - HIST_BREACH: Boolean flag if Clean P&L breached Historical VaR
        - VOL_SCALED_VAR_GBP: Vol-Scaled VaR (EWMA lambda)
        - VOL_SCALED_ECDF: Vol-Scaled empirical CDF value in [0, 1]
        - VOL_SCALED_BREACH: Boolean flag if Clean P&L breached Vol-Scaled VaR
    """
    if prices_gbp is None or prices_gbp.empty:
        raise ValueError("Price history dataframe cannot be empty.")
    if not positions:
        raise ValueError("Positions dictionary cannot be empty.")

    sorted_prices = prices_gbp.sort_index()

    if asof_date:
        asof_str = str(asof_date)[:10]
        date_strs = [str(d)[:10] for d in sorted_prices.index]
        if asof_str in date_strs:
            target_idx = date_strs.index(asof_str)
            sorted_prices = sorted_prices.iloc[:target_idx + 1]

    # Filter to active tickers with price data
    latest_row = sorted_prices.iloc[-1]
    active_pos = {t: float(sh) for t, sh in positions.items() if t in sorted_prices.columns and not pd.isna(latest_row[t]) and sh > 0}
    if not active_pos:
        raise ValueError("No active positions have corresponding price data in the price matrix.")

    active_tickers = sorted(list(active_pos.keys()))
    pos_shares = pd.Series({t: active_pos[t] for t in active_tickers})

    total_obs = len(sorted_prices)
    min_required_obs = 30  # Minimum observations to run any rolling backtest
    if total_obs < min_required_obs:
        raise ValueError(f"Insufficient price history: found {total_obs} dates, need at least {min_required_obs}.")

    # Determine feasible backtest window and scenario lookback
    eff_lookback = min(lookback_days, max(20, total_obs // 2))
    eff_backtest = min(backtest_days, total_obs - eff_lookback - 1)
    if eff_backtest <= 0:
        eff_backtest = max(5, total_obs - 22)
        eff_lookback = total_obs - eff_backtest - 1

    start_eval_idx = total_obs - eff_backtest
    price_subset = sorted_prices[active_tickers]
    log_returns_all = np.log(price_subset / price_subset.shift(1)).fillna(0.0)

    # Pre-calculate EWMA volatility matrix across the entire price matrix for speed
    decay = ewma_lambda
    alpha_ewma = 1.0 - decay
    var_matrix = (log_returns_all ** 2).ewm(alpha=alpha_ewma, min_periods=5, adjust=False).mean()
    vol_matrix_all = np.sqrt(var_matrix).replace(0.0, np.nan).bfill().fillna(1e-4)
    residuals_all = (log_returns_all / vol_matrix_all).fillna(0.0)

    timeline_rows = []
    alpha_tail = 1.0 - confidence_level

    for idx in range(start_eval_idx, total_obs):
        t_curr_date = sorted_prices.index[idx]
        t_prev_date = sorted_prices.index[idx - 1]

        p_prev = price_subset.iloc[idx - 1]
        p_curr = price_subset.iloc[idx]

        # 1. Realized Clean / Hypothetical P&L (Zero position impact on frozen positions)
        pv_baseline = float((pos_shares * p_prev).sum())
        if pv_baseline <= 0:
            continue

        clean_pnl = float((pos_shares * (p_curr - p_prev)).sum())
        clean_ret_pct = (clean_pnl / pv_baseline) * 100.0

        # 2. Slice historical scenario returns up to t-1
        slice_start = max(1, idx - eff_lookback)
        hist_rets = log_returns_all.iloc[slice_start:idx]
        m_scenarios = len(hist_rets)
        if m_scenarios < 10:
            continue

        pos_values_prev = pos_shares * p_prev

        # --- Model 1: Historical Simulation ---
        pnl_hist_matrix = (np.exp(hist_rets) - 1.0).mul(pos_values_prev, axis=1)
        sim_ppls_hist = pnl_hist_matrix.sum(axis=1).values
        sorted_ppls_hist = np.sort(sim_ppls_hist)

        k_loss_idx = max(0, min(int(alpha_tail * m_scenarios), m_scenarios - 1))
        k_gain_idx = max(0, min(int(confidence_level * m_scenarios), m_scenarios - 1))
        var_hist_loss = float(sorted_ppls_hist[k_loss_idx])
        var_hist_gain = float(sorted_ppls_hist[k_gain_idx])
        ecdf_hist = float(np.searchsorted(sorted_ppls_hist, clean_pnl, side="right") / m_scenarios)
        ecdf_hist = max(0.0, min(1.0, ecdf_hist))

        # --- Model 2: Volatility-Scaled VaR (EWMA) ---
        curr_vol = vol_matrix_all.iloc[idx - 1]
        hist_residuals = residuals_all.iloc[slice_start:idx]
        scaled_rets = hist_residuals.mul(curr_vol, axis=1)
        pnl_vol_matrix = (np.exp(scaled_rets) - 1.0).mul(pos_values_prev, axis=1)
        sim_ppls_vol = pnl_vol_matrix.sum(axis=1).values
        sorted_ppls_vol = np.sort(sim_ppls_vol)

        var_vol_loss = float(sorted_ppls_vol[k_loss_idx])
        var_vol_gain = float(sorted_ppls_vol[k_gain_idx])
        ecdf_vol = float(np.searchsorted(sorted_ppls_vol, clean_pnl, side="right") / m_scenarios)
        ecdf_vol = max(0.0, min(1.0, ecdf_vol))

        timeline_rows.append({
            "DATE": str(t_curr_date)[:10],
            "PREV_DATE": str(t_prev_date)[:10],
            "PORTFOLIO_VALUE_GBP": round(pv_baseline, 2),
            "CLEAN_PNL_GBP": round(clean_pnl, 2),
            "CLEAN_RETURN_PCT": round(clean_ret_pct, 4),
            "HIST_VAR_GBP": round(var_hist_loss, 2),
            "HIST_GAIN_VAR_GBP": round(var_hist_gain, 2),
            "HIST_VAR_PCT": round((var_hist_loss / pv_baseline) * 100.0, 4),
            "HIST_ECDF": round(ecdf_hist, 6),
            "HIST_BREACH": bool(clean_pnl < var_hist_loss),
            "HIST_GAIN_BREACH": bool(clean_pnl > var_hist_gain),
            "HIST_PPLS": sorted_ppls_hist,
            "VOL_SCALED_VAR_GBP": round(var_vol_loss, 2),
            "VOL_SCALED_GAIN_VAR_GBP": round(var_vol_gain, 2),
            "VOL_SCALED_VAR_PCT": round((var_vol_loss / pv_baseline) * 100.0, 4),
            "VOL_SCALED_ECDF": round(ecdf_vol, 6),
            "VOL_SCALED_BREACH": bool(clean_pnl < var_vol_loss),
            "VOL_SCALED_GAIN_BREACH": bool(clean_pnl > var_vol_gain),
            "VOL_SCALED_PPLS": sorted_ppls_vol,
            "SCENARIOS_COUNT": m_scenarios
        })

    df = pd.DataFrame(timeline_rows)
    if not df.empty:
        df["DATE"] = pd.to_datetime(df["DATE"])
    return df
