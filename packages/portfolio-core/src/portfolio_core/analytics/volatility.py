"""
Volatility Estimation and Time-Series Analytics Module.
Implements Rolling Sample Standard Deviation, RiskMetrics EWMA (Exponentially Weighted Moving Average),
Parkinson High-Low Volatility, and Volatility Scaling Multipliers.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import math
import numpy as np
import pandas as pd


def calculate_sample_volatility(
    returns: Union[pd.Series, pd.DataFrame],
    window: int = 30,
    annualize: bool = True,
    trading_days: int = 252
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculates rolling sample standard deviation over a fixed lookback window.
    Uses min_periods=max(2, window // 3) to prevent NaN gaps at series start.
    """
    min_p = max(2, window // 3)
    roll_std = returns.rolling(window=window, min_periods=min_p).std(ddof=1)
    # Forward-fill / backfill initial warm-up period cleanly
    roll_std = roll_std.bfill().ffill()
    if annualize:
        roll_std = roll_std * math.sqrt(trading_days)
    return roll_std


def calculate_ewma_volatility(
    returns: Union[pd.Series, pd.DataFrame],
    decay_factor: float = 0.94,
    annualize: bool = True,
    trading_days: int = 252
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculates RiskMetrics Exponentially Weighted Moving Average (EWMA) volatility:
    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2
    
    Parameters:
        returns: Daily simple or log return series / DataFrame.
        decay_factor: Smoothing parameter (lambda), default 0.94.
        annualize: If True, scales daily volatility by sqrt(252).
        trading_days: Number of trading days per year (default 252).
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: calculate_ewma_volatility(col, decay_factor, annualize, trading_days))

    clean_rets = returns.dropna()
    if clean_rets.empty:
        return returns.copy()

    n = len(clean_rets)
    var_series = np.zeros(n)

    # Initialize variance with initial sample variance or first squared return
    init_window = min(20, max(2, n))
    var_series[0] = float(np.var(clean_rets.iloc[:init_window], ddof=1)) if init_window > 1 else float(clean_rets.iloc[0] ** 2)
    if var_series[0] <= 0:
        var_series[0] = 1e-6

    ret_vals = clean_rets.values
    one_minus_lambda = 1.0 - decay_factor

    for t in range(1, n):
        var_series[t] = decay_factor * var_series[t - 1] + one_minus_lambda * (ret_vals[t - 1] ** 2)

    vol_series = pd.Series(np.sqrt(var_series), index=clean_rets.index)
    if annualize:
        vol_series = vol_series * math.sqrt(trading_days)

    return vol_series.reindex(returns.index).bfill().ffill()


def calculate_parkinson_volatility(
    high_prices: pd.Series,
    low_prices: pd.Series,
    window: int = 30,
    annualize: bool = True,
    trading_days: int = 252
) -> pd.Series:
    """
    Calculates Parkinson extreme-value volatility estimator using intraday High and Low prices:
    sigma^2 = (1 / (4 * ln(2))) * sum(ln(High / Low)^2) / window
    """
    log_hl_ratio_sq = (np.log(high_prices / low_prices)) ** 2
    factor = 1.0 / (4.0 * math.log(2.0))
    park_var = factor * log_hl_ratio_sq.rolling(window=window, min_periods=max(2, window // 3)).mean()
    park_vol = np.sqrt(park_var).bfill().ffill()
    if annualize:
        park_vol = park_vol * math.sqrt(trading_days)
    return park_vol


def calculate_scaling_factors(
    vol_series: Union[pd.Series, pd.DataFrame],
    asof_date: Optional[Union[str, pd.Timestamp]] = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculates the volatility scaling factor:
    Multiplier_t = sigma_today / sigma_t
    Values < 1.0 indicate historical shocks are dampened/compressed.
    Values > 1.0 indicate historical shocks are amplified.
    """
    if asof_date and asof_date in vol_series.index:
        current_vol = vol_series.loc[asof_date]
    else:
        current_vol = vol_series.iloc[-1]

    scaling_ratio = current_vol / vol_series
    return scaling_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)


def compute_volatility_summary_metrics(
    returns: pd.Series,
    ewma_lambda: float = 0.94,
    rolling_windows: List[int] = [10, 20, 30, 60, 90],
    trading_days: int = 252
) -> Dict[str, Any]:
    """
    Computes a comprehensive dictionary of volatility metrics for a single asset series.
    """
    clean_rets = returns.dropna()
    if len(clean_rets) < 5:
        return {}

    ewma_vol = calculate_ewma_volatility(clean_rets, decay_factor=ewma_lambda, annualize=True, trading_days=trading_days)
    full_sample_vol = clean_rets.std() * np.sqrt(trading_days)

    rolling_vols = {}
    for w in rolling_windows:
        rolling_vols[f"roll_{w}d"] = calculate_sample_volatility(
            clean_rets, window=w, annualize=True, trading_days=trading_days
        )

    roll_30 = rolling_vols.get("roll_30d", clean_rets.rolling(30).std() * np.sqrt(trading_days))

    latest_ewma = float(ewma_vol.iloc[-1])
    latest_30d = float(roll_30.iloc[-1]) if not roll_30.empty else latest_ewma
    mean_ewma = float(ewma_vol.mean())
    median_ewma = float(ewma_vol.median())
    min_ewma = float(ewma_vol.min())
    max_ewma = float(ewma_vol.max())

    # Percentile rank of current volatility in the period
    pct_rank = float((ewma_vol <= latest_ewma).mean() * 100.0)
    pct_days_higher = float((ewma_vol > latest_ewma).mean() * 100.0)

    # Scaling ratios
    scaling_factors = latest_ewma / ewma_vol
    min_scale = float(scaling_factors.min())
    max_scale = float(scaling_factors.max())
    mean_scale = float(scaling_factors.mean())

    return {
        "latest_ewma_vol_pct": latest_ewma * 100.0,
        "latest_30d_vol_pct": latest_30d * 100.0,
        "full_sample_vol_pct": full_sample_vol * 100.0,
        "mean_ewma_vol_pct": mean_ewma * 100.0,
        "median_ewma_vol_pct": median_ewma * 100.0,
        "min_ewma_vol_pct": min_ewma * 100.0,
        "max_ewma_vol_pct": max_ewma * 100.0,
        "current_vol_percentile_rank": pct_rank,
        "pct_days_higher_vol": pct_days_higher,
        "scaling_ratio_mean": mean_scale,
        "scaling_ratio_min": min_scale,
        "scaling_ratio_max": max_scale,
        "ewma_series": ewma_vol,
        "rolling_series": rolling_vols,
        "scaling_series": scaling_factors
    }
