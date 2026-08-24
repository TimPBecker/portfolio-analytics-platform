"""
Financial Statistics, Return Distribution Diagnostics, and Normality Testing Module.
Provides robust empirical calculations for asset price levels, returns, moments,
distribution fits (Normal, Student-t, KDE), and Q-Q diagnostics.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy import stats


def compute_asset_returns(
    prices: Union[pd.Series, pd.DataFrame],
    method: str = "log"
) -> Union[pd.Series, pd.DataFrame]:
    """
    Computes daily returns from price series.
    - 'log': r_t = ln(P_t / P_{t-1})
    - 'pct' or 'simple': r_t = (P_t / P_{t-1}) - 1
    """
    if method.lower() == "log":
        rets = np.log(prices / prices.shift(1))
    else:
        rets = (prices / prices.shift(1)) - 1.0

    return rets.dropna(how="all")


def compute_distribution_metrics(
    returns: pd.Series,
    trading_days: int = 252
) -> Dict[str, Any]:
    """
    Computes summary moments, distribution parameters, and normality tests for a return series.
    """
    clean_rets = returns.dropna()
    n = len(clean_rets)
    if n < 5:
        return {}

    mean_daily = float(clean_rets.mean())
    median_daily = float(clean_rets.median())
    std_daily = float(clean_rets.std())
    vol_ann = std_daily * np.sqrt(trading_days)
    mean_ann = mean_daily * trading_days

    skewness = float(stats.skew(clean_rets))
    kurtosis_excess = float(stats.kurtosis(clean_rets, fisher=True))  # 0 for Normal

    min_val = float(clean_rets.min())
    max_val = float(clean_rets.max())

    # Percentiles
    p01 = float(np.percentile(clean_rets, 1.0))
    p05 = float(np.percentile(clean_rets, 5.0))
    p25 = float(np.percentile(clean_rets, 25.0))
    p75 = float(np.percentile(clean_rets, 75.0))
    p95 = float(np.percentile(clean_rets, 95.0))
    p99 = float(np.percentile(clean_rets, 99.0))

    # Jarque-Bera Normality Test
    jb_stat, jb_pvalue = stats.jarque_bera(clean_rets)
    is_normal_5pct = bool(jb_pvalue > 0.05)

    # D'Agostino and Pearson's test
    try:
        norm_stat, norm_pvalue = stats.normaltest(clean_rets)
    except Exception:
        norm_stat, norm_pvalue = 0.0, 1.0

    # Student-t distribution fit
    try:
        df_t, loc_t, scale_t = stats.t.fit(clean_rets)
    except Exception:
        df_t, loc_t, scale_t = np.nan, mean_daily, std_daily

    return {
        "count": n,
        "mean_daily_pct": mean_daily * 100.0,
        "median_daily_pct": median_daily * 100.0,
        "mean_annualized_pct": mean_ann * 100.0,
        "std_daily_pct": std_daily * 100.0,
        "vol_annualized_pct": vol_ann * 100.0,
        "skewness": skewness,
        "kurtosis_excess": kurtosis_excess,
        "min_pct": min_val * 100.0,
        "max_pct": max_val * 100.0,
        "range_pct": (max_val - min_val) * 100.0,
        "p01_pct": p01 * 100.0,
        "p05_pct": p05 * 100.0,
        "p25_pct": p25 * 100.0,
        "p75_pct": p75 * 100.0,
        "p95_pct": p95 * 100.0,
        "p99_pct": p99 * 100.0,
        "jb_statistic": float(jb_stat),
        "jb_pvalue": float(jb_pvalue),
        "is_normal": is_normal_5pct,
        "norm_statistic": float(norm_stat),
        "norm_pvalue": float(norm_pvalue),
        "student_t_df": float(df_t) if not np.isnan(df_t) else None,
        "student_t_loc": float(loc_t),
        "student_t_scale": float(scale_t)
    }


def generate_density_curves(
    returns: pd.Series,
    num_points: int = 300
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Generates x-grid and evaluated probability density curves:
    1. x_grid: Return values range
    2. kde_y: Kernel Density Estimate
    3. norm_y: Fitted Gaussian Normal PDF
    4. t_y: Fitted Student-t PDF (if fit converges)
    """
    clean_rets = returns.dropna().values
    if len(clean_rets) < 5:
        return np.array([]), np.array([]), np.array([]), None

    min_x = np.percentile(clean_rets, 0.1) * 1.2
    max_x = np.percentile(clean_rets, 99.9) * 1.2
    if min_x == max_x:
        min_x -= 0.05
        max_x += 0.05

    x_grid = np.linspace(min_x, max_x, num_points)

    # 1. KDE
    try:
        kde = stats.gaussian_kde(clean_rets)
        kde_y = kde.evaluate(x_grid)
    except Exception:
        kde_y = np.zeros_like(x_grid)

    # 2. Fitted Normal PDF
    mu = float(np.mean(clean_rets))
    sigma = float(np.std(clean_rets))
    norm_y = stats.norm.pdf(x_grid, loc=mu, scale=sigma)

    # 3. Fitted Student-t PDF
    t_y = None
    try:
        df_t, loc_t, scale_t = stats.t.fit(clean_rets)
        t_y = stats.t.pdf(x_grid, df=df_t, loc=loc_t, scale=scale_t)
    except Exception:
        t_y = None

    return x_grid, kde_y, norm_y, t_y


def compute_qq_plot_data(returns: pd.Series) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Computes theoretical normal quantiles and sample ordered quantiles for Q-Q plotting.
    Also returns slope and intercept for the 45-degree reference line.
    """
    clean_rets = returns.dropna().values
    if len(clean_rets) < 5:
        return np.array([]), np.array([]), 1.0, 0.0

    (osm, osr), (slope, intercept, r) = stats.probplot(clean_rets, dist="norm")
    return np.asarray(osm), np.asarray(osr), float(slope), float(intercept)
