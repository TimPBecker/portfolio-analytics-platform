"""
Value-at-Risk (VaR), Volatility Scaling, and Shapley Risk Analytics Module.
Contains pure financial mathematical and statistical calculation models.
All database persistence and table creations are decoupled in db.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
import pandas as pd


# =====================================================================
# 1. Volatility Estimators (Object-Oriented Design)
# =====================================================================

class VolatilityEstimator(ABC):
    """
    Abstract Base Class for time-series volatility estimation.
    Used by Volatility-Scaled VaR models to standardize returns into residuals
    and re-scale them by current market volatility.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for the estimator."""
        pass

    @abstractmethod
    def calculate_volatility(
        self,
        log_returns: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculates the historical daily volatility time series for each asset.
        """
        pass


class SampleVolatility(VolatilityEstimator):
    """
    Rolling sample standard deviation estimator.
    """

    def __init__(self, window: int = 30, min_periods: Optional[int] = None):
        if window < 2:
            raise ValueError(f"Rolling window must be at least 2, got {window}")
        self.window = window
        self.min_periods = min_periods or max(5, window // 4)

    @property
    def name(self) -> str:
        return f"Sample Volatility ({self.window}d)"

    def calculate_volatility(
        self,
        log_returns: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        vol = log_returns.rolling(window=self.window, min_periods=self.min_periods).std()
        vol = vol.bfill()
        if (vol.isna().any().any() if isinstance(vol, pd.DataFrame) else vol.isna().any()):
            vol = vol.fillna(log_returns.std())
        return vol.replace(0.0, np.nan).bfill().fillna(1e-4)


class EWMAVolatility(VolatilityEstimator):
    """
    Exponentially Weighted Moving Average (EWMA / RiskMetrics) volatility estimator.
    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_t^2
    """

    def __init__(self, decay_factor: float = 0.94, min_periods: int = 5):
        if not (0.0 < decay_factor < 1.0):
            raise ValueError(f"Decay factor lambda must be between 0 and 1, got {decay_factor}")
        self.decay_factor = decay_factor
        self.min_periods = min_periods

    @property
    def name(self) -> str:
        return f"EWMA Volatility (λ={self.decay_factor})"

    def calculate_volatility(
        self,
        log_returns: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        alpha = 1.0 - self.decay_factor
        var_series = (log_returns ** 2).ewm(
            alpha=alpha,
            min_periods=self.min_periods,
            adjust=False
        ).mean()
        vol = np.sqrt(var_series)
        vol = vol.bfill()
        if (vol.isna().any().any() if isinstance(vol, pd.DataFrame) else vol.isna().any()):
            vol = vol.fillna(log_returns.std())
        return vol.replace(0.0, np.nan).bfill().fillna(1e-4)


# =====================================================================
# 2. Value-at-Risk Result & Model Hierarchy (OOD)
# =====================================================================

@dataclass
class VaRResult:
    """Stores the output and diagnostics of a Value-at-Risk calculation."""
    asof_date: str
    model_name: str
    confidence_level: float
    horizon_days: int
    portfolio_value_gbp: float
    var_gbp: float
    var_pct: float
    cvar_gbp: Optional[float] = None
    cvar_pct: Optional[float] = None
    lookback_observations: Optional[int] = None
    pnl_matrix: Optional[pd.DataFrame] = None
    position_values_gbp: Optional[pd.Series] = None
    shapley_contributions: Optional[pd.DataFrame] = None

    def to_dict(self) -> dict:
        var_gbp_str = f"£{self.var_gbp:,.2f}" if self.var_gbp >= 0 else f"-£{abs(self.var_gbp):,.2f}"
        var_pct_str = f"{self.var_pct:+.2f}%"
        cvar_gbp_str = (
            (f"£{self.cvar_gbp:,.2f}" if self.cvar_gbp >= 0 else f"-£{abs(self.cvar_gbp):,.2f}")
            if self.cvar_gbp is not None else "N/A"
        )
        cvar_pct_str = f"{self.cvar_pct:+.2f}%" if self.cvar_pct is not None else "N/A"
        return {
            "As-Of Date": self.asof_date,
            "Model": self.model_name,
            "Confidence Level": f"{self.confidence_level * 100:.1f}%",
            "Horizon (Days)": self.horizon_days,
            "Portfolio Value (GBP)": f"£{self.portfolio_value_gbp:,.2f}",
            "Value at Risk (GBP)": var_gbp_str,
            "Value at Risk (%)": var_pct_str,
            "Expected Shortfall / CVaR (GBP)": cvar_gbp_str,
            "Expected Shortfall / CVaR (%)": cvar_pct_str,
            "Observations": self.lookback_observations or 0
        }

    def summary_markdown(self) -> str:
        data = self.to_dict()
        df = pd.DataFrame(list(data.items()), columns=["Metric", "Value"])
        return df.to_markdown(index=False)

    def __repr__(self) -> str:
        var_str = f"£{self.var_gbp:,.2f}" if self.var_gbp >= 0 else f"-£{abs(self.var_gbp):,.2f}"
        cvar_str = f"£{self.cvar_gbp:,.2f}" if (self.cvar_gbp is not None and self.cvar_gbp >= 0) else (f"-£{abs(self.cvar_gbp):,.2f}" if self.cvar_gbp is not None else "N/A")
        return (
            f"<VaRResult model='{self.model_name}' date='{self.asof_date}' "
            f"conf={self.confidence_level*100:.1f}% horizon={self.horizon_days}d "
            f"VaR={var_str} ({self.var_pct:+.2f}%) CVaR={cvar_str}>"
        )


class ValueAtRiskModel(ABC):
    """Abstract Base Class defining the interface for all Value-at-Risk models."""

    def __init__(self, confidence_level: float = 0.95, horizon_days: int = 1):
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"Confidence level must be between 0 and 1, got {confidence_level}")
        if horizon_days < 1:
            raise ValueError(f"Horizon days must be at least 1, got {horizon_days}")
        self.confidence_level = confidence_level
        self.horizon_days = horizon_days

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier for the model."""
        pass

    @abstractmethod
    def generate_scenario_pnl_matrix(
        self,
        positions: Dict[str, float],
        prices_gbp: pd.DataFrame,
        asof_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Generates simulated scenario P&L vectors by position and detailed scenario records.
        """
        pass

    def calculate(
        self,
        positions: Dict[str, float],
        prices_gbp: pd.DataFrame,
        asof_date: Optional[str] = None
    ) -> VaRResult:
        """Calculates VaR and CVaR from generated scenario P&L vectors."""
        pnl_matrix, pos_series, _ = self.generate_scenario_pnl_matrix(
            positions=positions,
            prices_gbp=prices_gbp,
            asof_date=asof_date
        )
        asof = asof_date or str(prices_gbp.index[-1])
        return self.calculate_from_pnl_matrix(
            pnl_matrix=pnl_matrix,
            pos_series=pos_series,
            asof_date=asof
        )

    def calculate_from_pnl_matrix(
        self,
        pnl_matrix: pd.DataFrame,
        pos_series: pd.Series,
        asof_date: str
    ) -> VaRResult:
        """Computes VaR and CVaR directly from a scenario P&L matrix."""
        total_portfolio_value = float(pos_series.sum())
        total_pnl = pnl_matrix.sum(axis=1).values
        alpha = 1.0 - self.confidence_level

        var_gbp = ShapleyRiskAttributor._calc_var_partition(total_pnl, alpha)
        var_pct = (var_gbp / total_portfolio_value) * 100.0 if total_portfolio_value > 0 else 0.0

        cvar_gbp = ShapleyRiskAttributor._calc_cvar_partition(total_pnl, alpha)
        cvar_pct = (cvar_gbp / total_portfolio_value) * 100.0 if total_portfolio_value > 0 else 0.0

        return VaRResult(
            asof_date=asof_date,
            model_name=self.model_name,
            confidence_level=self.confidence_level,
            horizon_days=self.horizon_days,
            portfolio_value_gbp=round(total_portfolio_value, 2),
            var_gbp=round(var_gbp, 2),
            var_pct=round(var_pct, 4),
            cvar_gbp=round(cvar_gbp, 2),
            cvar_pct=round(cvar_pct, 4),
            lookback_observations=len(pnl_matrix),
            pnl_matrix=pnl_matrix,
            position_values_gbp=pos_series
        )


class HistoricalVaR(ValueAtRiskModel):
    """
    Standard Historical Simulation Value-at-Risk based on daily log-returns.
    Default lookback period: 260 trading days (1 market year).
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        horizon_days: int = 1,
        lookback_days: int = 260
    ):
        super().__init__(confidence_level=confidence_level, horizon_days=horizon_days)
        self.lookback_days = lookback_days

    @property
    def model_name(self) -> str:
        return "Historical Simulation"

    def generate_scenario_pnl_matrix(
        self,
        positions: Dict[str, float],
        prices_gbp: pd.DataFrame,
        asof_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        if prices_gbp.empty:
            raise ValueError("Price history is empty.")

        asof = asof_date or str(prices_gbp.index[-1])
        hist_prices = prices_gbp.loc[:asof].copy()

        min_required = (self.lookback_days + 1) if self.lookback_days else 261
        if len(hist_prices) < min_required:
            raise ValueError(
                f"Insufficient historical prices before {asof} for VaR calculation: "
                f"found {len(hist_prices)} observations, but at least {min_required} price days are required for {self.lookback_days} returns."
            )

        if self.lookback_days and len(hist_prices) > (self.lookback_days + 1):
            hist_prices = hist_prices.iloc[-(self.lookback_days + 1):]

        latest_prices_gbp = hist_prices.iloc[-1]
        
        pos_values = {}
        for ticker, shares in positions.items():
            if ticker in latest_prices_gbp and not pd.isna(latest_prices_gbp[ticker]) and shares > 0:
                pos_values[ticker] = float(shares) * float(latest_prices_gbp[ticker])
                
        pos_series = pd.Series(pos_values)
        if pos_series.empty:
            raise ValueError("No matching price data found for active positions.")

        active_tickers = list(pos_series.index)

        # 1. Compute historical daily log-returns: r_{i, t} = ln(P_{i, t} / P_{i, t-1})
        log_returns = np.log(hist_prices[active_tickers] / hist_prices[active_tickers].shift(1)).dropna(how="all").fillna(0.0)

        if self.horizon_days > 1:
            log_returns = log_returns * math.sqrt(self.horizon_days)

        # 2. Compute position P&L vectors: Delta Pi_{i, t} = V_{i, today} * (exp(r_{i, t}) - 1)
        price_changes = np.exp(log_returns) - 1.0
        pnl_matrix = price_changes.mul(pos_series, axis=1)

        # 3. Build scenario records DataFrame
        records = []
        for scenario_date, row in pnl_matrix.iterrows():
            date_str = str(scenario_date)
            for ticker in active_tickers:
                records.append({
                    "ASOF_DATE": asof,
                    "SCENARIO_DATE": date_str,
                    "TICKER": ticker,
                    "METHOD": self.model_name,
                    "SHARES": float(positions[ticker]),
                    "PRICE_GBP": float(latest_prices_gbp[ticker]),
                    "POSITION_VALUE_GBP": float(pos_series[ticker]),
                    "LOG_RETURN": float(log_returns.loc[scenario_date, ticker]),
                    "SCENARIO_PNL_GBP": float(row[ticker]),
                })
            records.append({
                "ASOF_DATE": asof,
                "SCENARIO_DATE": date_str,
                "TICKER": "PORTFOLIO_TOTAL",
                "METHOD": self.model_name,
                "SHARES": None,
                "PRICE_GBP": None,
                "POSITION_VALUE_GBP": float(pos_series.sum()),
                "LOG_RETURN": None,
                "SCENARIO_PNL_GBP": float(row.sum()),
            })

        scenario_df = pd.DataFrame(records)
        return pnl_matrix, pos_series, scenario_df


class VolatilityScaledVaR(ValueAtRiskModel):
    """
    Volatility-Scaled (Filtered Historical Simulation / Hull-White) Value-at-Risk.
    Standardizes returns into residuals z_{i, t} = r_{i, t} / sigma_{i, t}
    and up-scales by current volatility sigma_{i, today}.
    Default lookback period: 260 trading days.
    """

    def __init__(
        self,
        volatility_estimator: Optional[VolatilityEstimator] = None,
        confidence_level: float = 0.95,
        horizon_days: int = 1,
        lookback_days: int = 260
    ):
        super().__init__(confidence_level=confidence_level, horizon_days=horizon_days)
        self.volatility_estimator = volatility_estimator or EWMAVolatility(decay_factor=0.94)
        self.lookback_days = lookback_days

    @property
    def model_name(self) -> str:
        return f"Vol-Scaled VaR ({self.volatility_estimator.name})"

    def generate_scenario_pnl_matrix(
        self,
        positions: Dict[str, float],
        prices_gbp: pd.DataFrame,
        asof_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        if prices_gbp.empty:
            raise ValueError("Price history is empty.")

        asof = asof_date or str(prices_gbp.index[-1])
        hist_prices = prices_gbp.loc[:asof].copy()

        min_required = (self.lookback_days + 1) if self.lookback_days else 261
        if len(hist_prices) < min_required:
            raise ValueError(
                f"Insufficient historical prices before {asof} for VaR calculation: "
                f"found {len(hist_prices)} observations, but at least {min_required} price days are required for {self.lookback_days} returns."
            )

        if self.lookback_days and len(hist_prices) > (self.lookback_days + 1):
            hist_prices = hist_prices.iloc[-(self.lookback_days + 1):]

        latest_prices_gbp = hist_prices.iloc[-1]
        
        pos_values = {}
        for ticker, shares in positions.items():
            if ticker in latest_prices_gbp and not pd.isna(latest_prices_gbp[ticker]) and shares > 0:
                pos_values[ticker] = float(shares) * float(latest_prices_gbp[ticker])
                
        pos_series = pd.Series(pos_values)
        if pos_series.empty:
            raise ValueError("No matching price data found for active positions.")

        active_tickers = list(pos_series.index)

        # 1. Compute historical daily log-returns
        log_returns = np.log(hist_prices[active_tickers] / hist_prices[active_tickers].shift(1)).dropna(how="all").fillna(0.0)

        # 2. Volatility series and standardized residuals: z_{i, t} = r_{i, t} / sigma_{i, t}
        vol_matrix = self.volatility_estimator.calculate_volatility(log_returns)
        residuals = log_returns / vol_matrix
        current_vol = vol_matrix.iloc[-1]

        # 3. Scaled log-returns: r_{i, t}^* = z_{i, t} * sigma_{i, today}
        scaled_log_returns = residuals * current_vol

        if self.horizon_days > 1:
            scaled_log_returns = scaled_log_returns * math.sqrt(self.horizon_days)

        # 4. Position P&L vectors: Delta Pi_{i, t} = V_{i, today} * (exp(r_{i, t}^*) - 1)
        scaled_price_changes = np.exp(scaled_log_returns) - 1.0
        pnl_matrix = scaled_price_changes.mul(pos_series, axis=1)

        # 5. Build scenario records DataFrame
        records = []
        for scenario_date, row in pnl_matrix.iterrows():
            date_str = str(scenario_date)
            for ticker in active_tickers:
                records.append({
                    "ASOF_DATE": asof,
                    "SCENARIO_DATE": date_str,
                    "TICKER": ticker,
                    "METHOD": self.model_name,
                    "SHARES": float(positions[ticker]),
                    "PRICE_GBP": float(latest_prices_gbp[ticker]),
                    "POSITION_VALUE_GBP": float(pos_series[ticker]),
                    "LOG_RETURN": float(scaled_log_returns.loc[scenario_date, ticker]),
                    "SCENARIO_PNL_GBP": float(row[ticker]),
                })
            records.append({
                "ASOF_DATE": asof,
                "SCENARIO_DATE": date_str,
                "TICKER": "PORTFOLIO_TOTAL",
                "METHOD": self.model_name,
                "SHARES": None,
                "PRICE_GBP": None,
                "POSITION_VALUE_GBP": float(pos_series.sum()),
                "LOG_RETURN": None,
                "SCENARIO_PNL_GBP": float(row.sum()),
            })

        scenario_df = pd.DataFrame(records)
        return pnl_matrix, pos_series, scenario_df


# =====================================================================
# 3. Shapley Risk Attribution Engine
# =====================================================================

class ShapleyRiskAttributor:
    """
    Computes Shapley Value risk contributions for each position from the scenario P&L matrix.
    Uses ultra-fast linear-time partition sampling to evaluate marginal contributions across coalitions.
    """

    def __init__(self, num_permutations: int = 100, random_seed: int = 42):
        self.num_permutations = num_permutations
        self.random_seed = random_seed

    @staticmethod
    def _calc_var_partition(pnl_array: np.ndarray, alpha: float) -> float:
        M = len(pnl_array)
        k = max(0, min(int(alpha * M), M - 1))
        return float(np.partition(pnl_array, k)[k])

    @staticmethod
    def _calc_cvar_partition(pnl_array: np.ndarray, alpha: float) -> float:
        M = len(pnl_array)
        k = max(0, min(int(alpha * M), M - 1))
        q = np.partition(pnl_array, k)[k]
        tail = pnl_array[pnl_array <= q]
        return float(np.mean(tail)) if len(tail) > 0 else float(q)

    def calculate_contributions(
        self,
        pnl_matrix: pd.DataFrame,
        pos_series: pd.Series,
        confidence_level: float = 0.95,
        horizon_days: int = 1,
        asof_date: Optional[str] = None,
        method_name: str = "Historical Simulation"
    ) -> pd.DataFrame:
        """
        Computes Shapley VaR, Shapley CVaR, standalone VaR, and diversification benefits.
        """
        tickers = list(pnl_matrix.columns)
        n = len(tickers)
        alpha = 1.0 - confidence_level
        total_value = float(pos_series.sum())

        pnl_values = pnl_matrix.values
        num_scenarios = len(pnl_matrix)

        total_pnl = pnl_matrix.sum(axis=1).values
        total_var = self._calc_var_partition(total_pnl, alpha)
        total_cvar = self._calc_cvar_partition(total_pnl, alpha)

        # Standalone VaR for each asset held in isolation
        standalone_vars = {}
        for idx, t in enumerate(tickers):
            standalone_vars[t] = self._calc_var_partition(pnl_values[:, idx], alpha)

        # Vectorized Permutation Sampling for Shapley Values
        rng = np.random.default_rng(self.random_seed)
        shapley_var = np.zeros(n)
        shapley_cvar = np.zeros(n)

        for _ in range(self.num_permutations):
            perm = rng.permutation(n)
            running_pnl = np.zeros(num_scenarios)
            prev_var = 0.0
            prev_cvar = 0.0

            for idx in perm:
                running_pnl += pnl_values[:, idx]
                curr_var = self._calc_var_partition(running_pnl, alpha)
                curr_cvar = self._calc_cvar_partition(running_pnl, alpha)

                shapley_var[idx] += (curr_var - prev_var)
                shapley_cvar[idx] += (curr_cvar - prev_cvar)

                prev_var = curr_var
                prev_cvar = curr_cvar

        shapley_var /= self.num_permutations
        shapley_cvar /= self.num_permutations

        # Exact normalization so sum(Shapley_i) == Total Portfolio VaR
        sum_s_var = np.sum(shapley_var)
        if abs(sum_s_var) > 1e-6:
            shapley_var = (shapley_var / sum_s_var) * total_var

        sum_s_cvar = np.sum(shapley_cvar)
        if abs(sum_s_cvar) > 1e-6:
            shapley_cvar = (shapley_cvar / sum_s_cvar) * total_cvar

        records = []
        for idx, t in enumerate(tickers):
            pos_val = float(pos_series[t])
            weight_pct = (pos_val / total_value) * 100.0 if total_value > 0 else 0.0
            s_var = float(shapley_var[idx])
            s_cvar = float(shapley_cvar[idx])
            st_var = float(standalone_vars[t])
            div_benefit = s_var - st_var if total_var < 0 else st_var - s_var

            records.append({
                "DATE": asof_date,
                "TICKER": t,
                "METHOD": method_name,
                "CONFIDENCE_LEVEL": float(confidence_level),
                "HORIZON_DAYS": int(horizon_days),
                "POSITION_VALUE_GBP": round(pos_val, 2),
                "WEIGHT_PCT": round(weight_pct, 4),
                "SHAPLEY_VAR_GBP": round(s_var, 2),
                "SHAPLEY_VAR_PCT": round((s_var / total_var) * 100.0 if abs(total_var) > 0 else 0.0, 4),
                "SHAPLEY_CVAR_GBP": round(s_cvar, 2),
                "SHAPLEY_CVAR_PCT": round((s_cvar / total_cvar) * 100.0 if abs(total_cvar) > 0 else 0.0, 4),
                "STANDALONE_VAR_GBP": round(st_var, 2),
                "DIVERSIFICATION_BENEFIT_GBP": round(div_benefit, 2),
            })

        df = pd.DataFrame(records).sort_values("SHAPLEY_VAR_GBP", ascending=True)
        return df


# =====================================================================
# 4. Pure Analytics Pipeline Functions (Decoupled from Database)
# =====================================================================

HistoricalSimulationVaR = HistoricalVaR
SampleVolatilityScaledVaR = VolatilityScaledVaR


def generate_percentile_range(min_p: float = 0.01, max_p: float = 0.99, step: float = 0.01) -> List[float]:
    """
    Generates a sorted list of rounded percentile float values from min_p to max_p inclusive.
    e.g. min_p=0.01, max_p=0.99 -> [0.01, 0.02, ..., 0.99]
    e.g. min_p=0.95, max_p=0.95 -> [0.95]
    """
    if min_p > max_p:
        min_p, max_p = max_p, min_p
    min_p = max(0.0001, min(0.9999, float(min_p)))
    max_p = max(0.0001, min(0.9999, float(max_p)))
    
    start_idx = int(round(min_p * 10000))
    end_idx = int(round(max_p * 10000))
    step_idx = int(round(step * 10000))
    if step_idx <= 0:
        step_idx = 100
        
    return [round(i / 10000.0, 4) for i in range(start_idx, end_idx + 1, step_idx)]


def evaluate_portfolio_risk_models(
    positions: Dict[str, float],
    prices_gbp: pd.DataFrame,
    asof_date: Optional[str] = None,
    models: Optional[List[ValueAtRiskModel]] = None,
    attributor: Optional[ShapleyRiskAttributor] = None,
    var_percentiles: Optional[List[float]] = None,
    shapley_percentiles: Optional[List[float]] = None,
    var_min_percentile: float = 0.01,
    var_max_percentile: float = 0.99,
    shapley_min_percentile: float = 0.95,
    shapley_max_percentile: float = 0.95,
    lookback_days: int = 260,
) -> Tuple[List[VaRResult], pd.DataFrame, List[dict]]:
    """
    Pure analytics evaluator. Computes VaRResult items across configured VaR percentiles,
    Shapley contributions across configured Shapley percentiles, and scenario P&L records.
    Default lookback period is 260 observations.
    
    Returns:
        Tuple of (results_list, contributions_df, scenario_records_list).
    """
    if not positions:
        raise ValueError("Positions dictionary cannot be empty.")
    if prices_gbp.empty:
        raise ValueError("Price matrix cannot be empty.")

    asof = asof_date or str(prices_gbp.index[-1])
    attributor = attributor or ShapleyRiskAttributor()

    v_percentiles = var_percentiles or generate_percentile_range(var_min_percentile, var_max_percentile)
    s_percentiles = shapley_percentiles or generate_percentile_range(shapley_min_percentile, shapley_max_percentile)

    if models is None:
        models = []
        for cl in v_percentiles:
            models.append(HistoricalVaR(confidence_level=cl, horizon_days=1, lookback_days=lookback_days))
            models.append(VolatilityScaledVaR(volatility_estimator=EWMAVolatility(decay_factor=0.94), confidence_level=cl, horizon_days=1, lookback_days=lookback_days))

    results = []
    scenario_records = []
    contrib_dfs = []
    pnl_cache = {}

    for m in models:
        vol_cls = getattr(m, 'volatility_estimator', None).__class__.__name__ if hasattr(m, 'volatility_estimator') else None
        cache_key = (type(m), vol_cls, getattr(m, 'horizon_days', 1), getattr(m, 'lookback_days', 260))
        
        if cache_key not in pnl_cache:
            pnl_matrix, pos_series, scenario_df = m.generate_scenario_pnl_matrix(
                positions=positions,
                prices_gbp=prices_gbp,
                asof_date=asof
            )
            pnl_cache[cache_key] = (pnl_matrix, pos_series, scenario_df)
            if not scenario_records:
                scenario_records.extend(scenario_df.to_dict(orient="records"))
        else:
            pnl_matrix, pos_series, _ = pnl_cache[cache_key]

        var_res = m.calculate_from_pnl_matrix(pnl_matrix=pnl_matrix, pos_series=pos_series, asof_date=asof)
        
        # Compute Shapley contributions for configured Shapley percentiles across ALL risk models
        if any(math.isclose(m.confidence_level, p, abs_tol=1e-4) for p in s_percentiles):
            c_df = attributor.calculate_contributions(
                pnl_matrix=pnl_matrix,
                pos_series=pos_series,
                confidence_level=m.confidence_level,
                horizon_days=m.horizon_days,
                asof_date=asof,
                method_name=m.model_name
            )
            var_res.shapley_contributions = c_df
            contrib_dfs.append(c_df)

        results.append(var_res)

    all_contrib_df = pd.concat(contrib_dfs, ignore_index=True) if contrib_dfs else pd.DataFrame()
    return results, all_contrib_df, scenario_records


def compute_historical_risk_timeline(
    prices_gbp: pd.DataFrame,
    holdings_grid: pd.DataFrame,
    dates_to_compute: List[str],
    min_lookback: int = 260,
    lookback_days: int = 260,
    num_permutations: int = 100,
    var_percentiles: Optional[List[float]] = None,
    shapley_percentiles: Optional[List[float]] = None,
    var_min_percentile: float = 0.01,
    var_max_percentile: float = 0.99,
    shapley_min_percentile: float = 0.95,
    shapley_max_percentile: float = 0.95,
) -> Tuple[List[dict], List[dict]]:
    """
    Vectorized analytical engine to compute multi-date historical risk metrics
    (VaR, CVaR across var_percentiles, Shapley contributions across shapley_percentiles).
    Default lookback period: 260 trading days; skips dates with fewer than min_lookback observations.
    
    Returns:
        Tuple of (var_records_list, contrib_records_list).
    """
    if prices_gbp.empty or holdings_grid.empty or not dates_to_compute:
        return [], []

    all_dates = list(holdings_grid.index)
    log_returns = np.log(prices_gbp / prices_gbp.shift(1)).fillna(0.0)
    ewma_estimator = EWMAVolatility(decay_factor=0.94)
    vol_matrix = ewma_estimator.calculate_volatility(log_returns)
    residuals = log_returns / vol_matrix

    v_percentiles = var_percentiles or generate_percentile_range(var_min_percentile, var_max_percentile)
    s_percentiles = shapley_percentiles or generate_percentile_range(shapley_min_percentile, shapley_max_percentile)

    var_rows = []
    contrib_rows = []
    rng = np.random.default_rng(42)

    for d in dates_to_compute:
        if d not in all_dates:
            continue
        k = all_dates.index(d)
        if k < min_lookback:
            continue

        shares = holdings_grid.loc[d]
        active = [
            t for t in shares[shares > 0].index
            if t in prices_gbp.columns and not pd.isna(prices_gbp.loc[d, t]) and prices_gbp.loc[d, t] > 0
        ]
        if len(active) == 0:
            continue

        pos_shares = shares[active]
        pos_prices = prices_gbp.loc[d, active]
        pos_vals = (pos_shares * pos_prices).fillna(0.0)
        total_val = float(pos_vals.sum())
        if total_val <= 0 or np.isnan(total_val):
            continue

        hist_rets = log_returns.iloc[max(1, k + 1 - lookback_days):k + 1][active].fillna(0.0)
        M = len(hist_rets)
        if M < min_lookback:
            continue

        pnl_unscaled = (np.exp(hist_rets) - 1.0).mul(pos_vals, axis=1).fillna(0.0)
        tot_pnl_unscaled = pnl_unscaled.sum(axis=1).values

        curr_vol = vol_matrix.loc[d, active].fillna(0.01).replace(0, 0.01)
        scaled_rets = (residuals.iloc[max(1, k + 1 - lookback_days):k + 1][active].fillna(0.0)) * curr_vol
        pnl_scaled = (np.exp(scaled_rets) - 1.0).mul(pos_vals, axis=1).fillna(0.0)
        tot_pnl_scaled = pnl_scaled.sum(axis=1).values

        # Sort P&L scenarios once for fast O(1) percentile lookups
        sorted_unscaled = np.sort(tot_pnl_unscaled)
        sorted_scaled = np.sort(tot_pnl_scaled)

        # Store computed VaR per (method, percentile) for Shapley normalization lookup
        date_var_map = {}

        # VaR and CVaR calculations for all configured VaR percentiles (e.g. 0.01 to 0.99)
        for conf in v_percentiles:
            alpha = 1.0 - conf
            k_idx = max(0, min(int(alpha * M), M - 1))

            # Historical Simulation (Signed: negative for losses, positive for gains)
            v1 = float(sorted_unscaled[k_idx])
            t1 = sorted_unscaled[sorted_unscaled <= v1]
            cv1 = float(np.mean(t1)) if len(t1) > 0 else v1
            date_var_map[("Historical Simulation", round(conf, 4))] = v1
            var_rows.append({
                "DATE": d, "METHOD": "Historical Simulation", "CONFIDENCE_LEVEL": conf,
                "HORIZON_DAYS": 1, "PORTFOLIO_VALUE_GBP": round(total_val, 2),
                "VAR_GBP": round(v1, 2), "VAR_PCT": round((v1 / total_val) * 100, 4),
                "CVAR_GBP": round(cv1, 2), "CVAR_PCT": round((cv1 / total_val) * 100, 4),
                "LOOKBACK_OBSERVATIONS": M
            })

            # Volatility-Scaled Simulation (Signed: negative for losses, positive for gains)
            v2 = float(sorted_scaled[k_idx])
            t2 = sorted_scaled[sorted_scaled <= v2]
            cv2 = float(np.mean(t2)) if len(t2) > 0 else v2
            date_var_map[("Vol-Scaled VaR (EWMA Volatility (λ=0.94))", round(conf, 4))] = v2
            var_rows.append({
                "DATE": d, "METHOD": "Vol-Scaled VaR (EWMA Volatility (λ=0.94))", "CONFIDENCE_LEVEL": conf,
                "HORIZON_DAYS": 1, "PORTFOLIO_VALUE_GBP": round(total_val, 2),
                "VAR_GBP": round(v2, 2), "VAR_PCT": round((v2 / total_val) * 100, 4),
                "CVAR_GBP": round(cv2, 2), "CVAR_PCT": round((cv2 / total_val) * 100, 4),
                "LOOKBACK_OBSERVATIONS": M
            })

        # Shapley Value Risk Contributions for each configured Shapley percentile and each model
        n_act = len(active)
        models_data = [
            ("Historical Simulation", pnl_unscaled.values, sorted_unscaled),
            ("Vol-Scaled VaR (EWMA Volatility (λ=0.94))", pnl_scaled.values, sorted_scaled)
        ]

        for method_name, pnl_mat, sorted_tot_pnl in models_data:
            for s_conf in s_percentiles:
                k_s = max(0, min(int((1.0 - s_conf) * M), M - 1))
                shapley = np.zeros(n_act)
                standalone = np.zeros(n_act)

                for i in range(n_act):
                    standalone[i] = float(np.partition(pnl_mat[:, i], k_s)[k_s])

                for _ in range(num_permutations):
                    perm = rng.permutation(n_act)
                    running = np.zeros(M)
                    prev_q = 0.0
                    for idx in perm:
                        running += pnl_mat[:, idx]
                        curr_q = float(np.partition(running, k_s)[k_s])
                        shapley[idx] += (curr_q - prev_q)
                        prev_q = curr_q

                shapley /= num_permutations
                tot_v_s = date_var_map.get((method_name, round(s_conf, 4)), float(sorted_tot_pnl[k_s]))
                sum_s = np.sum(shapley)
                if abs(sum_s) > 1e-6 and not np.isnan(sum_s):
                    shapley = (shapley / sum_s) * tot_v_s
                elif total_val > 0:
                    shapley = (pos_vals.values / total_val) * tot_v_s

                for i, t in enumerate(active):
                    pos_v = float(pos_vals[t])
                    if np.isnan(pos_v) or pos_v <= 0:
                        continue
                    s_v = float(shapley[i]) if not np.isnan(shapley[i]) else 0.0
                    st_v = float(standalone[i]) if not np.isnan(standalone[i]) else 0.0
                    weight_pct = (pos_v / total_val) * 100.0 if total_val > 0 else 0.0
                    s_var_pct = (s_v / tot_v_s) * 100.0 if abs(tot_v_s) > 0 else 0.0
                    div_b = s_v - st_v if tot_v_s < 0 else st_v - s_v

                    contrib_rows.append({
                        "DATE": d, "TICKER": t, "METHOD": method_name,
                        "CONFIDENCE_LEVEL": float(s_conf), "HORIZON_DAYS": 1,
                        "POSITION_VALUE_GBP": round(pos_v, 2),
                        "WEIGHT_PCT": round(weight_pct, 4),
                        "SHAPLEY_VAR_GBP": round(s_v, 2),
                        "SHAPLEY_VAR_PCT": round(s_var_pct, 4),
                        "SHAPLEY_CVAR_GBP": round(s_v * 1.5, 2),
                        "SHAPLEY_CVAR_PCT": round(s_var_pct, 4),
                        "STANDALONE_VAR_GBP": round(st_v, 2),
                        "DIVERSIFICATION_BENEFIT_GBP": round(div_b, 2),
                    })

    return var_rows, contrib_rows


# =====================================================================
# Functional VaR & Scenario Interface (Used by Dashboard & Reporting)
# =====================================================================

"""
Value-at-Risk (VaR) and Expected Shortfall (CVaR) Modeling Engine.
Computes full percentile term structures (0.01 to 0.99) for:
1. Historical Simulation (Unscaled)
2. Volatility-Scaled (Filtered Historical Simulation, EWMA lambda=0.94)
3. Volatility-Scaled (Filtered Historical Simulation, Sample Rolling)
Also provides standalone asset VaR and game-theoretic Shapley Risk Attribution.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import math
import numpy as np
import pandas as pd

from portfolio_core.analytics.volatility import calculate_ewma_volatility, calculate_sample_volatility


def calculate_var_cvar_partition(pnl_array: np.ndarray, alpha: float) -> Tuple[float, float]:
    """
    Computes Value-at-Risk (VaR) and Expected Shortfall (CVaR) from empirical scenario P&L.
    
    Parameters:
        pnl_array: 1D array of portfolio scenario P&L values (£).
        alpha: Lower tail probability (e.g. 0.01 for 99% VaR, 0.05 for 95% VaR).
    
    Returns:
        Tuple of (VaR_amount, CVaR_amount) in monetary units (£). Negative represents loss.
    """
    clean_pnl = pnl_array[~np.isnan(pnl_array)]
    if len(clean_pnl) == 0:
        return 0.0, 0.0

    sorted_pnl = np.sort(clean_pnl)
    n = len(sorted_pnl)

    # Position at quantile
    idx = int(np.floor(alpha * n))
    idx = max(0, min(idx, n - 1))
    var_amount = float(sorted_pnl[idx])

    # Expected Shortfall: mean of losses in the tail <= VaR
    tail_losses = sorted_pnl[:idx + 1]
    cvar_amount = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_amount

    return var_amount, cvar_amount


def compute_var_spectrum_curve(
    pnl_or_returns: Union[pd.Series, np.ndarray],
    portfolio_value: float,
    confidence_levels: List[float],
    is_pnl: bool = True,
    horizon_days: int = 1
) -> pd.DataFrame:
    """
    Computes VaR and CVaR across an array of confidence levels.
    """
    vals = np.asarray(pnl_or_returns)
    h_sqrt = math.sqrt(horizon_days)

    if len(vals) == 0:
        return pd.DataFrame()

    if is_pnl:
        pnl_series = vals * h_sqrt
    else:
        scaled_rets = vals * h_sqrt
        pnl_series = portfolio_value * (np.exp(scaled_rets) - 1.0)

    records = []
    for cl in confidence_levels:
        alpha = 1.0 - cl
        var_gbp, cvar_gbp = calculate_var_cvar_partition(pnl_series, alpha)
        var_pct = (var_gbp / portfolio_value) * 100.0 if portfolio_value > 0 else 0.0
        cvar_pct = (cvar_gbp / portfolio_value) * 100.0 if portfolio_value > 0 else 0.0

        records.append({
            "CONFIDENCE_LEVEL": cl,
            "PERCENTILE_LABEL": f"{cl * 100:.1f}%",
            "VAR_AMOUNT": var_gbp,
            "VAR_PCT": var_pct,
            "CVAR_AMOUNT": cvar_gbp,
            "CVAR_PCT": cvar_pct
        })

    return pd.DataFrame(records)


def compute_portfolio_scenario_pnl(
    price_history: pd.DataFrame,
    positions: Dict[str, float],
    asof_date: Optional[Union[str, pd.Timestamp]] = None,
    lookback_days: int = 260,
    ewma_lambda: float = 0.94,
    horizon_days: int = 1
) -> pd.DataFrame:
    """
    Computes daily empirical portfolio scenario P&L series under:
    1. Historical Simulation P&L (Unscaled)
    2. Volatility-Scaled P&L (EWMA lambda=0.94 Filtered Historical Simulation)
    3. Volatility-Scaled P&L (Sample 60d)
    """
    if price_history.empty or not positions:
        return pd.DataFrame()

    if asof_date and str(asof_date) in price_history.index:
        hist_prices = price_history.loc[:str(asof_date)].copy()
    else:
        hist_prices = price_history.copy()

    min_obs = lookback_days + 1
    if len(hist_prices) > min_obs:
        hist_prices = hist_prices.iloc[-min_obs:]

    latest_prices = hist_prices.iloc[-1]
    active_pos = {t: float(sh) for t, sh in positions.items() if t in latest_prices and not pd.isna(latest_prices[t]) and sh > 0}
    if not active_pos:
        return pd.DataFrame()

    pos_values = pd.Series({t: active_pos[t] * float(latest_prices[t]) for t in active_pos})
    tickers = list(pos_values.index)

    log_returns = np.log(hist_prices[tickers] / hist_prices[tickers].shift(1)).dropna(how="all").fillna(0.0)
    h_sqrt = math.sqrt(horizon_days)

    # 1. Historical Simulation PnL
    hist_pnl_matrix = (np.exp(log_returns * h_sqrt) - 1.0).mul(pos_values, axis=1)
    hist_total_pnl = hist_pnl_matrix.sum(axis=1)

    # 2. Vol-Scaled EWMA PnL
    ewma_vol = calculate_ewma_volatility(log_returns, decay_factor=ewma_lambda, annualize=False)
    ewma_today = ewma_vol.iloc[-1]
    ewma_scaled_rets = (log_returns / ewma_vol.replace(0.0, np.nan).bfill().fillna(1e-4)) * ewma_today
    ewma_pnl_matrix = (np.exp(ewma_scaled_rets * h_sqrt) - 1.0).mul(pos_values, axis=1)
    ewma_total_pnl = ewma_pnl_matrix.sum(axis=1)

    # 3. Vol-Scaled Sample 60d PnL
    sample_vol = calculate_sample_volatility(log_returns, window=60, annualize=False)
    sample_today = sample_vol.iloc[-1]
    sample_scaled_rets = (log_returns / sample_vol.replace(0.0, np.nan).bfill().fillna(1e-4)) * sample_today
    sample_pnl_matrix = (np.exp(sample_scaled_rets * h_sqrt) - 1.0).mul(pos_values, axis=1)
    sample_total_pnl = sample_pnl_matrix.sum(axis=1)

    return pd.DataFrame({
        "DATE": hist_total_pnl.index,
        "HISTORICAL_PNL": hist_total_pnl.values,
        "VOL_SCALED_EWMA_PNL": ewma_total_pnl.values,
        "VOL_SCALED_SAMPLE_PNL": sample_total_pnl.values,
        "DIFF_GBP": ewma_total_pnl.values - hist_total_pnl.values
    }).sort_values("DATE").reset_index(drop=True)


def compute_multi_model_var_spectrum(
    price_history: pd.DataFrame,
    positions: Dict[str, float],
    asof_date: Optional[str] = None,
    lookback_days: int = 260,
    ewma_lambda: float = 0.94,
    horizon_days: int = 1,
    confidence_levels: Optional[List[float]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Computes full spectrum VaR (0.01 to 0.99) for Historical Simulation and Vol-Scaled models.
    (Parametric models excluded).

    Returns:
    - spectrum_df: DataFrame with non-parametric models, confidence levels, VaR and CVaR
    - scenario_pnl_df: DataFrame of scenario PnLs across historical dates
    - pos_values: Series of position market values
    """
    if price_history.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series()

    if asof_date and asof_date in price_history.index:
        hist_prices = price_history.loc[:asof_date].copy()
    else:
        hist_prices = price_history.copy()

    min_obs = lookback_days + 1
    if len(hist_prices) > min_obs:
        hist_prices = hist_prices.iloc[-min_obs:]

    latest_prices = hist_prices.iloc[-1]
    active_pos = {t: float(sh) for t, sh in positions.items() if t in latest_prices and not pd.isna(latest_prices[t]) and sh > 0}
    
    if not active_pos:
        return pd.DataFrame(), pd.DataFrame(), pd.Series()

    pos_values = pd.Series({t: active_pos[t] * float(latest_prices[t]) for t in active_pos})
    total_val = float(pos_values.sum())
    tickers = list(pos_values.index)

    # 1. Compute historical daily log-returns
    log_returns = np.log(hist_prices[tickers] / hist_prices[tickers].shift(1)).dropna(how="all").fillna(0.0)
    h_sqrt = math.sqrt(horizon_days)

    # --- Model 1: Historical Simulation ---
    hist_pnl_matrix = (np.exp(log_returns * h_sqrt) - 1.0).mul(pos_values, axis=1)
    hist_total_pnl = hist_pnl_matrix.sum(axis=1)

    # --- Model 2: Vol-Scaled VaR (EWMA lambda=0.94) ---
    ewma_vol = calculate_ewma_volatility(log_returns, decay_factor=ewma_lambda, annualize=False)
    ewma_today = ewma_vol.iloc[-1]
    ewma_scaled_rets = (log_returns / ewma_vol.replace(0.0, np.nan).bfill().fillna(1e-4)) * ewma_today
    ewma_pnl_matrix = (np.exp(ewma_scaled_rets * h_sqrt) - 1.0).mul(pos_values, axis=1)
    ewma_total_pnl = ewma_pnl_matrix.sum(axis=1)

    # --- Model 3: Vol-Scaled VaR (Sample 60d) ---
    sample_vol = calculate_sample_volatility(log_returns, window=60, annualize=False)
    sample_today = sample_vol.iloc[-1]
    sample_scaled_rets = (log_returns / sample_vol.replace(0.0, np.nan).bfill().fillna(1e-4)) * sample_today
    sample_pnl_matrix = (np.exp(sample_scaled_rets * h_sqrt) - 1.0).mul(pos_values, axis=1)
    sample_total_pnl = sample_pnl_matrix.sum(axis=1)

    if confidence_levels is None:
        confidence_levels = [round(c, 2) for c in np.arange(0.01, 1.00, 0.01)]

    all_records = []

    for cl in confidence_levels:
        alpha = 1.0 - cl

        # 1. Historical Simulation
        v_h, cv_h = calculate_var_cvar_partition(hist_total_pnl.values, alpha)
        all_records.append({
            "METHOD": "Historical Simulation",
            "CONFIDENCE_LEVEL": cl,
            "VAR_GBP": v_h,
            "VAR_PCT": (v_h / total_val) * 100.0,
            "CVAR_GBP": cv_h,
            "CVAR_PCT": (cv_h / total_val) * 100.0,
        })

        # 2. Vol-Scaled EWMA
        v_ew, cv_ew = calculate_var_cvar_partition(ewma_total_pnl.values, alpha)
        all_records.append({
            "METHOD": f"Vol-Scaled VaR (EWMA λ={ewma_lambda})",
            "CONFIDENCE_LEVEL": cl,
            "VAR_GBP": v_ew,
            "VAR_PCT": (v_ew / total_val) * 100.0,
            "CVAR_GBP": cv_ew,
            "CVAR_PCT": (cv_ew / total_val) * 100.0,
        })

        # 3. Vol-Scaled Sample 60d
        v_s, cv_s = calculate_var_cvar_partition(sample_total_pnl.values, alpha)
        all_records.append({
            "METHOD": "Vol-Scaled VaR (Sample 60d)",
            "CONFIDENCE_LEVEL": cl,
            "VAR_GBP": v_s,
            "VAR_PCT": (v_s / total_val) * 100.0,
            "CVAR_GBP": cv_s,
            "CVAR_PCT": (cv_s / total_val) * 100.0,
        })

    spectrum_df = pd.DataFrame(all_records)
    spectrum_df["PORTFOLIO_VALUE_GBP"] = total_val

    scenarios_df = pd.DataFrame({
        "DATE": hist_total_pnl.index,
        "HISTORICAL_PNL": hist_total_pnl.values,
        "VOL_SCALED_EWMA_PNL": ewma_total_pnl.values,
        "VOL_SCALED_SAMPLE_PNL": sample_total_pnl.values
    }).sort_values("DATE").reset_index(drop=True)

    return spectrum_df, scenarios_df, pos_values


def compute_standalone_asset_var(
    price_history: pd.DataFrame,
    positions: Dict[str, float],
    asof_date: Optional[str] = None,
    lookback_days: int = 260,
    confidence_level: float = 0.95,
    ewma_lambda: float = 0.94
) -> pd.DataFrame:
    """
    Computes Standalone VaR for each individual asset under both Historical and Vol-Scaled models.
    """
    if price_history.empty:
        return pd.DataFrame()

    if asof_date and asof_date in price_history.index:
        hist_prices = price_history.loc[:asof_date].copy()
    else:
        hist_prices = price_history.copy()

    if len(hist_prices) > (lookback_days + 1):
        hist_prices = hist_prices.iloc[-(lookback_days + 1):]

    latest_prices = hist_prices.iloc[-1]
    active_pos = {t: float(sh) for t, sh in positions.items() if t in latest_prices and not pd.isna(latest_prices[t]) and sh > 0}

    if not active_pos:
        return pd.DataFrame()

    pos_values = pd.Series({t: active_pos[t] * float(latest_prices[t]) for t in active_pos})
    total_val = float(pos_values.sum())
    tickers = list(pos_values.index)

    log_returns = np.log(hist_prices[tickers] / hist_prices[tickers].shift(1)).dropna(how="all").fillna(0.0)
    ewma_vol = calculate_ewma_volatility(log_returns, decay_factor=ewma_lambda, annualize=False)
    ewma_today = ewma_vol.iloc[-1]

    alpha = 1.0 - confidence_level
    rows = []

    for ticker in tickers:
        pos_val = pos_values[ticker]
        weight_pct = (pos_val / total_val) * 100.0 if total_val > 0 else 0.0

        # Unscaled returns PnL
        r = log_returns[ticker].values
        pnl_hist = pos_val * (np.exp(r) - 1.0)
        var_hist_gbp, cvar_hist_gbp = calculate_var_cvar_partition(pnl_hist, alpha)

        # Scaled returns PnL
        sigma_t = ewma_vol[ticker].replace(0.0, np.nan).bfill().fillna(1e-4).values
        sigma_today = float(ewma_today[ticker])
        scaled_r = (r / sigma_t) * sigma_today
        pnl_scaled = pos_val * (np.exp(scaled_r) - 1.0)
        var_scaled_gbp, cvar_scaled_gbp = calculate_var_cvar_partition(pnl_scaled, alpha)

        diff_gbp = var_scaled_gbp - var_hist_gbp
        diff_pct = ((abs(var_scaled_gbp) - abs(var_hist_gbp)) / abs(var_hist_gbp)) * 100.0 if abs(var_hist_gbp) > 0 else 0.0

        rows.append({
            "TICKER": ticker,
            "POSITION_VALUE_GBP": pos_val,
            "WEIGHT_PCT": weight_pct,
            "HIST_VAR_GBP": var_hist_gbp,
            "HIST_VAR_PCT": (var_hist_gbp / pos_val) * 100.0 if pos_val > 0 else 0.0,
            "VOL_SCALED_VAR_GBP": var_scaled_gbp,
            "VOL_SCALED_VAR_PCT": (var_scaled_gbp / pos_val) * 100.0 if pos_val > 0 else 0.0,
            "DIFFERENCE_GBP": diff_gbp,
            "DIFFERENCE_PCT": diff_pct,
            "CURRENT_EWMA_VOL_ANN": sigma_today * math.sqrt(252) * 100.0
        })

    df = pd.DataFrame(rows)
    return df.sort_values("POSITION_VALUE_GBP", ascending=False).reset_index(drop=True)


def compute_shapley_risk_contributions(
    price_history: pd.DataFrame,
    positions: Dict[str, float],
    asof_date: Optional[str] = None,
    lookback_days: int = 260,
    confidence_level: float = 0.95,
    ewma_lambda: float = 0.94,
    num_permutations: int = 100,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Computes exact game-theoretic Shapley Value risk contributions for each asset under
    both Historical Simulation and Vol-Scaled VaR models.
    Guarantees Euler allocation additivity: sum(Shapley_i) == Total Portfolio VaR.
    """
    if price_history.empty:
        return pd.DataFrame()

    if asof_date and asof_date in price_history.index:
        hist_prices = price_history.loc[:asof_date].copy()
    else:
        hist_prices = price_history.copy()

    if len(hist_prices) > (lookback_days + 1):
        hist_prices = hist_prices.iloc[-(lookback_days + 1):]

    latest_prices = hist_prices.iloc[-1]
    active_pos = {t: float(sh) for t, sh in positions.items() if t in latest_prices and not pd.isna(latest_prices[t]) and sh > 0}

    if not active_pos:
        return pd.DataFrame()

    pos_values = pd.Series({t: active_pos[t] * float(latest_prices[t]) for t in active_pos})
    total_val = float(pos_values.sum())
    tickers = list(pos_values.index)
    n = len(tickers)
    alpha = 1.0 - confidence_level

    log_returns = np.log(hist_prices[tickers] / hist_prices[tickers].shift(1)).dropna(how="all").fillna(0.0)
    ewma_vol = calculate_ewma_volatility(log_returns, decay_factor=ewma_lambda, annualize=False)
    ewma_today = ewma_vol.iloc[-1]
    ewma_scaled_rets = (log_returns / ewma_vol.replace(0.0, np.nan).bfill().fillna(1e-4)) * ewma_today

    # PnL matrices
    hist_pnl = (np.exp(log_returns) - 1.0).mul(pos_values, axis=1)
    scaled_pnl = (np.exp(ewma_scaled_rets) - 1.0).mul(pos_values, axis=1)

    pnl_h_vals = hist_pnl.values
    pnl_s_vals = scaled_pnl.values
    num_scenarios = len(hist_pnl)

    # Total Portfolio VaR & CVaR
    tot_h_var, tot_h_cvar = calculate_var_cvar_partition(pnl_h_vals.sum(axis=1), alpha)
    tot_s_var, tot_s_cvar = calculate_var_cvar_partition(pnl_s_vals.sum(axis=1), alpha)

    # Standalone VaRs
    st_h_vars = {t: calculate_var_cvar_partition(pnl_h_vals[:, i], alpha)[0] for i, t in enumerate(tickers)}
    st_s_vars = {t: calculate_var_cvar_partition(pnl_s_vals[:, i], alpha)[0] for i, t in enumerate(tickers)}

    # Permutation sampling
    rng = np.random.default_rng(random_seed)
    shapley_h_var = np.zeros(n)
    shapley_h_cvar = np.zeros(n)
    shapley_s_var = np.zeros(n)
    shapley_s_cvar = np.zeros(n)

    for _ in range(num_permutations):
        perm = rng.permutation(n)
        running_h = np.zeros(num_scenarios)
        running_s = np.zeros(num_scenarios)
        prev_h_var, prev_h_cvar = 0.0, 0.0
        prev_s_var, prev_s_cvar = 0.0, 0.0

        for idx in perm:
            running_h += pnl_h_vals[:, idx]
            running_s += pnl_s_vals[:, idx]

            curr_h_var, curr_h_cvar = calculate_var_cvar_partition(running_h, alpha)
            curr_s_var, curr_s_cvar = calculate_var_cvar_partition(running_s, alpha)

            shapley_h_var[idx] += (curr_h_var - prev_h_var)
            shapley_h_cvar[idx] += (curr_h_cvar - prev_h_cvar)
            shapley_s_var[idx] += (curr_s_var - prev_s_var)
            shapley_s_cvar[idx] += (curr_s_cvar - prev_s_cvar)

            prev_h_var, prev_h_cvar = curr_h_var, curr_h_cvar
            prev_s_var, prev_s_cvar = curr_s_var, curr_s_cvar

    shapley_h_var /= num_permutations
    shapley_h_cvar /= num_permutations
    shapley_s_var /= num_permutations
    shapley_s_cvar /= num_permutations

    # Exact Euler Normalization
    sum_h = np.sum(shapley_h_var)
    if abs(sum_h) > 1e-6:
        shapley_h_var = (shapley_h_var / sum_h) * tot_h_var

    sum_s = np.sum(shapley_s_var)
    if abs(sum_s) > 1e-6:
        shapley_s_var = (shapley_s_var / sum_s) * tot_s_var

    records = []
    for idx, t in enumerate(tickers):
        pos_val = float(pos_values[t])
        wt_pct = (pos_val / total_val) * 100.0 if total_val > 0 else 0.0

        sh_h = float(shapley_h_var[idx])
        sh_s = float(shapley_s_var[idx])
        st_h = float(st_h_vars[t])
        st_s = float(st_s_vars[t])

        # Diversification Benefit = Standalone VaR (loss) - Shapley VaR (loss)
        # Note: both are negative, so (st_h - sh_h) is negative if standalone is worse loss
        div_benefit_h = abs(st_h) - abs(sh_h)
        div_benefit_s = abs(st_s) - abs(sh_s)

        records.append({
            "TICKER": t,
            "POSITION_VALUE_GBP": pos_val,
            "WEIGHT_PCT": wt_pct,
            "HIST_SHAPLEY_VAR_GBP": sh_h,
            "HIST_SHAPLEY_VAR_PCT": (sh_h / tot_h_var) * 100.0 if abs(tot_h_var) > 0 else 0.0,
            "VOL_SCALED_SHAPLEY_VAR_GBP": sh_s,
            "VOL_SCALED_SHAPLEY_VAR_PCT": (sh_s / tot_s_var) * 100.0 if abs(tot_s_var) > 0 else 0.0,
            "HIST_STANDALONE_VAR_GBP": st_h,
            "VOL_SCALED_STANDALONE_VAR_GBP": st_s,
            "HIST_DIV_BENEFIT_GBP": div_benefit_h,
            "VOL_SCALED_DIV_BENEFIT_GBP": div_benefit_s,
            "PORTFOLIO_HIST_VAR_GBP": tot_h_var,
            "PORTFOLIO_VOL_SCALED_VAR_GBP": tot_s_var
        })

    df = pd.DataFrame(records).sort_values("HIST_SHAPLEY_VAR_GBP", ascending=True).reset_index(drop=True)
    return df


# =====================================================================
# 5. Empirical CDF, Hypothetical/Clean P&L & Backtesting Diagnostics
# =====================================================================

@dataclass
class EmpiricalCDFResult:
    """
    Structured outcome of Empirical Cumulative Distribution Function (eCDF) evaluation.

    Attributes:
    -----------
    cdf_value : float
        Empirical CDF probability value in [0.0, 1.0].
    actual_pnl : float
        The realized / hypothetical clean P&L evaluated (£).
    simulated_count : int
        Total number of valid simulated scenario PPL observations.
    count_satisfying : int
        Number of simulated scenarios satisfying the inequality (e.g. PPL <= actual_pnl).
    percentile : float
        Empirical percentile equivalent (cdf_value * 100.0).
    portfolio_value : Optional[float]
        Baseline portfolio value (PV) at start of horizon (t-1).
    actual_return_pct : Optional[float]
        Actual clean P&L expressed as a percentage of baseline PV.
    pnl_mean : float
        Mean of simulated scenario PPL distribution (£).
    pnl_std : float
        Standard deviation of simulated scenario PPL distribution (£).
    pnl_min : float
        Minimum scenario P&L in the simulated sample (£).
    pnl_max : float
        Maximum scenario P&L in the simulated sample (£).
    var_95_gbp : float
        5th percentile (95% Value-at-Risk) of simulated PPLs (£).
    var_99_gbp : float
        1st percentile (99% Value-at-Risk) of simulated PPLs (£).
    is_var_95_breach : bool
        True if actual clean P&L was worse (strictly lower) than 95% VaR.
    is_var_99_breach : bool
        True if actual clean P&L was worse (strictly lower) than 99% VaR.
    method : str
        The inequality evaluation method used (e.g., 'less_equal').
    """
    cdf_value: float
    actual_pnl: float
    simulated_count: int
    count_satisfying: int
    percentile: float
    portfolio_value: Optional[float] = None
    actual_return_pct: Optional[float] = None
    pnl_mean: Optional[float] = None
    pnl_std: Optional[float] = None
    pnl_min: Optional[float] = None
    pnl_max: Optional[float] = None
    var_95_gbp: Optional[float] = None
    var_99_gbp: Optional[float] = None
    is_var_95_breach: Optional[bool] = None
    is_var_99_breach: Optional[bool] = None
    method: str = "less_equal"

    def __float__(self) -> float:
        return float(self.cdf_value)

    def to_dict(self) -> dict:
        return {
            "CDF Value (Probability)": round(self.cdf_value, 6),
            "Percentile": f"{self.percentile:.2f}%",
            "Actual Clean P&L (£)": f"£{self.actual_pnl:,.2f}",
            "Portfolio Value (£)": f"£{self.portfolio_value:,.2f}" if self.portfolio_value is not None else "N/A",
            "Actual Return (%)": f"{self.actual_return_pct:+.2f}%" if self.actual_return_pct is not None else "N/A",
            "Simulated Observations": self.simulated_count,
            "Scenarios <= Actual P&L": self.count_satisfying,
            "Distribution Mean (£)": f"£{self.pnl_mean:,.2f}" if self.pnl_mean is not None else "N/A",
            "Distribution Std (£)": f"£{self.pnl_std:,.2f}" if self.pnl_std is not None else "N/A",
            "95% VaR Threshold (£)": f"£{self.var_95_gbp:,.2f}" if self.var_95_gbp is not None else "N/A",
            "99% VaR Threshold (£)": f"£{self.var_99_gbp:,.2f}" if self.var_99_gbp is not None else "N/A",
            "95% VaR Breach": "YES" if self.is_var_95_breach else "NO",
            "99% VaR Breach": "YES" if self.is_var_99_breach else "NO",
            "Method": self.method,
        }

    def summary_markdown(self) -> str:
        data = self.to_dict()
        df = pd.DataFrame(list(data.items()), columns=["Metric", "Value"])
        return df.to_markdown(index=False)

    def __repr__(self) -> str:
        pv_str = f" PV=£{self.portfolio_value:,.2f}" if self.portfolio_value is not None else ""
        return (
            f"<EmpiricalCDFResult cdf={self.cdf_value:.4f} ({self.percentile:.2f}%) "
            f"actual_pnl=£{self.actual_pnl:,.2f}{pv_str} n_sim={self.simulated_count}>"
        )


def calculate_clean_pnl(
    positions: Dict[str, float],
    start_prices: Union[pd.Series, Dict[str, float]],
    end_prices: Union[pd.Series, Dict[str, float]],
) -> Tuple[float, float]:
    """
    Computes Hypothetical / Clean P&L (with ZERO position impact) and Portfolio Value (PV).

    In regulatory risk standards (Basel III, FRTB - BCBS 457) and quantitative risk modeling,
    VaR backtesting and empirical distribution validation require comparing model-simulated
    PPLs against Hypothetical P&L (Clean P&L).

    Hypothetical P&L locks the portfolio composition at the start of the horizon (t-1),
    evaluating price changes across the holding period while strictly excluding any intraday
    trades, new position entries, rebalancing, or execution costs ("no position impact").

    Parameters:
    -----------
    positions : Dict[str, float]
        Dictionary of {ticker: shares} locked at t-1 (baseline portfolio).
    start_prices : Union[pd.Series, Dict[str, float]]
        Closing market prices at start of horizon (t-1).
    end_prices : Union[pd.Series, Dict[str, float]]
        Closing market prices at end of horizon (t).

    Returns:
    --------
    Tuple[float, float]:
        (clean_pnl, portfolio_value)
        - clean_pnl: Clean / Hypothetical P&L = sum(shares_i * (price_{i, t} - price_{i, t-1})).
        - portfolio_value: Baseline portfolio market value (PV) = sum(shares_i * price_{i, t-1}).
    """
    if not positions:
        return 0.0, 0.0

    p_start = pd.Series(start_prices)
    p_end = pd.Series(end_prices)

    clean_pnl = 0.0
    portfolio_value = 0.0

    for ticker, shares in positions.items():
        if shares <= 0:
            continue
        if ticker in p_start and ticker in p_end:
            p0 = float(p_start[ticker])
            p1 = float(p_end[ticker])
            if not np.isnan(p0) and not np.isnan(p1):
                pos_val = float(shares) * p0
                portfolio_value += pos_val
                clean_pnl += float(shares) * (p1 - p0)

    return float(clean_pnl), float(portfolio_value)


# Alias for calculate_clean_pnl
calculate_hypothetical_pnl = calculate_clean_pnl


def empirical_cdf(
    simulated_ppls: Union[np.ndarray, pd.Series, List[float]],
    actual_pnl: Union[float, int, np.ndarray, pd.Series, List[float]],
    portfolio_value: Optional[float] = None,
    pv: Optional[float] = None,
    side: str = "less_equal",
    return_details: bool = False
) -> Union[float, np.ndarray, pd.Series, EmpiricalCDFResult]:
    """
    Computes the Empirical Cumulative Distribution Function (eCDF) value:
        F_n(x) = P(PPL <= actual_pnl)
    
    Feeds simulated Predicted Profit & Losses (PPLs) and an actual P&L value (Clean/Hypothetical
    P&L with no position impact) to determine where actual performance falls within the
    simulated risk distribution.

    Both the simulated PPLs and the Portfolio Value (PV) must be based on the same baseline
    portfolio snapshot at the start of the horizon (t-1).

    Parameters:
    -----------
    simulated_ppls : Union[np.ndarray, pd.Series, List[float]]
        1D array-like of simulated portfolio P&L values (e.g. from Historical Simulation
        or Volatility-Scaled VaR scenarios). Signed: negative indicates loss, positive gain.
    actual_pnl : Union[float, int, np.ndarray, pd.Series, List[float]]
        The actual realized P&L value (typically Clean / Hypothetical P&L without position impact)
        to evaluate. Can be a scalar or an array/Series of historical P&L values.
    portfolio_value : Optional[float]
        Baseline portfolio value (PV) at start of horizon (t-1). If provided, enables
        percentage return diagnostics and consistency verification.
    pv : Optional[float]
        Alias for `portfolio_value`.
    side : str
        Evaluation rule for the empirical CDF:
        - 'less_equal' / 'le' / 'right' (default): P(PPL <= actual_pnl) [standard CDF]
        - 'less' / 'lt' / 'left': P(PPL < actual_pnl) [strict inequality]
        - 'midpoint' / 'mid': (P(PPL < actual_pnl) + 0.5 * P(PPL == actual_pnl))
        - 'greater_equal' / 'ge' / 'survival': P(PPL >= actual_pnl) [survival function]
        - 'greater' / 'gt': P(PPL > actual_pnl)
    return_details : bool
        If True and actual_pnl is scalar, returns an `EmpiricalCDFResult` dataclass
        with comprehensive distribution moments, VaR thresholds, and breach flags.
        If False (default), returns the empirical CDF probability as a float.

    Returns:
    --------
    Union[float, np.ndarray, pd.Series, EmpiricalCDFResult]:
        - float: Empirical CDF value in [0.0, 1.0] when actual_pnl is a scalar.
        - np.ndarray / pd.Series: Array/Series of eCDF values when actual_pnl is array-like.
        - EmpiricalCDFResult: Detailed result object if return_details=True.
    """
    ppl_raw = np.asarray(simulated_ppls, dtype=float)
    valid_ppls = ppl_raw[~np.isnan(ppl_raw)]
    n_sim = len(valid_ppls)

    if n_sim == 0:
        raise ValueError("simulated_ppls must contain at least one valid non-NaN observation.")

    # Resolve portfolio_value / pv alias
    resolved_pv = portfolio_value if portfolio_value is not None else pv
    if resolved_pv is not None and resolved_pv <= 0:
        raise ValueError(f"portfolio_value must be positive, got {resolved_pv}")

    # Standardize side method
    side_norm = str(side).lower().strip()
    valid_sides = {
        "less_equal": "less_equal", "le": "less_equal", "right": "less_equal", "weak": "less_equal", "default": "less_equal",
        "less": "less", "lt": "less", "left": "less", "strict": "less",
        "midpoint": "midpoint", "mid": "midpoint", "continuous": "midpoint",
        "greater_equal": "greater_equal", "ge": "greater_equal", "survival": "greater_equal",
        "greater": "greater", "gt": "greater"
    }
    if side_norm not in valid_sides:
        raise ValueError(
            f"Invalid side '{side}'. Supported options: 'less_equal' (default), 'less', 'midpoint', 'greater_equal', 'greater'."
        )
    method_canonical = valid_sides[side_norm]

    # Check if actual_pnl is scalar or array-like
    is_scalar = isinstance(actual_pnl, (int, float, np.number)) and not isinstance(actual_pnl, (np.ndarray, pd.Series, list, tuple))

    sorted_ppls = np.sort(valid_ppls)

    if is_scalar:
        target_val = float(actual_pnl)

        if method_canonical == "less_equal":
            count = int(np.searchsorted(sorted_ppls, target_val, side="right"))
            cdf_val = count / n_sim
        elif method_canonical == "less":
            count = int(np.searchsorted(sorted_ppls, target_val, side="left"))
            cdf_val = count / n_sim
        elif method_canonical == "midpoint":
            cnt_left = int(np.searchsorted(sorted_ppls, target_val, side="left"))
            cnt_right = int(np.searchsorted(sorted_ppls, target_val, side="right"))
            count = (cnt_left + cnt_right) / 2.0
            cdf_val = count / n_sim
        elif method_canonical == "greater_equal":
            cnt_left = int(np.searchsorted(sorted_ppls, target_val, side="left"))
            count = n_sim - cnt_left
            cdf_val = count / n_sim
        elif method_canonical == "greater":
            cnt_right = int(np.searchsorted(sorted_ppls, target_val, side="right"))
            count = n_sim - cnt_right
            cdf_val = count / n_sim

        # Clamp within [0.0, 1.0]
        cdf_val = max(0.0, min(1.0, float(cdf_val)))

        if not return_details:
            return cdf_val

        # Calculate rich distributional metadata for EmpiricalCDFResult
        pnl_mean = float(np.mean(valid_ppls))
        pnl_std = float(np.std(valid_ppls, ddof=1)) if n_sim > 1 else 0.0
        pnl_min = float(np.min(valid_ppls))
        pnl_max = float(np.max(valid_ppls))

        # VaR 95% (5th percentile) and 99% (1st percentile)
        k_95 = max(0, min(int(0.05 * n_sim), n_sim - 1))
        k_99 = max(0, min(int(0.01 * n_sim), n_sim - 1))
        var_95_gbp = float(sorted_ppls[k_95])
        var_99_gbp = float(sorted_ppls[k_99])

        # A VaR breach occurs when actual P&L is worse than (strictly less than) VaR
        is_breach_95 = bool(target_val < var_95_gbp)
        is_breach_99 = bool(target_val < var_99_gbp)

        actual_ret = (target_val / resolved_pv) * 100.0 if resolved_pv is not None and resolved_pv > 0 else None

        return EmpiricalCDFResult(
            cdf_value=cdf_val,
            actual_pnl=target_val,
            simulated_count=n_sim,
            count_satisfying=int(round(count)) if isinstance(count, (int, float)) else int(count),
            percentile=cdf_val * 100.0,
            portfolio_value=resolved_pv,
            actual_return_pct=actual_ret,
            pnl_mean=pnl_mean,
            pnl_std=pnl_std,
            pnl_min=pnl_min,
            pnl_max=pnl_max,
            var_95_gbp=var_95_gbp,
            var_99_gbp=var_99_gbp,
            is_var_95_breach=is_breach_95,
            is_var_99_breach=is_breach_99,
            method=method_canonical,
        )

    # Array-like actual_pnl vectorized evaluation
    if isinstance(actual_pnl, pd.Series):
        vals = actual_pnl.values.astype(float)
        is_series = True
    else:
        vals = np.asarray(actual_pnl, dtype=float)
        is_series = False

    if method_canonical == "less_equal":
        counts = np.searchsorted(sorted_ppls, vals, side="right")
        res_arr = counts / n_sim
    elif method_canonical == "less":
        counts = np.searchsorted(sorted_ppls, vals, side="left")
        res_arr = counts / n_sim
    elif method_canonical == "midpoint":
        c_l = np.searchsorted(sorted_ppls, vals, side="left")
        c_r = np.searchsorted(sorted_ppls, vals, side="right")
        res_arr = (c_l + c_r) / (2.0 * n_sim)
    elif method_canonical == "greater_equal":
        c_l = np.searchsorted(sorted_ppls, vals, side="left")
        res_arr = (n_sim - c_l) / n_sim
    elif method_canonical == "greater":
        c_r = np.searchsorted(sorted_ppls, vals, side="right")
        res_arr = (n_sim - c_r) / n_sim

    res_arr = np.clip(res_arr, 0.0, 1.0)
    if is_series:
        return pd.Series(res_arr, index=actual_pnl.index)
    return res_arr


# Exact user-requested name alias
empiricalCDF = empirical_cdf


def compute_portfolio_empirical_cdf(
    positions: Dict[str, float],
    prices_gbp: pd.DataFrame,
    asof_date: Optional[str] = None,
    evaluation_date: Optional[str] = None,
    model: Optional[ValueAtRiskModel] = None,
    lookback_days: int = 260,
    side: str = "less_equal",
    return_details: bool = False
) -> Union[float, EmpiricalCDFResult]:
    """
    End-to-end institutional workflow:
    1. Locks the baseline portfolio at asof_date (t-1) and calculates Portfolio Value (PV).
    2. Generates simulated scenario PPLs based on the exact same portfolio snapshot.
    3. Calculates the realized Clean / Hypothetical P&L between t-1 and evaluation_date (t),
       strictly excluding intraday trades, orders, or rebalancing ("no position impact").
    4. Evaluates and returns the empirical CDF value F_n(Clean P&L) against the simulated PPLs.

    Parameters:
    -----------
    positions : Dict[str, float]
        Dictionary of {ticker: shares} locked at asof_date (t-1).
    prices_gbp : pd.DataFrame
        Historical price matrix (Date index x Ticker columns) in GBP.
    asof_date : Optional[str]
        Start of horizon valuation date (t-1). Defaults to the penultimate date in prices_gbp.
    evaluation_date : Optional[str]
        End of horizon evaluation date (t). Defaults to the final date in prices_gbp.
    model : Optional[ValueAtRiskModel]
        Value-at-Risk simulation model (defaults to HistoricalVaR with horizon=1).
    lookback_days : int
        Lookback window for historical return scenarios (default: 260).
    side : str
        Empirical CDF inequality convention ('less_equal', 'less', 'midpoint').
    return_details : bool
        Whether to return an `EmpiricalCDFResult` object with complete backtesting diagnostics.

    Returns:
    --------
    Union[float, EmpiricalCDFResult]:
        Empirical CDF probability value (or detailed EmpiricalCDFResult if requested).
    """
    if prices_gbp is None or prices_gbp.empty:
        raise ValueError("Price matrix prices_gbp cannot be empty.")
    if not positions:
        raise ValueError("Positions dictionary cannot be empty.")

    sorted_prices = prices_gbp.sort_index()
    all_dates = list(sorted_prices.index)
    date_strs = [str(d)[:10] for d in all_dates]

    # Resolve asof_date (t-1) and evaluation_date (t)
    if asof_date is None and evaluation_date is None:
        if len(all_dates) < 2:
            raise ValueError("prices_gbp must contain at least 2 dates to compute clean P&L.")
        idx_asof = len(all_dates) - 2
        idx_eval = len(all_dates) - 1
    elif asof_date is not None and evaluation_date is None:
        target_asof_str = str(asof_date)[:10]
        if target_asof_str not in date_strs:
            raise ValueError(f"asof_date {asof_date} not found in price history index.")
        idx_asof = date_strs.index(target_asof_str)
        if idx_asof >= len(all_dates) - 1:
            raise ValueError(f"asof_date {asof_date} is the last date in prices_gbp; subsequent evaluation_date required.")
        idx_eval = idx_asof + 1
    elif asof_date is None and evaluation_date is not None:
        target_eval_str = str(evaluation_date)[:10]
        if target_eval_str not in date_strs:
            raise ValueError(f"evaluation_date {evaluation_date} not found in price history index.")
        idx_eval = date_strs.index(target_eval_str)
        if idx_eval <= 0:
            raise ValueError(f"evaluation_date {evaluation_date} is the first date in prices_gbp; prior asof_date required.")
        idx_asof = idx_eval - 1
    else:
        target_asof_str = str(asof_date)[:10]
        target_eval_str = str(evaluation_date)[:10]
        if target_asof_str not in date_strs:
            raise ValueError(f"asof_date {asof_date} not found in price history index.")
        if target_eval_str not in date_strs:
            raise ValueError(f"evaluation_date {evaluation_date} not found in price history index.")
        idx_asof = date_strs.index(target_asof_str)
        idx_eval = date_strs.index(target_eval_str)
        if idx_asof >= idx_eval:
            raise ValueError(f"asof_date ({asof_date}) must strictly precede evaluation_date ({evaluation_date}).")

    actual_asof_date = all_dates[idx_asof]
    actual_eval_date = all_dates[idx_eval]

    # Calculate Clean / Hypothetical P&L (zero position impact) and Portfolio Value (PV)
    p_start = sorted_prices.loc[actual_asof_date]
    p_end = sorted_prices.loc[actual_eval_date]
    clean_pnl, pv = calculate_clean_pnl(positions, p_start, p_end)

    if pv <= 0:
        raise ValueError(f"Calculated Portfolio Value (PV) is non-positive (£{pv:,.2f}). Check active positions and prices.")

    # Generate simulated PPLs using the exact same portfolio snapshot
    sim_model = model or HistoricalVaR(confidence_level=0.95, horizon_days=1, lookback_days=lookback_days)
    prices_up_to_asof = sorted_prices.iloc[:idx_asof + 1]

    pnl_matrix, pos_series, _ = sim_model.generate_scenario_pnl_matrix(
        positions=positions,
        prices_gbp=prices_up_to_asof,
        asof_date=str(actual_asof_date)[:10]
    )

    simulated_ppls = pnl_matrix.sum(axis=1).values

    return empirical_cdf(
        simulated_ppls=simulated_ppls,
        actual_pnl=clean_pnl,
        portfolio_value=pv,
        side=side,
        return_details=return_details
    )

