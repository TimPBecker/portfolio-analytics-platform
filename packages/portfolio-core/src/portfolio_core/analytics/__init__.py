"""
Financial Analytics Package.
Provides volatility estimators, Value-at-Risk modeling, Expected Shortfall (CVaR),
game-theoretic Shapley risk attribution, and empirical statistical diagnostics.
"""

from portfolio_core.analytics.volatility import (
    calculate_sample_volatility,
    calculate_ewma_volatility,
    calculate_parkinson_volatility,
    calculate_scaling_factors,
    compute_volatility_summary_metrics
)

from portfolio_core.analytics.var import (
    HistoricalVaR,
    HistoricalSimulationVaR,
    VolatilityScaledVaR,
    SampleVolatilityScaledVaR,
    ShapleyRiskAttributor,
    calculate_var_cvar_partition,
    compute_multi_model_var_spectrum,
    compute_standalone_asset_var,
    compute_shapley_risk_contributions,
    compute_portfolio_scenario_pnl,
    compute_historical_risk_timeline,
    EmpiricalCDFResult,
    calculate_clean_pnl,
    calculate_hypothetical_pnl,
    empirical_cdf,
    empiricalCDF,
    compute_portfolio_empirical_cdf
)

from portfolio_core.analytics.statistics import (
    compute_asset_returns,
    compute_distribution_metrics,
    generate_density_curves,
    compute_qq_plot_data,
    compute_top_position_movers
)

from portfolio_core.analytics.backtesting import (
    UniformityTestResult,
    BinomialOutlierResult,
    KupiecPOFResult,
    IndependenceTestResult,
    ConditionalCoverageResult,
    evaluate_ecdf_uniformity,
    test_ecdf_uniformity,
    evaluate_binomial_outliers,
    kupiec_pof_test,
    kupiec_independence_test,
    christoffersen_conditional_coverage_test,
    run_backtest_diagnostics,
    generate_portfolio_backtest_timeline
)

__all__ = [
    "calculate_sample_volatility",
    "calculate_ewma_volatility",
    "calculate_parkinson_volatility",
    "calculate_scaling_factors",
    "compute_volatility_summary_metrics",
    "HistoricalVaR",
    "HistoricalSimulationVaR",
    "VolatilityScaledVaR",
    "SampleVolatilityScaledVaR",
    "ShapleyRiskAttributor",
    "calculate_var_cvar_partition",
    "compute_multi_model_var_spectrum",
    "compute_standalone_asset_var",
    "compute_shapley_risk_contributions",
    "compute_portfolio_scenario_pnl",
    "compute_historical_risk_timeline",
    "compute_asset_returns",
    "compute_distribution_metrics",
    "generate_density_curves",
    "compute_qq_plot_data",
    "compute_top_position_movers",
    "EmpiricalCDFResult",
    "calculate_clean_pnl",
    "calculate_hypothetical_pnl",
    "empirical_cdf",
    "empiricalCDF",
    "compute_portfolio_empirical_cdf",
    "UniformityTestResult",
    "BinomialOutlierResult",
    "KupiecPOFResult",
    "IndependenceTestResult",
    "ConditionalCoverageResult",
    "evaluate_ecdf_uniformity",
    "test_ecdf_uniformity",
    "evaluate_binomial_outliers",
    "kupiec_pof_test",
    "kupiec_independence_test",
    "christoffersen_conditional_coverage_test",
    "run_backtest_diagnostics",
    "generate_portfolio_backtest_timeline"
]



