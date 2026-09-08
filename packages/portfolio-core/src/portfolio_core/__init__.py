"""
Portfolio Core Library.
Centralized database access and financial risk analytics module.
"""

from portfolio_core.config import load_config, get_db_config, get_risk_config, is_dev_environment
__version__ = "0.1.0"
__all__ = ["load_config", "get_db_config", "get_risk_config", "is_dev_environment", "db", "analytics"]


def __getattr__(name: str):
    if name == "db":
        import portfolio_core.db as db
        return db
    if name == "analytics":
        import portfolio_core.analytics as analytics
        return analytics
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
