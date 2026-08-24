"""
Portfolio Core Library.
Centralized database access and financial risk analytics module.
"""

from portfolio_core.config import load_config, get_db_config, get_risk_config
import portfolio_core.db as db
import portfolio_core.analytics as analytics

__version__ = "0.1.0"
__all__ = ["load_config", "get_db_config", "get_risk_config", "db", "analytics"]
