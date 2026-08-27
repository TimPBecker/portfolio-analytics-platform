"""
Centralized Configuration Loader.
Loads YAML configuration and environment variables (.env) safely without hardcoding secrets.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


def find_config_path(start_path: Optional[str] = None) -> Path:
    """Finds config.yaml by searching upwards from current or start path, and standard container paths."""
    current = Path(start_path).resolve() if start_path else Path.cwd().resolve()
    
    # Check standard container mount locations first if they exist
    for standard_path in [Path("/app/config.yaml"), Path("/opt/dagster/app/config.yaml")]:
        if standard_path.is_file():
            return standard_path

    # Check current directory and parent levels
    for p in [current, *current.parents[:4]]:
        cand = p / "config.yaml"
        if cand.is_file():
            return cand
        cand_app = p / "apps" / "pipeline" / "config.yaml"
        if cand_app.is_file():
            return cand_app
        cand_dash = p / "apps" / "dashboard" / "config.yaml"
        if cand_dash.is_file():
            return cand_dash

    return Path(__file__).resolve().parent.parent.parent.parent / "apps" / "pipeline" / "config.yaml"


def find_env_path(start_path: Optional[str] = None) -> Optional[Path]:
    """Finds .env file by searching upwards to the repository root and container paths."""
    for standard_env in [Path("/app/.env"), Path("/opt/dagster/app/.env")]:
        if standard_env.is_file():
            return standard_env
            
    current = Path(start_path).resolve() if start_path else Path.cwd().resolve()
    for p in [current, *current.parents[:4]]:
        cand = p / ".env"
        if cand.is_file():
            return cand
    return None


def load_config(config_path: Optional[str] = None, env_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads configuration dictionary from YAML file and loads .env variables."""
    env_file = Path(env_path) if env_path else find_env_path()
    if env_file and env_file.is_file():
        load_dotenv(dotenv_path=env_file, override=False)
    else:
        load_dotenv(override=False)

    cfg_file = Path(config_path) if config_path else find_config_path()
    if not cfg_file.is_file():
        return {}

    with open(cfg_file, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    return config_dict


def get_db_config(config_dict: Optional[Dict[str, Any]] = None, is_test: Optional[bool] = None) -> Dict[str, Any]:
    """
    Extracts database configuration dictionary.
    When in test environment (is_test=True, PORTFOLIO_ENV=test, TEST_MODE=1, or active pytest session),
    automatically redirects MariaDB database names to '<database>_dev' and SQLite paths to dev/test databases.
    """
    cfg = config_dict if config_dict is not None else load_config()
    db_cfg = cfg.get("resources", {}).get("db", {}).get("config", {})
    if not db_cfg:
        db_cfg = cfg.get("database", {})
    if not db_cfg:
        db_cfg = cfg.get("db", {})
    
    # Return a copy to avoid mutating source dictionary
    resolved_cfg = dict(db_cfg)

    # Determine if in test mode
    in_test = is_test
    if in_test is None:
        env_mode = os.getenv("PORTFOLIO_ENV", "").lower()
        in_test = env_mode in ("test", "testing") or os.getenv("TEST_MODE") == "1" or "PYTEST_CURRENT_TEST" in os.environ

    if in_test and resolved_cfg:
        db_type = str(resolved_cfg.get("type", "mariadb")).lower()
        if db_type in ("mariadb", "mysql"):
            base_db = resolved_cfg.get("database") or os.getenv("DB_NAME")
            if base_db and not base_db.endswith("_dev"):
                resolved_cfg["database"] = f"{base_db}_dev"
        elif db_type == "sqlite":
            sp = resolved_cfg.get("sqlite_path")
            if sp and not sp.endswith("_dev.s3db") and sp != ":memory:":
                base = sp[:-5] if sp.endswith(".s3db") else sp
                resolved_cfg["sqlite_path"] = f"{base}_dev.s3db"

    return resolved_cfg


def get_risk_config(config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extracts risk analytics configuration dictionary."""
    cfg = config_dict if config_dict is not None else load_config()
    risk_cfg = cfg.get("resources", {}).get("risk", {}).get("config", {})
    if not risk_cfg:
        risk_cfg = cfg.get("risk", {})
    if not risk_cfg:
        risk_cfg = cfg.get("analytics", {})
    return risk_cfg


class AppConfig:
    """Convenience accessor for application configuration."""

    def __init__(self, config_path: Optional[str] = None):
        self._cfg = load_config(config_path)

    @property
    def raw_config(self) -> Dict[str, Any]:
        return self._cfg

    @property
    def db_config(self) -> Dict[str, Any]:
        return get_db_config(self._cfg)

    @property
    def analytics_config(self) -> Dict[str, Any]:
        return get_risk_config(self._cfg)

    @property
    def ui_config(self) -> Dict[str, Any]:
        return self._cfg.get("ui", {
            "app_title": "Portfolio Risk & Volatility Analytics",
            "app_icon": "📈",
            "theme": "light",
            "initial_sidebar_state": "collapsed"
        })


config = AppConfig()
