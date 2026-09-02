"""
Centralized Configuration Loader.
Loads YAML configuration and environment variables (.env) safely without hardcoding secrets.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
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


def is_dev_environment(is_dev: Optional[bool] = None) -> bool:
    """
    Determines whether current execution context is in development mode.
    Checks explicit parameter or environment variables (PORTFOLIO_ENV in ('development', 'dev'), DEV_MODE=1).
    """
    if is_dev is not None:
        return bool(is_dev)
    env_mode = os.getenv("PORTFOLIO_ENV", "").lower()
    if env_mode in ("development", "dev"):
        return True
    if os.getenv("DEV_MODE") == "1":
        return True
    return False


def get_db_config(
    config_dict: Optional[Dict[str, Any]] = None,
    is_test: Optional[bool] = None,
    is_dev: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Extracts database configuration dictionary.
    When in test or development environment (is_test=True, is_dev=True, PORTFOLIO_ENV in ('development', 'dev', 'test'),
    TEST_MODE=1, DEV_MODE=1, or active pytest session), strictly restricts MariaDB database names to '<database>_dev'
    and filters configured databases list to only include databases ending with '_dev'.
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

    in_dev = is_dev_environment(is_dev) if is_dev is not None else (is_dev_environment() if not in_test else False)

    # Resolve active database name from environment if provided
    env_db = os.getenv("DB_NAME")
    if env_db:
        resolved_cfg["database"] = env_db

    # Parse configured available databases list
    raw_dbs = None
    if os.getenv("DB_DATABASES"):
        raw_dbs = [d.strip() for d in os.getenv("DB_DATABASES").split(",") if d.strip()]
    elif "databases" in resolved_cfg:
        raw_dbs = resolved_cfg.get("databases")
    elif "available_databases" in resolved_cfg:
        raw_dbs = resolved_cfg.get("available_databases")

    if isinstance(raw_dbs, list):
        databases = [str(d).strip() for d in raw_dbs if str(d).strip()]
    elif isinstance(raw_dbs, str):
        databases = [d.strip() for d in raw_dbs.split(",") if d.strip()]
    else:
        databases = [resolved_cfg.get("database", "stocks")]

    # Ensure active database is included in the list
    active_db = resolved_cfg.get("database", "stocks")
    if active_db and active_db not in databases:
        databases = [active_db] + [d for d in databases if d != active_db]

    if (in_test or in_dev) and resolved_cfg:
        db_type = str(resolved_cfg.get("type", "mariadb")).lower()
        if db_type in ("mariadb", "mysql"):
            base_db = resolved_cfg.get("database") or os.getenv("DB_NAME")
            if base_db and not base_db.endswith("_dev"):
                resolved_cfg["database"] = f"{base_db}_dev"
            # Map all available database names to dev names in test/dev mode and filter to only _dev
            dev_dbs = []
            for d in databases:
                clean_d = str(d).strip()
                dev_dbs.append(clean_d if clean_d.endswith("_dev") else f"{clean_d}_dev")
            databases = [d for d in dict.fromkeys(dev_dbs) if d.endswith("_dev")]
        elif db_type == "sqlite":
            sp = resolved_cfg.get("sqlite_path")
            if sp and not sp.endswith("_dev.s3db") and sp != ":memory:":
                base = sp[:-5] if sp.endswith(".s3db") else sp
                resolved_cfg["sqlite_path"] = f"{base}_dev.s3db"

    resolved_cfg["databases"] = databases
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
    def databases(self) -> List[str]:
        return self.db_config.get("databases", ["stocks"])

    @property
    def analytics_config(self) -> Dict[str, Any]:
        return get_risk_config(self._cfg)

    @property
    def ui_config(self) -> Dict[str, Any]:
        return self._cfg.get("ui", {
            "app_title": "Portfolio Risk Analytics",
            "app_icon": "📈",
            "theme": "light",
            "initial_sidebar_state": "collapsed"
        })


config = AppConfig()
