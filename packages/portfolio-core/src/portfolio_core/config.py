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
    """Finds config.yaml by searching upwards from current or start path."""
    current = Path(start_path).resolve() if start_path else Path.cwd().resolve()
    
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
    """Finds .env file by searching upwards to the repository root."""
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


def get_db_config(config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extracts database configuration dictionary."""
    cfg = config_dict if config_dict is not None else load_config()
    db_cfg = cfg.get("resources", {}).get("db", {}).get("config", {})
    if not db_cfg:
        db_cfg = cfg.get("database", {})
    if not db_cfg:
        db_cfg = cfg.get("db", {})
    return db_cfg


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
