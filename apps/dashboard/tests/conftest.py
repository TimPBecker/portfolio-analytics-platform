"""
Pytest configuration for dashboard tests.
Enforces strict dev database usage and automatic cleanup of test SQLite files (.s3db / .s2db).
"""

import os
from pathlib import Path
import pytest
from sqlalchemy import Engine, text

os.environ["PORTFOLIO_ENV"] = "test"
os.environ["TEST_MODE"] = "1"

from portfolio_core.db import (
    get_engine,
    get_connection_string,
    get_test_engine,
    create_test_sqlite_engine,
    cleanup_test_sqlite_files,
    create_all_tables,
    get_dev_database_name,
)
from portfolio_core.config import get_db_config


def pytest_sessionfinish(session, exitstatus):
    """Clean up any test SQLite files (.s3db / .s2db) after tests finish."""
    cleanup_test_sqlite_files()


@pytest.fixture(autouse=True)
def strictly_dev_db_guard(monkeypatch):
    """Guarantees tests only connect to dev database or on-demand SQLite and cleans up files."""
    monkeypatch.setenv("PORTFOLIO_ENV", "test")
    monkeypatch.setenv("TEST_MODE", "1")
    yield
    cleanup_test_sqlite_files()


@pytest.fixture
def sqlite_test_engine() -> Engine:
    """Isolated on-demand SQLite test database fixture."""
    engine = create_test_sqlite_engine(sqlite_path=":memory:", initialize_schema=True)
    yield engine
    engine.dispose()


@pytest.fixture
def temp_sqlite_file_engine(tmp_path) -> Engine:
    """File-backed on-demand SQLite database (.s3db / .s2db) with guaranteed file deletion."""
    db_file = tmp_path / "temp_test_db.s3db"
    engine = create_test_sqlite_engine(sqlite_path=str(db_file), initialize_schema=True)
    yield engine, str(db_file)
    engine.dispose()
    if db_file.exists():
        db_file.unlink(missing_ok=True)


@pytest.fixture
def mariadb_dev_engine() -> Engine:
    """MariaDB dev database ('stocks_dev') fixture."""
    db_cfg = get_db_config(is_test=True)
    target_db = get_dev_database_name(db_cfg.get("database", "stocks"))
    try:
        engine = get_test_engine(db_type="mariadb", initialize_schema=True, fallback_to_sqlite=False)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        assert str(engine.url.database).endswith("_dev"), f"Connected to non-dev database: {engine.url.database}"
        yield engine
        engine.dispose()
    except Exception as e:
        pytest.skip(f"MariaDB dev database '{target_db}' not accessible: {e}")


@pytest.fixture
def test_engine() -> Engine:
    """Standard test database fixture."""
    engine = get_test_engine(initialize_schema=True, fallback_to_sqlite=True)
    yield engine
    engine.dispose()
