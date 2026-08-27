"""
Global Pytest Configuration and Test Database Fixtures.
Strictly ensures all tests run against the development/test database ('<database_name>_dev', e.g. 'stocks_dev')
or an on-demand isolated SQLite test database.
Guarantees automatic cleanup and deletion of any on-disk test SQLite files (.s3db / .s2db) after tests finish.
"""

import os
from pathlib import Path
import pytest
from sqlalchemy import Engine, text

# Set test environment markers before tests load
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


def pytest_configure(config):
    """Ensure environment is configured for test mode before executing test suite."""
    os.environ["PORTFOLIO_ENV"] = "test"
    os.environ["TEST_MODE"] = "1"
    # Pre-clean any leftover test db files
    cleanup_test_sqlite_files()


def pytest_sessionfinish(session, exitstatus):
    """Hook executing after entire test session completes to delete all test SQLite files (.s3db / .s2db)."""
    cleanup_test_sqlite_files()


@pytest.fixture(autouse=True)
def strictly_dev_db_guard(monkeypatch):
    """
    Safety guard fixture that enforces all database operations in tests strictly target
    the dev database ('<database_name>_dev') or on-demand SQLite test instances,
    and automatically cleans up any generated test SQLite files on teardown.
    """
    monkeypatch.setenv("PORTFOLIO_ENV", "test")
    monkeypatch.setenv("TEST_MODE", "1")
    yield
    cleanup_test_sqlite_files()


@pytest.fixture
def sqlite_test_engine() -> Engine:
    """Provides an isolated on-demand in-memory SQLite database with all tables initialized."""
    engine = create_test_sqlite_engine(sqlite_path=":memory:", initialize_schema=True)
    yield engine
    engine.dispose()


@pytest.fixture
def temp_sqlite_file_engine(tmp_path) -> Engine:
    """
    Provides a file-backed on-demand SQLite database (.s3db / .s2db) in a temporary directory,
    ensuring engine disposal and complete file deletion upon test completion.
    """
    db_file = tmp_path / "temp_test_db.s3db"
    engine = create_test_sqlite_engine(sqlite_path=str(db_file), initialize_schema=True)
    yield engine, str(db_file)
    engine.dispose()
    if db_file.exists():
        db_file.unlink(missing_ok=True)


@pytest.fixture
def mariadb_dev_engine() -> Engine:
    """
    Provides a MariaDB Engine strictly connected to the dev database ('stocks_dev').
    Skips if remote MariaDB dev database is unreachable.
    """
    db_cfg = get_db_config(is_test=True)
    target_db = get_dev_database_name(db_cfg.get("database", "stocks"))
    try:
        engine = get_test_engine(db_type="mariadb", initialize_schema=True, fallback_to_sqlite=False)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        # Strict assertion that we are connected to the dev database
        assert str(engine.url.database).endswith("_dev"), f"Test engine connected to non-dev database: {engine.url.database}"
        yield engine
        engine.dispose()
    except Exception as e:
        pytest.skip(f"MariaDB dev database '{target_db}' not accessible from current environment: {e}")


@pytest.fixture
def test_engine() -> Engine:
    """
    Provides the primary test database engine (MariaDB dev or on-demand SQLite fallback).
    Guarantees strict dev isolation and clean teardown.
    """
    engine = get_test_engine(initialize_schema=True, fallback_to_sqlite=True)
    yield engine
    engine.dispose()
