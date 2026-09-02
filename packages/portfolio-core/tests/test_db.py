"""
Comprehensive unit and integration tests for database configuration,
on-demand test SQLite databases, and strict MariaDB '<database>_dev' routing.
"""

import os
import tempfile
import pytest
from sqlalchemy import text, Engine
import pandas as pd

from portfolio_core.db import (
    get_dev_database_name,
    is_test_environment,
    get_connection_string,
    get_engine,
    create_test_sqlite_engine,
    get_test_engine,
    create_all_tables,
    record_transaction,
    fetch_all_transactions,
    fetch_benchmarks_info,
    calculate_and_store_daily_benchmark_values,
)
from portfolio_core.config import get_db_config


def test_get_dev_database_name():
    """Verify that get_dev_database_name appends '_dev' only when not already present."""
    assert get_dev_database_name("stocks") == "stocks_dev"
    assert get_dev_database_name("stocks_dev") == "stocks_dev"
    assert get_dev_database_name("portfolio_analytics") == "portfolio_analytics_dev"
    assert get_dev_database_name("prod_db") == "prod_db_dev"
    assert get_dev_database_name("") == "stocks_dev"
    assert get_dev_database_name(None) == "stocks_dev"


def test_is_test_environment():
    """Verify test environment detection."""
    assert is_test_environment(is_test=True) is True
    assert is_test_environment(is_test=False) is False
    # Under pytest execution, should default to True
    assert is_test_environment() is True


def test_get_connection_string_strictly_uses_dev_mariadb():
    """Verify that get_connection_string in test mode routes MariaDB to <database>_dev."""
    conn_str = get_connection_string(
        db_type="mariadb",
        user="test_user",
        password="test_password",
        host="localhost",
        port=3306,
        database="stocks",
        is_test=True,
    )
    assert conn_str.endswith("/stocks_dev")
    assert "/stocks" not in conn_str.replace("/stocks_dev", "")


def test_get_connection_string_sqlite_test_routing():
    """Verify on-demand SQLite test database routing."""
    # When no path is specified, defaults to :memory: in test mode
    conn_str_mem = get_connection_string(db_type="sqlite", is_test=True)
    assert conn_str_mem == "sqlite:///:memory:"

    # When a .s3db path is specified, converts to _dev.s3db
    conn_str_file = get_connection_string(db_type="sqlite", sqlite_path="stocks.s3db", is_test=True)
    assert conn_str_file == "sqlite:///stocks_dev.s3db"


def test_get_db_config_test_mode():
    """Verify get_db_config resolves database to _dev in test mode."""
    cfg = {
        "resources": {
            "db": {
                "config": {
                    "type": "mariadb",
                    "host": "localhost",
                    "port": 3306,
                    "user": "stocks",
                    "database": "stocks"
                }
            }
        }
    }
    resolved = get_db_config(cfg, is_test=True)
    assert resolved["database"] == "stocks_dev"


def test_create_test_sqlite_engine_in_memory():
    """Verify on-demand in-memory SQLite database generation and full schema initialization."""
    engine = create_test_sqlite_engine(sqlite_path=":memory:", initialize_schema=True)
    assert engine is not None
    assert engine.dialect.name == "sqlite"

    with engine.connect() as conn:
        tables = [r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()]
        assert "TRANSACTIONS" in tables
        assert "ASSET_PRICES" in tables
        assert "FX_RATES" in tables
        assert "CASHFLOWS" in tables
        assert "CASHACCOUNT" in tables
        assert "PORTFOLIO_VALUES" in tables
        assert "PORTFOLIO_VAR" in tables
        assert "PORTFOLIO_SCENARIO_PNL" in tables
        assert "PORTFOLIO_RISK_CONTRIBUTIONS" in tables
        assert "BENCHMARKS" in tables
        assert "BENCHMARK_TRANSACTIONS" in tables
        assert "BENCHMARK_VALUES" in tables

    # Test CRUD operations on on-demand SQLite test database
    tx = record_transaction(ticker="AAPL", transaction_date="2026-08-20", quantity=50.0, engine=engine)
    assert tx["ticker"] == "AAPL"
    assert tx["quantity"] == 50.0

    all_tx = fetch_all_transactions(engine=engine)
    assert len(all_tx) == 1
    assert all_tx["TICKER"].iloc[0] == "AAPL"


def test_create_test_sqlite_engine_on_disk():
    """Verify on-demand on-disk SQLite (.s3db) test database generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_portfolio.s3db")
        engine = create_test_sqlite_engine(sqlite_path=db_path, initialize_schema=True)
        assert os.path.exists(db_path)

        with engine.connect() as conn:
            bm_count = conn.execute(text("SELECT COUNT(*) FROM BENCHMARKS")).scalar()
            assert bm_count > 0

        engine.dispose()


def test_get_test_engine_sqlite():
    """Verify get_test_engine with SQLite."""
    engine = get_test_engine(db_type="sqlite", initialize_schema=True)
    assert engine.dialect.name == "sqlite"
    with engine.connect() as conn:
        count = conn.execute(text("SELECT 1")).scalar()
        assert count == 1


def test_mariadb_dev_database_connection(mariadb_dev_engine):
    """
    Integration test verifying strict MariaDB '<database>_dev' database connection and tables.
    Runs only when MariaDB server is reachable.
    """
    assert mariadb_dev_engine.dialect.name in ("mysql", "mariadb")
    db_name = str(mariadb_dev_engine.url.database)
    assert db_name.endswith("_dev"), f"Database name '{db_name}' must end with '_dev'"

    with mariadb_dev_engine.connect() as conn:
        tables = [r[0] for r in conn.execute(text("SHOW TABLES")).fetchall()]
        assert "TRANSACTIONS" in tables
        assert "ASSET_PRICES" in tables
        assert "FX_RATES" in tables
        assert "PORTFOLIO_VALUES" in tables
        assert "PORTFOLIO_VAR" in tables
        assert "BENCHMARKS" in tables


def test_temp_sqlite_file_engine_fixture_cleanup(temp_sqlite_file_engine):
    """Verify that file-backed SQLite database fixture initializes schema and tracks file path."""
    engine, db_path = temp_sqlite_file_engine
    assert os.path.exists(db_path)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1


def test_cleanup_test_sqlite_files_removes_s3db_and_s2db(tmp_path):
    """Verify cleanup_test_sqlite_files locates and removes test .s3db and .s2db files."""
    from portfolio_core.db import cleanup_test_sqlite_files

    # Create dummy test files
    f1 = tmp_path / "stocks_dev.s3db"
    f2 = tmp_path / "test_portfolio.s2db"
    f3 = tmp_path / "unit_test_sample.s3db"
    f1.write_text("dummy")
    f2.write_text("dummy")
    f3.write_text("dummy")

    assert f1.exists() and f2.exists() and f3.exists()

    removed = cleanup_test_sqlite_files(base_dirs=[str(tmp_path)])
    assert len(removed) == 3
    assert not f1.exists()
    assert not f2.exists()
    assert not f3.exists()


def test_get_db_config_databases_list():
    """Verify that get_db_config resolves configured databases list and defaults."""
    from portfolio_core.config import get_db_config, is_dev_environment

    # Production mode without explicit databases list
    cfg_empty = {"db": {"type": "mariadb", "database": "stocks"}}
    res_empty = get_db_config(cfg_empty, is_test=False, is_dev=False)
    assert res_empty["databases"] == ["stocks"]

    # Production mode with explicit databases list
    cfg_multi = {"db": {"type": "mariadb", "database": "stocks", "databases": ["stocks", "stocks_dev", "sandbox"]}}
    res_multi = get_db_config(cfg_multi, is_test=False, is_dev=False)
    assert res_multi["databases"] == ["stocks", "stocks_dev", "sandbox"]

    # In test/dev mode: databases are strictly mapped to and filtered by dev names
    res_dev = get_db_config(cfg_multi, is_test=False, is_dev=True)
    assert res_dev["databases"] == ["stocks_dev", "sandbox_dev"]
    assert "stocks" not in res_dev["databases"]
    assert "sandbox" not in res_dev["databases"]


def test_dev_environment_database_restriction(monkeypatch):
    """Verify that development mode strictly prevents connecting to non-dev databases."""
    from portfolio_core.db import get_connection_string, is_dev_environment

    # Simulate dev container environment
    monkeypatch.setenv("PORTFOLIO_ENV", "development")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert is_dev_environment() is True

    # Connecting to non-dev database in dev mode MUST raise ValueError
    with pytest.raises(ValueError, match="Security restriction: Connection to non-dev database 'stocks' is forbidden in development mode"):
        get_connection_string(
            db_type="mariadb",
            user="user",
            password="pwd",
            host="localhost",
            port=3306,
            database="stocks",
            is_test=False
        )

    # Connecting to dev database in dev mode succeeds
    conn_dev = get_connection_string(
        db_type="mariadb",
        user="user",
        password="pwd",
        host="localhost",
        port=3306,
        database="stocks_dev",
        is_test=False
    )
    assert conn_dev.endswith("/stocks_dev")


def test_get_engine_with_database_param():
    """Verify get_engine accepts explicit database name parameter."""
    from portfolio_core.db import get_engine, get_connection_string

    # In test mode (asserted by is_test=True)
    conn_str = get_connection_string(database="stocks_dev", is_test=True)
    assert conn_str.endswith("/stocks_dev")

    # In production non-test mode explicit resolution
    conn_str_prod = get_connection_string(
        db_type="mariadb",
        user="user",
        password="pwd",
        host="localhost",
        port=3306,
        database="stocks",
        is_test=False
    )
    assert conn_str_prod.endswith("/stocks")



