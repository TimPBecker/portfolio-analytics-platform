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
    """Verify get_engine and get_connection_string accept explicit database name parameter."""
    from portfolio_core.db import get_engine, get_connection_string

    # In test mode with explicit dev database parameter
    conn_str_test = get_connection_string(
        db_type="mariadb",
        user="test_user",
        password="test_password",
        host="localhost",
        port=3306,
        database="custom_portfolio_dev",
        is_test=True
    )
    assert conn_str_test.endswith("/custom_portfolio_dev")

    # In test mode with base database name (auto-mapped to _dev)
    conn_str_auto_dev = get_connection_string(
        db_type="mariadb",
        user="test_user",
        password="test_password",
        host="localhost",
        port=3306,
        database="custom_portfolio",
        is_test=True
    )
    assert conn_str_auto_dev.endswith("/custom_portfolio_dev")

    # In production non-test mode explicit resolution
    conn_str_prod = get_connection_string(
        db_type="mariadb",
        user="test_user",
        password="test_password",
        host="localhost",
        port=3306,
        database="stocks",
        is_test=False,
        is_dev=False
    )
    assert conn_str_prod.endswith("/stocks")


def test_delete_data_after_date(sqlite_test_engine):
    """Verify delete_data_after_date prunes tables correctly, respects dry_run and include_transactions."""
    from portfolio_core.db import delete_data_after_date, create_all_tables
    import pytest

    sqlite_engine = sqlite_test_engine

    create_all_tables(sqlite_engine)

    with sqlite_engine.begin() as conn:
        # Insert sample rows on 2026-09-05, 2026-09-07, and 2026-09-08
        conn.execute(text("INSERT INTO PORTFOLIO_VALUES (DATE, TOTAL_VALUE, STOCKS, CASH, CURRENCY) VALUES ('2026-09-05', 100, 90, 10, 'GBP')"))
        conn.execute(text("INSERT INTO PORTFOLIO_VALUES (DATE, TOTAL_VALUE, STOCKS, CASH, CURRENCY) VALUES ('2026-09-07', 105, 95, 10, 'GBP')"))
        conn.execute(text("INSERT INTO PORTFOLIO_VALUES (DATE, TOTAL_VALUE, STOCKS, CASH, CURRENCY) VALUES ('2026-09-08', 110, 100, 10, 'GBP')"))

        conn.execute(text("INSERT INTO BENCHMARK_VALUES (DATE, BENCHMARK_CODE, TOTAL_VALUE, STOCKS, CASH, CURRENCY) VALUES ('2026-09-07', 'BM1', 100, 100, 0, 'GBP')"))
        conn.execute(text("INSERT INTO BENCHMARK_VALUES (DATE, BENCHMARK_CODE, TOTAL_VALUE, STOCKS, CASH, CURRENCY) VALUES ('2026-09-08', 'BM1', 102, 102, 0, 'GBP')"))

        conn.execute(text("INSERT INTO TRANSACTIONS (ID, TICKER, TRANSACTION_DATE, QUANTITY) VALUES (1, 'AAPL', '2026-09-07', 10)"))
        conn.execute(text("INSERT INTO TRANSACTIONS (ID, TICKER, TRANSACTION_DATE, QUANTITY) VALUES (2, 'AAPL', '2026-09-08', 20)"))

    # 1. Test Dry Run
    preview = delete_data_after_date("2026-09-07", engine=sqlite_engine, dry_run=True)
    assert preview["PORTFOLIO_VALUES"] == 1
    assert preview["BENCHMARK_VALUES"] == 1
    assert "TRANSACTIONS" not in preview  # excluded by default

    # Verify no rows deleted yet
    with sqlite_engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM PORTFOLIO_VALUES")).scalar() == 3

    # 2. Test Deletion preserving transactions
    res = delete_data_after_date("2026-09-07", engine=sqlite_engine, include_transactions=False)
    assert res["PORTFOLIO_VALUES"] == 1
    assert res["BENCHMARK_VALUES"] == 1
    assert "TRANSACTIONS" not in res

    with sqlite_engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM PORTFOLIO_VALUES")).scalar() == 2
        assert conn.execute(text("SELECT MAX(DATE) FROM PORTFOLIO_VALUES")).scalar() == "2026-09-07"
        assert conn.execute(text("SELECT COUNT(*) FROM BENCHMARK_VALUES")).scalar() == 1
        # Transactions preserved
        assert conn.execute(text("SELECT COUNT(*) FROM TRANSACTIONS")).scalar() == 2

    # 3. Test Deletion including transactions
    res_tx = delete_data_after_date("2026-09-07", engine=sqlite_engine, include_transactions=True)
    assert res_tx["TRANSACTIONS"] == 1

    with sqlite_engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM TRANSACTIONS")).scalar() == 1
        assert conn.execute(text("SELECT MAX(TRANSACTION_DATE) FROM TRANSACTIONS")).scalar() == "2026-09-07"

    # 4. Test invalid date format raises ValueError
    with pytest.raises(ValueError):
        delete_data_after_date("invalid-date", engine=sqlite_engine)


def test_get_target_download_date():
    from portfolio_core.db import get_target_download_date
    from datetime import date

    # Monday -> Monday
    assert get_target_download_date("2026-09-21") == "2026-09-21"
    # Friday -> Friday
    assert get_target_download_date("2026-09-25") == "2026-09-25"
    # Saturday -> Friday
    assert get_target_download_date("2026-09-26") == "2026-09-25"
    # Sunday -> Friday
    assert get_target_download_date("2026-09-27") == "2026-09-25"
    # date object
    assert get_target_download_date(date(2026, 9, 27)) == "2026-09-25"
    # default (no args) must be a weekday
    res_today = get_target_download_date()
    dt = pd.to_datetime(res_today).date()
    assert dt.weekday() < 5


def test_get_current_day_delete_date(sqlite_test_engine):
    from portfolio_core.db import get_current_day_delete_date, create_all_tables
    sqlite_engine = sqlite_test_engine
    create_all_tables(sqlite_engine)

    # Weekday -> that day
    assert get_current_day_delete_date(engine=sqlite_engine, asof="2026-09-25") == "2026-09-25"
    # Saturday with no data -> last weekday (Friday)
    assert get_current_day_delete_date(engine=sqlite_engine, asof="2026-09-26") == "2026-09-25"

    # Saturday WITH data in ASSET_PRICES -> Saturday
    with sqlite_engine.begin() as conn:
        conn.execute(text("INSERT INTO ASSET_PRICES (DATE, TICKER, CURRENCY, OPEN, HIGH, LOW, CLOSE, VOLUME) VALUES ('2026-09-26', 'AAPL', 'USD', 150, 155, 149, 152, 1000)"))
    assert get_current_day_delete_date(engine=sqlite_engine, asof="2026-09-26") == "2026-09-26"


def test_delete_data_for_date(sqlite_test_engine):
    """Verify delete_data_for_date deletes only the target date and respects dry_run and include_transactions."""
    from portfolio_core.db import delete_data_for_date, create_all_tables

    sqlite_engine = sqlite_test_engine
    create_all_tables(sqlite_engine)

    with sqlite_engine.begin() as conn:
        # Insert sample rows for 2026-09-24, 2026-09-25, 2026-09-26
        conn.execute(text("INSERT INTO PORTFOLIO_VALUES (DATE, TOTAL_VALUE, STOCKS, CASH, CURRENCY) VALUES ('2026-09-24', 100, 90, 10, 'GBP')"))
        conn.execute(text("INSERT INTO PORTFOLIO_VALUES (DATE, TOTAL_VALUE, STOCKS, CASH, CURRENCY) VALUES ('2026-09-25', 105, 95, 10, 'GBP')"))
        conn.execute(text("INSERT INTO PORTFOLIO_VALUES (DATE, TOTAL_VALUE, STOCKS, CASH, CURRENCY) VALUES ('2026-09-26', 110, 100, 10, 'GBP')"))

        conn.execute(text("INSERT INTO BENCHMARK_VALUES (DATE, BENCHMARK_CODE, TOTAL_VALUE, STOCKS, CASH, CURRENCY) VALUES ('2026-09-24', 'BM1', 100, 100, 0, 'GBP')"))
        conn.execute(text("INSERT INTO BENCHMARK_VALUES (DATE, BENCHMARK_CODE, TOTAL_VALUE, STOCKS, CASH, CURRENCY) VALUES ('2026-09-25', 'BM1', 102, 102, 0, 'GBP')"))
        conn.execute(text("INSERT INTO BENCHMARK_VALUES (DATE, BENCHMARK_CODE, TOTAL_VALUE, STOCKS, CASH, CURRENCY) VALUES ('2026-09-26', 'BM1', 104, 104, 0, 'GBP')"))

        conn.execute(text("INSERT INTO TRANSACTIONS (ID, TICKER, TRANSACTION_DATE, QUANTITY) VALUES (1, 'AAPL', '2026-09-24', 10)"))
        conn.execute(text("INSERT INTO TRANSACTIONS (ID, TICKER, TRANSACTION_DATE, QUANTITY) VALUES (2, 'AAPL', '2026-09-25', 20)"))
        conn.execute(text("INSERT INTO TRANSACTIONS (ID, TICKER, TRANSACTION_DATE, QUANTITY) VALUES (3, 'AAPL', '2026-09-26', 30)"))

    # 1. Test Dry Run for 2026-09-25
    preview = delete_data_for_date("2026-09-25", engine=sqlite_engine, dry_run=True)
    assert preview["PORTFOLIO_VALUES"] == 1
    assert preview["BENCHMARK_VALUES"] == 1
    assert "TRANSACTIONS" not in preview  # excluded by default

    # Verify no rows deleted yet
    with sqlite_engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM PORTFOLIO_VALUES")).scalar() == 3

    # 2. Test Deletion preserving transactions
    res = delete_data_for_date("2026-09-25", engine=sqlite_engine, include_transactions=False)
    assert res["PORTFOLIO_VALUES"] == 1
    assert res["BENCHMARK_VALUES"] == 1
    assert "TRANSACTIONS" not in res

    with sqlite_engine.connect() as conn:
        # 2026-09-24 and 2026-09-26 remain
        assert conn.execute(text("SELECT COUNT(*) FROM PORTFOLIO_VALUES")).scalar() == 2
        remaining_dates = conn.execute(text("SELECT DATE FROM PORTFOLIO_VALUES ORDER BY DATE")).scalars().all()
        assert remaining_dates == ["2026-09-24", "2026-09-26"]

        assert conn.execute(text("SELECT COUNT(*) FROM BENCHMARK_VALUES")).scalar() == 2
        # Transactions preserved: all 3 remain
        assert conn.execute(text("SELECT COUNT(*) FROM TRANSACTIONS")).scalar() == 3

    # 3. Test Deletion including transactions on 2026-09-26
    res_tx = delete_data_for_date("2026-09-26", engine=sqlite_engine, include_transactions=True)
    assert res_tx["TRANSACTIONS"] == 1
    assert res_tx["PORTFOLIO_VALUES"] == 1

    with sqlite_engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM TRANSACTIONS")).scalar() == 2
        tx_dates = conn.execute(text("SELECT TRANSACTION_DATE FROM TRANSACTIONS ORDER BY TRANSACTION_DATE")).scalars().all()
        assert tx_dates == ["2026-09-24", "2026-09-25"]

    # 4. Invalid date format
    with pytest.raises(ValueError):
        delete_data_for_date("bad-format", engine=sqlite_engine)


def test_trigger_data_download_empty_transactions(sqlite_test_engine):
    """Verify trigger_data_download gracefully reports when no tickers are recorded."""
    from portfolio_core.db import trigger_data_download, create_all_tables

    sqlite_engine = sqlite_test_engine
    create_all_tables(sqlite_engine)

    res = trigger_data_download(target_date="2026-09-25", engine=sqlite_engine)
    assert res["status"] == "warning"
    assert "No transactions found" in res["message"]
    assert res["tickers_count"] == 0


def test_processed_dates_operations(sqlite_test_engine):
    """Verify PROCESSED_DATES table creation, seeding, record, query, and pruning on delete."""
    from portfolio_core.db import (
        create_all_tables,
        seed_initial_processed_dates,
        record_processed_date,
        fetch_processed_dates,
        get_latest_processed_date,
        is_date_processed,
        delete_data_for_date
    )

    sqlite_engine = sqlite_test_engine
    create_all_tables(sqlite_engine)

    # 1. Table is seeded automatically by create_all_tables
    latest = get_latest_processed_date(engine=sqlite_engine)
    assert latest == "2026-09-24"
    assert is_date_processed("2024-01-01", engine=sqlite_engine) is True
    assert is_date_processed("2026-09-24", engine=sqlite_engine) is True
    assert is_date_processed("2026-09-25", engine=sqlite_engine) is False

    # 2. Record a new processed date
    rec = record_processed_date("2026-09-25", status="SUCCESS", engine=sqlite_engine)
    assert rec == "2026-09-25"
    assert is_date_processed("2026-09-25", engine=sqlite_engine) is True
    assert get_latest_processed_date(engine=sqlite_engine) == "2026-09-25"

    # 3. Fetch with order
    desc_dates = fetch_processed_dates(engine=sqlite_engine, order="DESC")
    assert desc_dates[0] == "2026-09-25"
    asc_dates = fetch_processed_dates(engine=sqlite_engine, order="ASC")
    assert asc_dates[0] == "2024-01-01"

    # 4. Deleting data for 2026-09-25 also prunes PROCESSED_DATES
    del_res = delete_data_for_date("2026-09-25", engine=sqlite_engine)
    assert del_res.get("PROCESSED_DATES") == 1
    assert is_date_processed("2026-09-25", engine=sqlite_engine) is False
    assert get_latest_processed_date(engine=sqlite_engine) == "2026-09-24"

    # 5. Invalid date error
    with pytest.raises(ValueError):
        record_processed_date("invalid-date", engine=sqlite_engine)

