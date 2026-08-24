"""
Database access and financial calculation module for pi-dagster-dask.
Provides standalone database interaction, data ingestion, backfilling,
dividend cashflow collection, and portfolio valuation independent of Dagster.
"""

import os
from urllib.parse import quote_plus
from datetime import date
from typing import Optional, List, Dict, Tuple, Any, Union
import pandas as pd
import numpy as np
import yfinance as yf
import time
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def get_connection_string(
    db_type: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    database: Optional[str] = None,
    sqlite_path: Optional[str] = None,
    config_path: Optional[str] = None
) -> str:
    """
    Constructs a database connection string with safe URL-encoding of credentials.
    Supports both 'mariadb' / 'mysql' and 'sqlite' (must end in .s3db).
    All non-secret parameters are loaded from config.yaml without hardcoded code defaults.
    The secret password is read from the DB_PASSWORD environment variable for MariaDB.
    Raises ValueError / FileNotFoundError if any required parameter or config is missing.
    """
    cfg_file = config_path or os.path.join(os.path.dirname(__file__), "config.yaml")
    loaded_cfg = {}
    if os.path.exists(cfg_file):
        import yaml
        with open(cfg_file, "r") as f:
            loaded_cfg = yaml.safe_load(f) or {}

    db_cfg = loaded_cfg.get("resources", {}).get("db", {}).get("config", {})
    t = (db_type or db_cfg.get("type") or "mariadb").lower()

    if t == "sqlite":
        sp = sqlite_path or db_cfg.get("sqlite_path")
        if not sp:
            raise ValueError(
                f"Database configuration error: Missing 'sqlite_path' for SQLite database in '{cfg_file}'. "
                "Please specify resources.db.config.sqlite_path."
            )
        if not sp.endswith(".s3db"):
            raise ValueError(
                f"Database configuration error: SQLite database file path must end with '.s3db' (got '{sp}'). "
                "Ensure your local SQLite file uses the .s3db extension so it is gitignored."
            )
        return f"sqlite:///{sp}"

    # MariaDB / MySQL
    u = user or os.getenv("DB_USER") or db_cfg.get("user")
    p = password
    if hasattr(p, "get_value"):
        p = p.get_value()
    elif not p:
        p = os.getenv("DB_PASSWORD")
    h = host or os.getenv("DB_HOST") or db_cfg.get("host")
    prt = port or (int(os.getenv("DB_PORT")) if os.getenv("DB_PORT") else None) or db_cfg.get("port")
    db = database or os.getenv("DB_NAME") or db_cfg.get("database")

    missing = []
    if not u: missing.append("user")
    if not h: missing.append("host")
    if not prt: missing.append("port")
    if not db: missing.append("database")
    if not p: missing.append("password (DB_PASSWORD environment variable)")

    if missing:
        raise ValueError(
            f"Database configuration error: Missing required parameter(s): {', '.join(missing)}. "
            "Please ensure config.yaml contains host, port, user, database and DB_PASSWORD is set in .env."
        )

    return f"mysql+pymysql://{u}:{quote_plus(str(p))}@{h}:{int(prt)}/{db}"


def get_engine(connection_string=None):
    """Returns a SQLAlchemy engine."""
    if hasattr(connection_string, "connect"):
        return connection_string
    return create_engine(connection_string or get_connection_string())


def test_db_connection(engine: Optional[Engine] = None) -> Tuple[bool, str]:
    """Tests if the database connection is alive and returns status message."""
    try:
        eng = engine or get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connected successfully"
    except Exception as e:
        return False, str(e)


def create_all_tables(engine=None):
    """Ensures that all necessary database tables exist in MariaDB/MySQL or SQLite."""
    engine = get_engine(engine)
    is_sqlite = engine.dialect.name == "sqlite"
    
    if is_sqlite:
        ddl_statements = [
            """
            CREATE TABLE IF NOT EXISTS `TRANSACTIONS` (
                `ID` INTEGER PRIMARY KEY AUTOINCREMENT,
                `TICKER` TEXT NOT NULL,
                `TRANSACTION_DATE` TEXT NOT NULL,
                `QUANTITY` REAL NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS `ASSET_PRICES` (
                `id` INTEGER PRIMARY KEY AUTOINCREMENT,
                `DATE` TEXT NOT NULL,
                `TICKER` TEXT NOT NULL,
                `OPEN` REAL,
                `HIGH` REAL,
                `LOW` REAL,
                `CLOSE` REAL,
                `VOLUME` INTEGER,
                `DIVIDENDS` REAL,
                `STOCK_SPLITS` REAL,
                `CURRENCY` TEXT NOT NULL,
                `COMMENT` TEXT
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS `idx_asset_prices_ticker_date` ON `ASSET_PRICES` (`TICKER`, `DATE`);
            """,
            """
            CREATE TABLE IF NOT EXISTS `FX_RATES` (
                `DATE` TEXT NOT NULL,
                `FROM_CURRENCY` TEXT NOT NULL,
                `TO_CURRENCY` TEXT NOT NULL DEFAULT 'GBP',
                `RATE` REAL NOT NULL,
                `OPEN` REAL,
                `HIGH` REAL,
                `LOW` REAL,
                `CLOSE` REAL,
                `COMMENT` TEXT
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS `idx_fx_rates_pair_date` ON `FX_RATES` (`FROM_CURRENCY`, `TO_CURRENCY`, `DATE`);
            """,
            """
            CREATE TABLE IF NOT EXISTS `CASHFLOWS` (
                `DATE` TEXT NOT NULL,
                `TICKER` TEXT NOT NULL,
                `TYPE` TEXT NOT NULL DEFAULT 'DIVIDEND',
                `SHARES` REAL NOT NULL,
                `DIVIDEND_PER_SHARE` REAL NOT NULL,
                `AMOUNT` REAL NOT NULL,
                `CURRENCY` TEXT NOT NULL,
                `AMOUNT_GBP` REAL NOT NULL,
                `CREATED_AT` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (`DATE`, `TICKER`, `TYPE`)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS `CASHACCOUNT` (
                `DATE` TEXT NOT NULL,
                `CURRENCY` TEXT NOT NULL,
                `DAILY_AMOUNT` REAL NOT NULL DEFAULT 0.0,
                `DAILY_AMOUNT_GBP` REAL NOT NULL DEFAULT 0.0,
                `CUMULATIVE_AMOUNT` REAL NOT NULL,
                `CUMULATIVE_AMOUNT_GBP` REAL NOT NULL,
                `UPDATED_AT` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (`DATE`, `CURRENCY`)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS `PORTFOLIO_VALUES` (
                `DATE` TEXT NOT NULL,
                `TOTAL_VALUE` REAL NOT NULL,
                `STOCKS` REAL NOT NULL,
                `CASH` REAL NOT NULL,
                `CURRENCY` TEXT NOT NULL DEFAULT 'GBP',
                PRIMARY KEY (`DATE`, `CURRENCY`)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS `PORTFOLIO_VAR` (
                `DATE` TEXT NOT NULL,
                `METHOD` TEXT NOT NULL DEFAULT 'Historical Simulation',
                `CONFIDENCE_LEVEL` REAL NOT NULL DEFAULT 0.95,
                `HORIZON_DAYS` INTEGER NOT NULL DEFAULT 1,
                `PORTFOLIO_VALUE_GBP` REAL NOT NULL,
                `VAR_GBP` REAL NOT NULL,
                `VAR_PCT` REAL NOT NULL,
                `CVAR_GBP` REAL,
                `CVAR_PCT` REAL,
                `LOOKBACK_OBSERVATIONS` INTEGER,
                `CREATED_AT` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (`DATE`, `METHOD`, `CONFIDENCE_LEVEL`, `HORIZON_DAYS`)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS `PORTFOLIO_SCENARIO_PNL` (
                `ASOF_DATE` TEXT NOT NULL,
                `SCENARIO_DATE` TEXT NOT NULL,
                `TICKER` TEXT NOT NULL,
                `METHOD` TEXT NOT NULL DEFAULT 'Historical Simulation',
                `SHARES` REAL,
                `PRICE_GBP` REAL,
                `POSITION_VALUE_GBP` REAL,
                `LOG_RETURN` REAL,
                `SCENARIO_PNL_GBP` REAL NOT NULL,
                `CREATED_AT` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (`ASOF_DATE`, `SCENARIO_DATE`, `TICKER`, `METHOD`)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS `PORTFOLIO_RISK_CONTRIBUTIONS` (
                `DATE` TEXT NOT NULL,
                `TICKER` TEXT NOT NULL,
                `METHOD` TEXT NOT NULL DEFAULT 'Historical Simulation',
                `CONFIDENCE_LEVEL` REAL NOT NULL DEFAULT 0.95,
                `HORIZON_DAYS` INTEGER NOT NULL DEFAULT 1,
                `POSITION_VALUE_GBP` REAL NOT NULL,
                `WEIGHT_PCT` REAL NOT NULL,
                `SHAPLEY_VAR_GBP` REAL NOT NULL,
                `SHAPLEY_VAR_PCT` REAL NOT NULL,
                `SHAPLEY_CVAR_GBP` REAL NOT NULL,
                `SHAPLEY_CVAR_PCT` REAL NOT NULL,
                `STANDALONE_VAR_GBP` REAL,
                `DIVERSIFICATION_BENEFIT_GBP` REAL,
                `CREATED_AT` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (`DATE`, `TICKER`, `METHOD`, `CONFIDENCE_LEVEL`, `HORIZON_DAYS`)
            );
            """
        ]
    else:
        ddl_statements = [
            """
            CREATE TABLE IF NOT EXISTS `TRANSACTIONS` (
                `ID` INT AUTO_INCREMENT PRIMARY KEY,
                `TICKER` VARCHAR(255) NOT NULL,
                `TRANSACTION_DATE` DATE NOT NULL,
                `QUANTITY` DECIMAL(15, 6) NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """,
            """
            CREATE TABLE IF NOT EXISTS `ASSET_PRICES` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `DATE` DATE NOT NULL,
                `TICKER` VARCHAR(255) NOT NULL,
                `OPEN` DECIMAL(12, 4) DEFAULT NULL,
                `HIGH` DECIMAL(12, 4) DEFAULT NULL,
                `LOW` DECIMAL(12, 4) DEFAULT NULL,
                `CLOSE` DECIMAL(12, 4) DEFAULT NULL,
                `VOLUME` BIGINT DEFAULT NULL,
                `DIVIDENDS` DECIMAL(10, 4) DEFAULT NULL,
                `STOCK_SPLITS` DECIMAL(10, 4) DEFAULT NULL,
                `CURRENCY` VARCHAR(10) NOT NULL,
                `COMMENT` VARCHAR(255) DEFAULT NULL,
                INDEX `idx_asset_prices_ticker_date` (`TICKER`, `DATE`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """,
            """
            CREATE TABLE IF NOT EXISTS `FX_RATES` (
                `DATE` VARCHAR(10) NOT NULL,
                `FROM_CURRENCY` VARCHAR(10) NOT NULL,
                `TO_CURRENCY` VARCHAR(10) NOT NULL DEFAULT 'GBP',
                `RATE` DOUBLE NOT NULL,
                `OPEN` DOUBLE DEFAULT NULL,
                `HIGH` DOUBLE DEFAULT NULL,
                `LOW` DOUBLE DEFAULT NULL,
                `CLOSE` DOUBLE DEFAULT NULL,
                `COMMENT` VARCHAR(255) DEFAULT NULL,
                INDEX `idx_fx_rates_pair_date` (`FROM_CURRENCY`, `TO_CURRENCY`, `DATE`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """,
            """
            CREATE TABLE IF NOT EXISTS `CASHFLOWS` (
                `DATE` DATE NOT NULL,
                `TICKER` VARCHAR(255) NOT NULL,
                `TYPE` VARCHAR(50) NOT NULL DEFAULT 'DIVIDEND',
                `SHARES` DECIMAL(15, 6) NOT NULL,
                `DIVIDEND_PER_SHARE` DECIMAL(12, 4) NOT NULL,
                `AMOUNT` DECIMAL(15, 2) NOT NULL COMMENT 'Dividend payout in original currency',
                `CURRENCY` VARCHAR(10) NOT NULL COMMENT 'Payment currency',
                `AMOUNT_GBP` DECIMAL(15, 2) NOT NULL COMMENT 'Dividend payout converted to GBP',
                `CREATED_AT` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (`DATE`, `TICKER`, `TYPE`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """,
            """
            CREATE TABLE IF NOT EXISTS `CASHACCOUNT` (
                `DATE` DATE NOT NULL,
                `CURRENCY` VARCHAR(10) NOT NULL,
                `DAILY_AMOUNT` DECIMAL(15, 2) NOT NULL DEFAULT 0.00 COMMENT 'Net cashflow on this date in original currency',
                `DAILY_AMOUNT_GBP` DECIMAL(15, 2) NOT NULL DEFAULT 0.00 COMMENT 'Net cashflow on this date converted to GBP',
                `CUMULATIVE_AMOUNT` DECIMAL(15, 2) NOT NULL COMMENT 'Cumulative cash balance in original currency up to this date',
                `CUMULATIVE_AMOUNT_GBP` DECIMAL(15, 2) NOT NULL COMMENT 'Cumulative cash balance converted to GBP up to this date',
                `UPDATED_AT` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (`DATE`, `CURRENCY`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """,
            """
            CREATE TABLE IF NOT EXISTS `PORTFOLIO_VALUES` (
                `DATE` DATE NOT NULL,
                `TOTAL_VALUE` DECIMAL(15, 2) NOT NULL COMMENT 'Total portfolio valuation (STOCKS + CASH)',
                `STOCKS` DECIMAL(15, 2) NOT NULL COMMENT 'Market value of stock holdings',
                `CASH` DECIMAL(15, 2) NOT NULL COMMENT 'Cumulative cash balance from CASHACCOUNT',
                `CURRENCY` VARCHAR(3) NOT NULL DEFAULT 'GBP',
                PRIMARY KEY (`DATE`, `CURRENCY`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """,
            """
            CREATE TABLE IF NOT EXISTS `PORTFOLIO_VAR` (
                `DATE` DATE NOT NULL,
                `METHOD` VARCHAR(100) NOT NULL DEFAULT 'Historical Simulation',
                `CONFIDENCE_LEVEL` DECIMAL(5, 4) NOT NULL DEFAULT 0.9500,
                `HORIZON_DAYS` INT NOT NULL DEFAULT 1,
                `PORTFOLIO_VALUE_GBP` DECIMAL(15, 2) NOT NULL,
                `VAR_GBP` DECIMAL(15, 2) NOT NULL,
                `VAR_PCT` DECIMAL(8, 4) NOT NULL,
                `CVAR_GBP` DECIMAL(15, 2) DEFAULT NULL,
                `CVAR_PCT` DECIMAL(8, 4) DEFAULT NULL,
                `LOOKBACK_OBSERVATIONS` INT DEFAULT NULL,
                `CREATED_AT` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (`DATE`, `METHOD`, `CONFIDENCE_LEVEL`, `HORIZON_DAYS`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """,
            """
            CREATE TABLE IF NOT EXISTS `PORTFOLIO_SCENARIO_PNL` (
                `ASOF_DATE` DATE NOT NULL,
                `SCENARIO_DATE` DATE NOT NULL,
                `TICKER` VARCHAR(255) NOT NULL,
                `METHOD` VARCHAR(100) NOT NULL DEFAULT 'Historical Simulation',
                `SHARES` DECIMAL(15, 6) DEFAULT NULL,
                `PRICE_GBP` DECIMAL(12, 4) DEFAULT NULL,
                `POSITION_VALUE_GBP` DECIMAL(15, 2) DEFAULT NULL,
                `LOG_RETURN` DECIMAL(12, 6) DEFAULT NULL,
                `SCENARIO_PNL_GBP` DECIMAL(15, 2) NOT NULL,
                `CREATED_AT` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (`ASOF_DATE`, `SCENARIO_DATE`, `TICKER`, `METHOD`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """,
            """
            CREATE TABLE IF NOT EXISTS `PORTFOLIO_RISK_CONTRIBUTIONS` (
                `DATE` DATE NOT NULL,
                `TICKER` VARCHAR(255) NOT NULL,
                `METHOD` VARCHAR(100) NOT NULL DEFAULT 'Historical Simulation',
                `CONFIDENCE_LEVEL` DECIMAL(5, 4) NOT NULL DEFAULT 0.9500,
                `HORIZON_DAYS` INT NOT NULL DEFAULT 1,
                `POSITION_VALUE_GBP` DECIMAL(15, 2) NOT NULL,
                `WEIGHT_PCT` DECIMAL(8, 4) NOT NULL,
                `SHAPLEY_VAR_GBP` DECIMAL(15, 2) NOT NULL,
                `SHAPLEY_VAR_PCT` DECIMAL(8, 4) NOT NULL,
                `SHAPLEY_CVAR_GBP` DECIMAL(15, 2) NOT NULL,
                `SHAPLEY_CVAR_PCT` DECIMAL(8, 4) NOT NULL,
                `STANDALONE_VAR_GBP` DECIMAL(15, 2) DEFAULT NULL,
                `DIVERSIFICATION_BENEFIT_GBP` DECIMAL(15, 2) DEFAULT NULL,
                `CREATED_AT` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (`DATE`, `TICKER`, `METHOD`, `CONFIDENCE_LEVEL`, `HORIZON_DAYS`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
        ]
    
    with engine.connect() as conn:
        # Schema migration check: If CASHACCOUNT has old snapshot schema (no DATE column), recreate table
        try:
            if is_sqlite:
                cols = conn.execute(text("PRAGMA table_info(CASHACCOUNT)")).fetchall()
                col_names = [c[1] for c in cols]
            else:
                cols = conn.execute(text("SHOW COLUMNS FROM `CASHACCOUNT`")).fetchall()
                col_names = [c[0] for c in cols]
            if col_names and "DATE" not in col_names:
                conn.execute(text("DROP TABLE `CASHACCOUNT`"))
                conn.commit()
        except Exception:
            pass

        for stmt in ddl_statements:
            conn.execute(text(stmt))
        conn.commit()


def fetch_all_historical_tickers(engine=None) -> List[str]:
    """
    Retrieves all unique tickers that have ever appeared in the TRANSACTIONS table,
    including both currently held and previously held positions.
    """
    engine = engine or get_engine()
    create_all_tables(engine)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT `TICKER` FROM `TRANSACTIONS` WHERE `TICKER` IS NOT NULL AND `TICKER` <> '' ORDER BY `TICKER` ASC")
        ).scalars().all()
        return [str(r) for r in rows if r]


def fetch_portfolio_positions(asof_date=None, asof=None, engine=None):
    """
    Loads current net share positions for each ticker from the TRANSACTIONS table.
    Filters out closed positions with a net holding of 0. Supports both asof_date and asof args.
    """
    engine = engine or get_engine()
    create_all_tables(engine)
    target_asof = asof_date or asof
    
    if target_asof:
        query = """
            SELECT `TICKER`, SUM(`QUANTITY`) AS `shares`
            FROM `TRANSACTIONS`
            WHERE `TRANSACTION_DATE` <= :asof
            GROUP BY `TICKER`
            HAVING SUM(`QUANTITY`) <> 0
            ORDER BY `TICKER` ASC
        """
        params = {"asof": str(target_asof)[:10]}
    else:
        query = """
            SELECT `TICKER`, SUM(`QUANTITY`) AS `shares`
            FROM `TRANSACTIONS`
            GROUP BY `TICKER`
            HAVING SUM(`QUANTITY`) <> 0
            ORDER BY `TICKER` ASC
        """
        params = {}

    with engine.connect() as conn:
        positions = conn.execute(text(query), params).mappings()
        return {row["TICKER"]: float(row["shares"]) for row in positions}


def fetch_and_store_ticker(ticker: str, shares: float = 0.0, history_days: int = 520, connection_string: Optional[str] = None, engine=None):
    """
    Fetches and backfills daily price and dividend history from yfinance for a given ticker,
    ensuring at least `history_days` (default: 520) of continuous price history up to today.
    Inserts only new/missing dates into ASSET_PRICES.
    """
    engine = get_engine(engine or connection_string)
    create_all_tables(engine)
    
    # Calculate target calendar start date for the required history_days (e.g. 520 trading days ~ 780 calendar days)
    cal_days = int(history_days * 1.5)
    target_start = (pd.Timestamp.now() - pd.Timedelta(days=cal_days)).strftime("%Y-%m-%d")
    
    existing_dates = set()
    latest_date = None
    min_date = None
    try:
        with engine.connect() as conn:
            dates_res = conn.execute(
                text("SELECT `DATE` FROM `ASSET_PRICES` WHERE `TICKER` = :ticker"),
                {"ticker": ticker}
            ).scalars().all()
            if dates_res:
                existing_dates = {str(d) for d in dates_res}
                latest_date = max(existing_dates)
                min_date = min(existing_dates)
    except Exception:
        pass

    stock = yf.Ticker(ticker)
    
    # If no data exists, or if existing history does not go back far enough (min_date > target_start):
    # Fetch from target_start to backfill the full required historical window
    if not existing_dates or (min_date and min_date > target_start):
        fetch_start = target_start
    elif latest_date:
        fetch_start = latest_date
    else:
        fetch_start = target_start

    hist = pd.DataFrame()
    for attempt in range(3):
        try:
            hist = stock.history(start=fetch_start)
            if not hist.empty:
                break
        except Exception:
            time.sleep(1)

    # Fallback to period="2y" if start-date query returned empty or single row on backfill
    if (hist.empty or len(hist) <= 1) and (not existing_dates or (min_date and min_date > target_start)):
        try:
            hist = stock.history(period="2y")
        except Exception:
            pass
    
    currency = None
    try:
        if hasattr(stock, "history_metadata") and stock.history_metadata:
            currency = stock.history_metadata.get("currency")
    except Exception:
        pass
    if not currency:
        try:
            currency = getattr(stock.fast_info, "currency", None) if hasattr(stock, "fast_info") else None
        except Exception:
            pass
    if not currency:
        try:
            currency = stock.info.get("currency")
        except Exception:
            pass

    if not hist.empty:
        # Drop rows where CLOSE is NaN
        close_col = "Close" if "Close" in hist.columns else "CLOSE"
        if close_col in hist.columns:
            hist = hist.dropna(subset=[close_col])
        if hist.empty:
            return None

        hist = hist.reset_index()
        hist.columns = [col.replace(" ", "_").upper() for col in hist.columns]
        hist["DATE"] = pd.to_datetime(hist["DATE"]).dt.strftime("%Y-%m-%d")
        hist["TICKER"] = ticker
        hist["CURRENCY"] = currency
        
        # Filter out rows that already exist in ASSET_PRICES
        if existing_dates:
            new_rows = hist[~hist["DATE"].isin(existing_dates)].copy()
        else:
            new_rows = hist.copy()

        target_columns = [
            "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME",
            "DIVIDENDS", "STOCK_SPLITS", "TICKER", "CURRENCY"
        ]
        columns_to_keep = [col for col in target_columns if col in new_rows.columns]
        new_rows = new_rows[columns_to_keep]

        if not new_rows.empty:
            new_rows.to_sql("ASSET_PRICES", con=engine, if_exists="append", index=False)
            rows_written = len(new_rows)
        else:
            rows_written = 0
        
        latest_price = float(hist["CLOSE"].iloc[-1])
        return {
            "Ticker": ticker,
            "Currency": currency,
            "Rows Written": rows_written,
            "Total Observations": len(existing_dates) + rows_written,
            "Latest Close": latest_price,
            "Total Value": latest_price * shares
        }
    else:
        if latest_date:
            with engine.connect() as conn:
                res = conn.execute(
                    text("SELECT `CLOSE`, `CURRENCY` FROM `ASSET_PRICES` WHERE `TICKER` = :ticker ORDER BY `DATE` DESC LIMIT 1"),
                    {"ticker": ticker}
                ).mappings().first()
                latest_price = float(res["CLOSE"]) if res and res.get("CLOSE") is not None else 0.0
                currency = res.get("CURRENCY") if res else currency
            return {
                "Ticker": ticker,
                "Currency": currency,
                "Rows Written": 0,
                "Total Observations": len(existing_dates),
                "Latest Close": latest_price,
                "Total Value": latest_price * shares
            }
        return None


def backfill_missing_prices(engine=None):
    """
    Forward-fills missing dates across all tickers in ASSET_PRICES so all assets
    share a contiguous daily timeline. Also purges any corrupted NULL/NaN price rows.
    """
    engine = engine or get_engine()
    
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM `ASSET_PRICES` WHERE `CLOSE` IS NULL OR `CLOSE` = 0 OR `CLOSE` = 'NaN'"))
        conn.commit()
        df = pd.read_sql(
            text("SELECT `DATE`, `OPEN`, `HIGH`, `LOW`, `CLOSE`, `VOLUME`, `DIVIDENDS`, `STOCK_SPLITS`, `TICKER`, `CURRENCY` FROM `ASSET_PRICES` WHERE `CLOSE` IS NOT NULL AND `CLOSE` > 0 ORDER BY `DATE` ASC"),
            con=conn
        )
    
    if df.empty:
        return {"rows_backfilled": 0, "summary_df": pd.DataFrame()}
        
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["CLOSE"])
    df = df.drop_duplicates(subset=["TICKER", "DATE"], keep="last")
    
    all_dates = sorted(df["DATE"].unique())
    tickers = df["TICKER"].unique()
    
    backfill_rows = []
    summary_records = []
    
    for ticker in tickers:
        ticker_df = df[df["TICKER"] == ticker].sort_values("DATE")
        if ticker_df.empty:
            continue
            
        start_date = ticker_df["DATE"].min()
        existing_dates = set(ticker_df["DATE"])
        target_dates = [d for d in all_dates if d >= start_date]
        missing_dates = sorted(list(set(target_dates) - existing_dates))
        
        if not missing_dates:
            continue
            
        grid_df = pd.DataFrame({"DATE": target_dates})
        merged = pd.merge(grid_df, ticker_df, on="DATE", how="left")
        
        merged["TICKER"] = ticker
        merged["CURRENCY"] = merged["CURRENCY"].ffill()
        merged["CLOSE"] = merged["CLOSE"].ffill()
        merged["OPEN"] = merged["OPEN"].fillna(merged["CLOSE"])
        merged["HIGH"] = merged["HIGH"].fillna(merged["CLOSE"])
        merged["LOW"] = merged["LOW"].fillna(merged["CLOSE"])
        merged["VOLUME"] = merged["VOLUME"].fillna(0).astype(int)
        merged["DIVIDENDS"] = merged["DIVIDENDS"].fillna(0.0).astype(float)
        merged["STOCK_SPLITS"] = merged["STOCK_SPLITS"].fillna(0.0).astype(float)
        
        missing_rows = merged[merged["DATE"].isin(missing_dates)].copy()
        missing_rows["COMMENT"] = "Backfilled from previous day"
        backfill_rows.append(missing_rows)
        
        summary_records.append({
            "Ticker": ticker,
            "Missing Dates Backfilled": len(missing_dates),
            "Date Range": f"{missing_dates[0]} to {missing_dates[-1]}"
        })

    if backfill_rows:
        backfill_df = pd.concat(backfill_rows, ignore_index=True)
        target_columns = [
            "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME",
            "DIVIDENDS", "STOCK_SPLITS", "TICKER", "CURRENCY", "COMMENT"
        ]
        columns_to_keep = [col for col in target_columns if col in backfill_df.columns]
        backfill_df = backfill_df[columns_to_keep]
        backfill_df.to_sql("ASSET_PRICES", con=engine, if_exists="append", index=False)
        total_backfilled = len(backfill_df)
        summary_df = pd.DataFrame(summary_records)
    else:
        total_backfilled = 0
        summary_df = pd.DataFrame([{"Status": "All tickers have complete coverage"}])
        
    return {"rows_backfilled": total_backfilled, "summary_df": summary_df}


def get_foreign_currencies_from_prices(engine=None):
    """Returns list of distinct foreign currencies in ASSET_PRICES needing FX rates to GBP."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        currencies = conn.execute(
            text("SELECT DISTINCT `CURRENCY` FROM `ASSET_PRICES` WHERE `CURRENCY` IS NOT NULL AND `CURRENCY` <> ''")
        ).scalars().all()
    return [curr for curr in currencies if curr.strip() != "GBP"]


def fetch_and_store_fx_rate(from_curr: str, to_curr: str = "GBP", connection_string: Optional[str] = None, engine=None):
    """
    Fetches exchange rate history from yfinance for from_curr/to_curr (with 0.01 for GBp)
    and appends incremental rows to FX_RATES.
    """
    engine = engine or get_engine(connection_string)
    from_curr_raw = from_curr.strip()
    to_curr_clean = to_curr.strip().upper()
    
    if from_curr_raw in ("GBp", "GBX", "GBp_PENCE"):
        latest_date = None
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT MAX(`DATE`) FROM `FX_RATES` WHERE `FROM_CURRENCY` = :from_curr AND `TO_CURRENCY` = :to_curr"),
                    {"from_curr": from_curr_raw, "to_curr": to_curr_clean}
                ).scalar()
                if result is not None:
                    latest_date = str(result)
        except Exception:
            latest_date = None

        with engine.connect() as conn:
            query = "SELECT DISTINCT `DATE` FROM `ASSET_PRICES` WHERE `CURRENCY` = :from_curr"
            if latest_date:
                query += " AND `DATE` > :latest_date"
            query += " ORDER BY `DATE` ASC"
            dates = conn.execute(
                text(query),
                {"from_curr": from_curr_raw, "latest_date": latest_date} if latest_date else {"from_curr": from_curr_raw}
            ).scalars().all()
            
        if dates:
            rows = pd.DataFrame({
                "DATE": [pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates],
                "FROM_CURRENCY": from_curr_raw,
                "TO_CURRENCY": "GBP",
                "RATE": 0.01,
                "OPEN": 0.01,
                "HIGH": 0.01,
                "LOW": 0.01,
                "CLOSE": 0.01,
                "COMMENT": None
            })
            rows.to_sql("FX_RATES", con=engine, if_exists="append", index=False)
            return {"Pair": f"{from_curr_raw}/GBP", "Rows Written": len(rows), "Latest Rate": 0.01}
        return {"Pair": f"{from_curr_raw}/GBP", "Rows Written": 0, "Latest Rate": 0.01}

    from_curr_clean = from_curr_raw.upper()
    if from_curr_clean == to_curr_clean:
        return None
    
    with engine.connect() as conn:
        earliest_asset_date = conn.execute(
            text("SELECT MIN(`DATE`) FROM `ASSET_PRICES` WHERE `CURRENCY` = :from_curr"),
            {"from_curr": from_curr_raw}
        ).scalar()
        earliest_fx_date = conn.execute(
            text("SELECT MIN(`DATE`) FROM `FX_RATES` WHERE `FROM_CURRENCY` = :from_curr AND `TO_CURRENCY` = :to_curr"),
            {"from_curr": from_curr_raw, "to_curr": to_curr_clean}
        ).scalar()
        latest_fx_date = conn.execute(
            text("SELECT MAX(`DATE`) FROM `FX_RATES` WHERE `FROM_CURRENCY` = :from_curr AND `TO_CURRENCY` = :to_curr"),
            {"from_curr": from_curr_raw, "to_curr": to_curr_clean}
        ).scalar()

    pair_ticker = f"{from_curr_clean}{to_curr_clean}=X"
    stock = yf.Ticker(pair_ticker)
    
    # If historical FX coverage does not reach earliest asset price date, fetch 5 years
    if earliest_asset_date and (earliest_fx_date is None or pd.to_datetime(earliest_fx_date) > pd.to_datetime(earliest_asset_date)):
        hist = stock.history(period="5y")
    elif latest_fx_date:
        hist = stock.history(start=str(latest_fx_date))
    else:
        hist = stock.history(period="5y")
        
    if not hist.empty:
        hist = hist.reset_index()
        hist.columns = [col.replace(" ", "_").upper() for col in hist.columns]
        hist["DATE"] = pd.to_datetime(hist["DATE"]).dt.strftime("%Y-%m-%d")
        hist["FROM_CURRENCY"] = from_curr_raw
        hist["TO_CURRENCY"] = to_curr_clean
        hist["RATE"] = hist["CLOSE"]
        hist["COMMENT"] = None
        
        records = [
            {
                "DATE": str(row["DATE"]),
                "FROM_CURRENCY": str(row["FROM_CURRENCY"]),
                "TO_CURRENCY": str(row["TO_CURRENCY"]),
                "RATE": float(row["RATE"]),
                "OPEN": float(row["OPEN"]),
                "HIGH": float(row["HIGH"]),
                "LOW": float(row["LOW"]),
                "CLOSE": float(row["CLOSE"]),
                "COMMENT": None,
            }
            for _, row in hist.iterrows()
        ]

        if engine.dialect.name == "sqlite":
            upsert_sql = """
            INSERT INTO `FX_RATES` (`DATE`, `FROM_CURRENCY`, `TO_CURRENCY`, `RATE`, `OPEN`, `HIGH`, `LOW`, `CLOSE`, `COMMENT`)
            VALUES (:DATE, :FROM_CURRENCY, :TO_CURRENCY, :RATE, :OPEN, :HIGH, :LOW, :CLOSE, :COMMENT)
            ON CONFLICT(`DATE`, `FROM_CURRENCY`, `TO_CURRENCY`) DO UPDATE SET
                `RATE` = excluded.`RATE`, `CLOSE` = excluded.`CLOSE`, `OPEN` = excluded.`OPEN`, `HIGH` = excluded.`HIGH`, `LOW` = excluded.`LOW`;
            """
        else:
            upsert_sql = """
            INSERT INTO `FX_RATES` (`DATE`, `FROM_CURRENCY`, `TO_CURRENCY`, `RATE`, `OPEN`, `HIGH`, `LOW`, `CLOSE`, `COMMENT`)
            VALUES (:DATE, :FROM_CURRENCY, :TO_CURRENCY, :RATE, :OPEN, :HIGH, :LOW, :CLOSE, :COMMENT)
            ON DUPLICATE KEY UPDATE
                `RATE` = VALUES(`RATE`), `CLOSE` = VALUES(`CLOSE`), `OPEN` = VALUES(`OPEN`), `HIGH` = VALUES(`HIGH`), `LOW` = VALUES(`LOW`);
            """
        with engine.connect() as conn:
            conn.execute(text(upsert_sql), records)
            conn.commit()
            
        latest_rate = float(hist["CLOSE"].iloc[-1])
        return {
            "Pair": f"{from_curr_raw}/{to_curr_clean}",
            "Rows Written": len(records),
            "Latest Rate": latest_rate
        }
    else:
        if latest_fx_date:
            with engine.connect() as conn:
                res = conn.execute(
                    text("SELECT `CLOSE` FROM `FX_RATES` WHERE `FROM_CURRENCY` = :from_curr AND `TO_CURRENCY` = :to_curr ORDER BY `DATE` DESC LIMIT 1"),
                    {"from_curr": from_curr_raw, "to_curr": to_curr_clean}
                ).scalar()
                latest_rate = float(res) if res is not None else 0.0
            return {
                "Pair": f"{from_curr_raw}/{to_curr_clean}",
                "Rows Written": 0,
                "Latest Rate": latest_rate
            }
        return None


def backfill_missing_fx_rates(engine=None):
    """
    Forward-fills and back-fills missing FX dates across all currency pairs aligned with asset price dates.
    """
    engine = engine or get_engine()
    
    with engine.connect() as conn:
        fx_df = pd.read_sql(
            text("SELECT `DATE`, `FROM_CURRENCY`, `TO_CURRENCY`, `RATE`, `OPEN`, `HIGH`, `LOW`, `CLOSE`, `COMMENT` FROM `FX_RATES` ORDER BY `DATE` ASC"),
            con=conn
        )
        asset_dates = conn.execute(
            text("SELECT DISTINCT `DATE` FROM `ASSET_PRICES` ORDER BY `DATE` ASC")
        ).scalars().all()

    if fx_df.empty:
        return {"rows_backfilled": 0, "summary_df": pd.DataFrame()}

    fx_df["DATE"] = pd.to_datetime(fx_df["DATE"]).dt.strftime("%Y-%m-%d")
    fx_df = fx_df.drop_duplicates(subset=["FROM_CURRENCY", "TO_CURRENCY", "DATE"], keep="last")
    
    all_dates = sorted(list(set([pd.to_datetime(d).strftime("%Y-%m-%d") for d in asset_dates] + list(fx_df["DATE"].unique()))))
    pairs = fx_df[["FROM_CURRENCY", "TO_CURRENCY"]].drop_duplicates().to_dict("records")
    
    backfill_rows = []
    summary_records = []
    
    for pair in pairs:
        from_curr = pair["FROM_CURRENCY"]
        to_curr = pair["TO_CURRENCY"]
        pair_df = fx_df[(fx_df["FROM_CURRENCY"] == from_curr) & (fx_df["TO_CURRENCY"] == to_curr)].sort_values("DATE")
        
        if pair_df.empty:
            continue
            
        existing_dates = set(pair_df["DATE"])
        target_dates = all_dates
        missing_dates = sorted(list(set(target_dates) - existing_dates))
        
        if not missing_dates:
            continue
            
        grid_df = pd.DataFrame({"DATE": target_dates})
        merged = pd.merge(grid_df, pair_df, on="DATE", how="left")
        
        merged["FROM_CURRENCY"] = from_curr
        merged["TO_CURRENCY"] = to_curr
        merged["CLOSE"] = merged["CLOSE"].ffill().bfill()
        merged["RATE"] = merged["RATE"].fillna(merged["CLOSE"])
        merged["OPEN"] = merged["OPEN"].fillna(merged["CLOSE"])
        merged["HIGH"] = merged["HIGH"].fillna(merged["CLOSE"])
        merged["LOW"] = merged["LOW"].fillna(merged["CLOSE"])
        
        missing_rows = merged[merged["DATE"].isin(missing_dates)].copy()
        missing_rows["COMMENT"] = "Backfilled from adjacent FX rates"
        backfill_rows.append(missing_rows)
        
        summary_records.append({
            "Pair": f"{from_curr}/{to_curr}",
            "Missing Dates Backfilled": len(missing_dates),
            "Date Range": f"{missing_dates[0]} to {missing_dates[-1]}"
        })

    if backfill_rows:
        backfill_df = pd.concat(backfill_rows, ignore_index=True)
        target_columns = [
            "DATE", "FROM_CURRENCY", "TO_CURRENCY", "RATE",
            "OPEN", "HIGH", "LOW", "CLOSE", "COMMENT"
        ]
        columns_to_keep = [col for col in target_columns if col in backfill_df.columns]
        backfill_df = backfill_df[columns_to_keep]
        backfill_df.to_sql("FX_RATES", con=engine, if_exists="append", index=False)
        total_backfilled = len(backfill_df)
        summary_df = pd.DataFrame(summary_records)
    else:
        total_backfilled = 0
        summary_df = pd.DataFrame([{"Status": "All currency pairs have complete coverage"}])

    return {"rows_backfilled": total_backfilled, "summary_df": summary_df}


def collect_and_store_dividend_cashflows_and_cash_account(backfill_days: int = 0, engine=None):
    """
    Identifies all dividend payouts for held shares from ASSET_PRICES & TRANSACTIONS,
    converts to GBP via FX_RATES, writes individual events to CASHFLOWS,
    and updates aggregated cash balances by currency in CASHACCOUNT.
    
    backfill_days:
      0  -> (Default) Only records dividend cashflows from today / latest date.
      N  -> Backfills dividend cashflows for the last N days (e.g. 30, 90).
      -1 -> Backfills all historical dividend cashflows.
    """
    engine = engine or get_engine()
    create_all_tables(engine)
    
    with engine.connect() as conn:
        prices_df = pd.read_sql(
            text("SELECT `DATE`, `TICKER`, `DIVIDENDS`, `CURRENCY` FROM `ASSET_PRICES` WHERE `DIVIDENDS` > 0 ORDER BY `DATE` ASC"),
            conn
        )
        all_price_dates = conn.execute(
            text("SELECT MIN(`DATE`), MAX(`DATE`) FROM `ASSET_PRICES`")
        ).fetchone()
        fx_df = pd.read_sql(
            text("SELECT `DATE`, `FROM_CURRENCY`, `TO_CURRENCY`, `RATE` FROM `FX_RATES` WHERE `TO_CURRENCY` = 'GBP' ORDER BY `DATE` ASC"),
            conn
        )
        tx_df = pd.read_sql(
            text("SELECT `TICKER`, `TRANSACTION_DATE`, `QUANTITY` FROM `TRANSACTIONS` ORDER BY `TRANSACTION_DATE` ASC"),
            conn
        )
        
    if tx_df.empty:
        return {"records_stored": 0, "total_gbp": 0.0, "summary_df": pd.DataFrame()}
        
    prices_df["DATE"] = pd.to_datetime(prices_df["DATE"]).dt.strftime("%Y-%m-%d") if not prices_df.empty else pd.Series(dtype=str)
    fx_df["DATE"] = pd.to_datetime(fx_df["DATE"]).dt.strftime("%Y-%m-%d") if not fx_df.empty else pd.Series(dtype=str)
    tx_df["TRANSACTION_DATE"] = pd.to_datetime(tx_df["TRANSACTION_DATE"]).dt.strftime("%Y-%m-%d")
    
    all_fx = fx_df[["DATE", "FROM_CURRENCY", "RATE"]].copy() if not fx_df.empty else pd.DataFrame(columns=["DATE", "FROM_CURRENCY", "RATE"])
    all_dates = pd.DataFrame({"DATE": prices_df["DATE"].unique()}) if not prices_df.empty else pd.DataFrame(columns=["DATE"])
    gbp_fx = all_dates.copy()
    gbp_fx["FROM_CURRENCY"] = "GBP"
    gbp_fx["RATE"] = 1.0
    all_fx = pd.concat([all_fx, gbp_fx], ignore_index=True).drop_duplicates(subset=["DATE", "FROM_CURRENCY"])
    
    prices_merged = pd.merge(
        prices_df,
        all_fx,
        left_on=["DATE", "CURRENCY"],
        right_on=["DATE", "FROM_CURRENCY"],
        how="left"
    ) if not prices_df.empty else pd.DataFrame()
    if not prices_merged.empty:
        prices_merged.loc[prices_merged["CURRENCY"].isin(["GBp", "GBX", "GBp_PENCE"]) & prices_merged["RATE"].isna(), "RATE"] = 0.01
        prices_merged.loc[(prices_merged["CURRENCY"] == "GBP") & prices_merged["RATE"].isna(), "RATE"] = 1.0
    
    tx_pivot = tx_df.pivot_table(index="TRANSACTION_DATE", columns="TICKER", values="QUANTITY", aggfunc="sum").fillna(0)
    
    max_price_date = str(all_price_dates[1]) if all_price_dates and all_price_dates[1] else str(tx_df["TRANSACTION_DATE"].max())
    min_price_date = str(all_price_dates[0]) if all_price_dates and all_price_dates[0] else str(tx_df["TRANSACTION_DATE"].min())
    start_date = min(tx_df["TRANSACTION_DATE"].min(), min_price_date)
    end_date = max(tx_df["TRANSACTION_DATE"].max(), max_price_date)
    full_date_range = pd.date_range(start=start_date, end=end_date, freq="D").strftime("%Y-%m-%d")
    tx_grid = tx_pivot.reindex(full_date_range).fillna(0).cumsum()
    
    holdings_long = tx_grid.reset_index().melt(id_vars="index", var_name="TICKER", value_name="SHARES")
    holdings_long.rename(columns={"index": "DATE"}, inplace=True)
    holdings_long = holdings_long[holdings_long["SHARES"] > 0]
    
    div_df = pd.merge(prices_merged, holdings_long, on=["DATE", "TICKER"], how="inner").sort_values(["DATE", "TICKER"])
    if div_df.empty:
        return {"records_stored": 0, "total_gbp": 0.0, "summary_df": pd.DataFrame()}
        
    div_df["TYPE"] = "DIVIDEND"
    div_df["SHARES"] = div_df["SHARES"].astype(float)
    div_df["DIVIDEND_PER_SHARE"] = div_df["DIVIDENDS"].astype(float)
    div_df["AMOUNT"] = (div_df["SHARES"] * div_df["DIVIDEND_PER_SHARE"]).round(2)
    div_df["AMOUNT_GBP"] = (div_df["AMOUNT"] * div_df["RATE"].astype(float)).round(2)
    
    cf_records = [
        {
            "DATE": str(row["DATE"]),
            "TICKER": str(row["TICKER"]),
            "TYPE": "DIVIDEND",
            "SHARES": float(row["SHARES"]),
            "DIVIDEND_PER_SHARE": float(row["DIVIDEND_PER_SHARE"]),
            "AMOUNT": float(row["AMOUNT"]),
            "CURRENCY": str(row["CURRENCY"]),
            "AMOUNT_GBP": float(row["AMOUNT_GBP"]),
        }
        for _, row in div_df.iterrows()
    ]

    # Filter cashflow records based on backfill_days
    if backfill_days == 0:
        latest_cf_date = div_df["DATE"].max() if not div_df.empty else None
        cf_records_to_store = [r for r in cf_records if r["DATE"] == latest_cf_date]
    elif backfill_days is not None and backfill_days > 0:
        unique_cf_dates = sorted(list({r["DATE"] for r in cf_records}))
        cutoff_dates = set(unique_cf_dates[-backfill_days:])
        cf_records_to_store = [r for r in cf_records if r["DATE"] in cutoff_dates]
    else:  # backfill_days < 0 -> all
        cf_records_to_store = cf_records
    
    if engine.dialect.name == "sqlite":
        cf_upsert = """
        INSERT INTO `CASHFLOWS` (`DATE`, `TICKER`, `TYPE`, `SHARES`, `DIVIDEND_PER_SHARE`, `AMOUNT`, `CURRENCY`, `AMOUNT_GBP`)
        VALUES (:DATE, :TICKER, :TYPE, :SHARES, :DIVIDEND_PER_SHARE, :AMOUNT, :CURRENCY, :AMOUNT_GBP)
        ON CONFLICT(`DATE`, `TICKER`, `TYPE`) DO UPDATE SET
            `SHARES` = excluded.`SHARES`,
            `DIVIDEND_PER_SHARE` = excluded.`DIVIDEND_PER_SHARE`,
            `AMOUNT` = excluded.`AMOUNT`,
            `CURRENCY` = excluded.`CURRENCY`,
            `AMOUNT_GBP` = excluded.`AMOUNT_GBP`;
        """
        account_upsert = """
        INSERT INTO `CASHACCOUNT` (`CURRENCY`, `TOTAL_AMOUNT`, `TOTAL_AMOUNT_GBP`, `LAST_TRANSACTION_DATE`)
        VALUES (:CURRENCY, :TOTAL_AMOUNT, :TOTAL_AMOUNT_GBP, :LAST_TRANSACTION_DATE)
        ON CONFLICT(`CURRENCY`) DO UPDATE SET
            `TOTAL_AMOUNT` = excluded.`TOTAL_AMOUNT`,
            `TOTAL_AMOUNT_GBP` = excluded.`TOTAL_AMOUNT_GBP`,
            `LAST_TRANSACTION_DATE` = excluded.`LAST_TRANSACTION_DATE`;
        """
    else:
        cf_upsert = """
        INSERT INTO `CASHFLOWS` (`DATE`, `TICKER`, `TYPE`, `SHARES`, `DIVIDEND_PER_SHARE`, `AMOUNT`, `CURRENCY`, `AMOUNT_GBP`)
        VALUES (:DATE, :TICKER, :TYPE, :SHARES, :DIVIDEND_PER_SHARE, :AMOUNT, :CURRENCY, :AMOUNT_GBP)
        ON DUPLICATE KEY UPDATE
            `SHARES` = VALUES(`SHARES`),
            `DIVIDEND_PER_SHARE` = VALUES(`DIVIDEND_PER_SHARE`),
            `AMOUNT` = VALUES(`AMOUNT`),
            `CURRENCY` = VALUES(`CURRENCY`),
            `AMOUNT_GBP` = VALUES(`AMOUNT_GBP`);
        """
        account_upsert = """
        INSERT INTO `CASHACCOUNT` (`CURRENCY`, `TOTAL_AMOUNT`, `TOTAL_AMOUNT_GBP`, `LAST_TRANSACTION_DATE`)
        VALUES (:CURRENCY, :TOTAL_AMOUNT, :TOTAL_AMOUNT_GBP, :LAST_TRANSACTION_DATE)
        ON DUPLICATE KEY UPDATE
            `TOTAL_AMOUNT` = VALUES(`TOTAL_AMOUNT`),
            `TOTAL_AMOUNT_GBP` = VALUES(`TOTAL_AMOUNT_GBP`),
            `LAST_TRANSACTION_DATE` = VALUES(`LAST_TRANSACTION_DATE`);
        """
    
    # 1. Aggregate daily cashflows by DATE and CURRENCY
    daily_cf = div_df.groupby(["DATE", "CURRENCY"]).agg(
        DAILY_AMOUNT=("AMOUNT", "sum"),
        DAILY_AMOUNT_GBP=("AMOUNT_GBP", "sum")
    ).reset_index()
    
    # 2. Build complete daily historical timeline for each currency across all dates
    currencies = list(div_df["CURRENCY"].unique())
    if not currencies:
        currencies = ["GBP"]
        
    grid_dates = pd.DataFrame({"DATE": full_date_range})
    all_account_records = []
    
    for curr in currencies:
        curr_cf = daily_cf[daily_cf["CURRENCY"] == curr] if not daily_cf.empty else pd.DataFrame()
        merged = pd.merge(grid_dates, curr_cf, on="DATE", how="left")
        merged["CURRENCY"] = curr
        merged["DAILY_AMOUNT"] = merged["DAILY_AMOUNT"].fillna(0.0).round(2)
        merged["DAILY_AMOUNT_GBP"] = merged["DAILY_AMOUNT_GBP"].fillna(0.0).round(2)
        merged["CUMULATIVE_AMOUNT"] = merged["DAILY_AMOUNT"].cumsum().round(2)
        merged["CUMULATIVE_AMOUNT_GBP"] = merged["DAILY_AMOUNT_GBP"].cumsum().round(2)
        
        for _, row in merged.iterrows():
            all_account_records.append({
                "DATE": str(row["DATE"]),
                "CURRENCY": str(row["CURRENCY"]),
                "DAILY_AMOUNT": float(row["DAILY_AMOUNT"]),
                "DAILY_AMOUNT_GBP": float(row["DAILY_AMOUNT_GBP"]),
                "CUMULATIVE_AMOUNT": float(row["CUMULATIVE_AMOUNT"]),
                "CUMULATIVE_AMOUNT_GBP": float(row["CUMULATIVE_AMOUNT_GBP"]),
            })
            
    # 3. Filter records according to backfill_days
    if backfill_days == 0:
        latest_date = full_date_range[-1]
        account_records_to_store = [r for r in all_account_records if r["DATE"] == latest_date]
        cf_records_to_store = [r for r in cf_records if r["DATE"] == latest_date]
    elif backfill_days is not None and backfill_days > 0:
        cutoff_dates = set(full_date_range[-backfill_days:])
        account_records_to_store = [r for r in all_account_records if r["DATE"] in cutoff_dates]
        cf_records_to_store = [r for r in cf_records if r["DATE"] in cutoff_dates]
    else:  # backfill_days < 0 -> all historical dates
        account_records_to_store = all_account_records
        cf_records_to_store = cf_records
    
    if engine.dialect.name == "sqlite":
        cf_upsert = """
        INSERT INTO `CASHFLOWS` (`DATE`, `TICKER`, `TYPE`, `SHARES`, `DIVIDEND_PER_SHARE`, `AMOUNT`, `CURRENCY`, `AMOUNT_GBP`)
        VALUES (:DATE, :TICKER, :TYPE, :SHARES, :DIVIDEND_PER_SHARE, :AMOUNT, :CURRENCY, :AMOUNT_GBP)
        ON CONFLICT(`DATE`, `TICKER`, `TYPE`) DO UPDATE SET
            `SHARES` = excluded.`SHARES`,
            `DIVIDEND_PER_SHARE` = excluded.`DIVIDEND_PER_SHARE`,
            `AMOUNT` = excluded.`AMOUNT`,
            `CURRENCY` = excluded.`CURRENCY`,
            `AMOUNT_GBP` = excluded.`AMOUNT_GBP`;
        """
        account_upsert = """
        INSERT INTO `CASHACCOUNT` (`DATE`, `CURRENCY`, `DAILY_AMOUNT`, `DAILY_AMOUNT_GBP`, `CUMULATIVE_AMOUNT`, `CUMULATIVE_AMOUNT_GBP`)
        VALUES (:DATE, :CURRENCY, :DAILY_AMOUNT, :DAILY_AMOUNT_GBP, :CUMULATIVE_AMOUNT, :CUMULATIVE_AMOUNT_GBP)
        ON CONFLICT(`DATE`, `CURRENCY`) DO UPDATE SET
            `DAILY_AMOUNT` = excluded.`DAILY_AMOUNT`,
            `DAILY_AMOUNT_GBP` = excluded.`DAILY_AMOUNT_GBP`,
            `CUMULATIVE_AMOUNT` = excluded.`CUMULATIVE_AMOUNT`,
            `CUMULATIVE_AMOUNT_GBP` = excluded.`CUMULATIVE_AMOUNT_GBP`;
        """
    else:
        cf_upsert = """
        INSERT INTO `CASHFLOWS` (`DATE`, `TICKER`, `TYPE`, `SHARES`, `DIVIDEND_PER_SHARE`, `AMOUNT`, `CURRENCY`, `AMOUNT_GBP`)
        VALUES (:DATE, :TICKER, :TYPE, :SHARES, :DIVIDEND_PER_SHARE, :AMOUNT, :CURRENCY, :AMOUNT_GBP)
        ON DUPLICATE KEY UPDATE
            `SHARES` = VALUES(`SHARES`),
            `DIVIDEND_PER_SHARE` = VALUES(`DIVIDEND_PER_SHARE`),
            `AMOUNT` = VALUES(`AMOUNT`),
            `CURRENCY` = VALUES(`CURRENCY`),
            `AMOUNT_GBP` = VALUES(`AMOUNT_GBP`);
        """
        account_upsert = """
        INSERT INTO `CASHACCOUNT` (`DATE`, `CURRENCY`, `DAILY_AMOUNT`, `DAILY_AMOUNT_GBP`, `CUMULATIVE_AMOUNT`, `CUMULATIVE_AMOUNT_GBP`)
        VALUES (:DATE, :CURRENCY, :DAILY_AMOUNT, :DAILY_AMOUNT_GBP, :CUMULATIVE_AMOUNT, :CUMULATIVE_AMOUNT_GBP)
        ON DUPLICATE KEY UPDATE
            `DAILY_AMOUNT` = VALUES(`DAILY_AMOUNT`),
            `DAILY_AMOUNT_GBP` = VALUES(`DAILY_AMOUNT_GBP`),
            `CUMULATIVE_AMOUNT` = VALUES(`CUMULATIVE_AMOUNT`),
            `CUMULATIVE_AMOUNT_GBP` = VALUES(`CUMULATIVE_AMOUNT_GBP`);
        """
    
    latest_date = full_date_range[-1]
    summary_records = [r for r in all_account_records if r["DATE"] == latest_date]
    summary_df = pd.DataFrame(summary_records)
    total_gbp = float(summary_df["CUMULATIVE_AMOUNT_GBP"].sum()) if not summary_df.empty else 0.0
    
    with engine.connect() as conn:
        if cf_records_to_store:
            conn.execute(text(cf_upsert), cf_records_to_store)
        if account_records_to_store:
            conn.execute(text(account_upsert), account_records_to_store)
        conn.commit()
        
    return {
        "records_stored": len(account_records_to_store),
        "cashflows_stored": len(cf_records_to_store),
        "total_gbp": total_gbp,
        "summary_df": summary_df
    }


def calculate_and_store_daily_portfolio_values(backfill_days: int = 0, engine=None):
    """
    Calculates the total portfolio value in GBP for each date by:
    1. Tracking cumulative net holdings per ticker across time from the TRANSACTIONS table.
    2. Joining with daily stock closing prices from ASSET_PRICES.
    3. Applying daily foreign exchange rates to GBP from FX_RATES (with 1.0 for GBP and 0.01 for GBp).
    4. Calculating market value for STOCKS and cumulative cash balance for CASH.
    5. Computing TOTAL_VALUE = STOCKS + CASH and upserting into the PORTFOLIO_VALUES table.
    
    backfill_days:
      0  -> Only calculate and record today / latest valuation (no backfill).
      N  -> Backfill the last N historical days (e.g., 30, 90).
      -1 -> Backfill all available historical dates.
    """
    engine = engine or get_engine()
    create_all_tables(engine)
    
    with engine.connect() as conn:
        prices_df = pd.read_sql(
            text("SELECT `DATE`, `TICKER`, `CLOSE`, `CURRENCY` FROM `ASSET_PRICES` ORDER BY `DATE` ASC"),
            conn
        )
        fx_df = pd.read_sql(
            text("SELECT `DATE`, `FROM_CURRENCY`, `TO_CURRENCY`, `RATE` FROM `FX_RATES` WHERE `TO_CURRENCY` = 'GBP' ORDER BY `DATE` ASC"),
            conn
        )
        tx_df = pd.read_sql(
            text("SELECT `TICKER`, `TRANSACTION_DATE`, `QUANTITY` FROM `TRANSACTIONS` ORDER BY `TRANSACTION_DATE` ASC"),
            conn
        )
        cash_df = pd.read_sql(
            text("SELECT `DATE`, `AMOUNT_GBP` FROM `CASHFLOWS` ORDER BY `DATE` ASC"),
            conn
        )
        
    if prices_df.empty or tx_df.empty:
        return {"records_stored": 0, "summary_df": pd.DataFrame(), "latest_summary": {}}
        
    prices_df["DATE"] = pd.to_datetime(prices_df["DATE"]).dt.strftime("%Y-%m-%d")
    fx_df["DATE"] = pd.to_datetime(fx_df["DATE"]).dt.strftime("%Y-%m-%d")
    tx_df["TRANSACTION_DATE"] = pd.to_datetime(tx_df["TRANSACTION_DATE"]).dt.strftime("%Y-%m-%d")
    if not cash_df.empty:
        cash_df["DATE"] = pd.to_datetime(cash_df["DATE"]).dt.strftime("%Y-%m-%d")
    
    all_fx = fx_df[["DATE", "FROM_CURRENCY", "RATE"]].copy()
    all_dates = pd.DataFrame({"DATE": prices_df["DATE"].unique()})
    gbp_fx = all_dates.copy()
    gbp_fx["FROM_CURRENCY"] = "GBP"
    gbp_fx["RATE"] = 1.0
    all_fx = pd.concat([all_fx, gbp_fx], ignore_index=True).drop_duplicates(subset=["DATE", "FROM_CURRENCY"])
    
    prices_merged = pd.merge(
        prices_df,
        all_fx,
        left_on=["DATE", "CURRENCY"],
        right_on=["DATE", "FROM_CURRENCY"],
        how="left"
    )
    prices_merged.loc[prices_merged["CURRENCY"].isin(["GBp", "GBX", "GBp_PENCE"]) & prices_merged["RATE"].isna(), "RATE"] = 0.01
    prices_merged.loc[(prices_merged["CURRENCY"] == "GBP") & prices_merged["RATE"].isna(), "RATE"] = 1.0
    prices_merged = prices_merged.dropna(subset=["CLOSE"])
    prices_merged["CLOSE_GBP"] = prices_merged["CLOSE"].astype(float) * prices_merged["RATE"].astype(float)
    
    dates_sorted = sorted(prices_df["DATE"].unique())
    tx_pivot = tx_df.pivot_table(index="TRANSACTION_DATE", columns="TICKER", values="QUANTITY", aggfunc="sum").fillna(0)
    
    start_date = min(tx_df["TRANSACTION_DATE"].min(), dates_sorted[0])
    end_date = dates_sorted[-1]
    full_date_range = pd.date_range(start=start_date, end=end_date, freq="D").strftime("%Y-%m-%d")
    tx_grid = tx_pivot.reindex(full_date_range).fillna(0).cumsum()
    
    holdings_long = tx_grid.reset_index().melt(id_vars="index", var_name="TICKER", value_name="SHARES")
    holdings_long.rename(columns={"index": "DATE"}, inplace=True)
    holdings_long = holdings_long[holdings_long["SHARES"] != 0]
    
    # Forward-fill price matrix across all calendar dates to guarantee 100% complete price coverage
    price_grid = prices_merged.pivot_table(index="DATE", columns="TICKER", values="CLOSE_GBP").sort_index().ffill().reindex(full_date_range).ffill()
    prices_ffilled = price_grid.reset_index().melt(id_vars="index", var_name="TICKER", value_name="CLOSE_GBP")
    prices_ffilled.rename(columns={"index": "DATE"}, inplace=True)
    prices_ffilled = prices_ffilled.dropna(subset=["CLOSE_GBP"])

    val_df = pd.merge(prices_ffilled, holdings_long, on=["DATE", "TICKER"], how="inner")
    val_df["POSITION_VALUE_GBP"] = val_df["SHARES"].astype(float) * val_df["CLOSE_GBP"].astype(float)
    
    daily_holdings = val_df.groupby("DATE").agg(
        HOLDINGS_VALUE_GBP=("POSITION_VALUE_GBP", "sum")
    ).reset_index().sort_values("DATE")
    
    if not cash_df.empty:
        daily_cash = cash_df.groupby("DATE").agg(DAILY_CASH_GBP=("AMOUNT_GBP", "sum")).reset_index()
        cal_df = pd.DataFrame({"DATE": full_date_range})
        cal_cash = pd.merge(cal_df, daily_cash, on="DATE", how="left")
        cal_cash["DAILY_CASH_GBP"] = cal_cash["DAILY_CASH_GBP"].fillna(0.0)
        cal_cash["CUMULATIVE_CASH_GBP"] = cal_cash["DAILY_CASH_GBP"].cumsum()
        
        daily_portfolio = pd.merge(daily_holdings, cal_cash[["DATE", "CUMULATIVE_CASH_GBP"]], on="DATE", how="left")
        daily_portfolio["CUMULATIVE_CASH_GBP"] = daily_portfolio["CUMULATIVE_CASH_GBP"].fillna(0.0)
    else:
        daily_portfolio = daily_holdings.copy()
        daily_portfolio["CUMULATIVE_CASH_GBP"] = 0.0
        
    daily_portfolio["STOCKS"] = daily_portfolio["HOLDINGS_VALUE_GBP"].round(2)
    daily_portfolio["CASH"] = daily_portfolio["CUMULATIVE_CASH_GBP"].round(2)
    daily_portfolio["TOTAL_VALUE"] = (daily_portfolio["STOCKS"] + daily_portfolio["CASH"]).round(2)
    daily_portfolio["CURRENCY"] = "GBP"
    
    all_records = [
        {
            "DATE": str(row["DATE"]),
            "TOTAL_VALUE": float(row["TOTAL_VALUE"]),
            "STOCKS": float(row["STOCKS"]),
            "CASH": float(row["CASH"]),
            "CURRENCY": str(row["CURRENCY"]),
        }
        for _, row in daily_portfolio.iterrows()
    ]

    # If backfill_days is 0, only store the latest date record
    if backfill_days == 0:
        records_to_store = all_records[-1:] if all_records else []
    elif backfill_days is not None and backfill_days > 0:
        records_to_store = all_records[-backfill_days:]
    else:  # backfill_days < 0 or None -> backfill all
        records_to_store = all_records
    
    if engine.dialect.name == "sqlite":
        upsert_query = """
        INSERT INTO `PORTFOLIO_VALUES` (`DATE`, `TOTAL_VALUE`, `STOCKS`, `CASH`, `CURRENCY`)
        VALUES (:DATE, :TOTAL_VALUE, :STOCKS, :CASH, :CURRENCY)
        ON CONFLICT(`DATE`, `CURRENCY`) DO UPDATE SET
            `TOTAL_VALUE` = excluded.`TOTAL_VALUE`,
            `STOCKS` = excluded.`STOCKS`,
            `CASH` = excluded.`CASH`,
            `CURRENCY` = excluded.`CURRENCY`;
        """
    else:
        upsert_query = """
        INSERT INTO `PORTFOLIO_VALUES` (`DATE`, `TOTAL_VALUE`, `STOCKS`, `CASH`, `CURRENCY`)
        VALUES (:DATE, :TOTAL_VALUE, :STOCKS, :CASH, :CURRENCY)
        ON DUPLICATE KEY UPDATE
            `TOTAL_VALUE` = VALUES(`TOTAL_VALUE`),
            `STOCKS` = VALUES(`STOCKS`),
            `CASH` = VALUES(`CASH`),
            `CURRENCY` = VALUES(`CURRENCY`);
        """
    
    if records_to_store:
        with engine.connect() as conn:
            conn.execute(text(upsert_query), records_to_store)
            conn.commit()
        
    latest = daily_portfolio.iloc[-1]
    return {
        "records_stored": len(records_to_store),
        "total_historical_records": len(all_records),
        "latest_date": str(latest["DATE"]),
        "latest_total_value": float(latest["TOTAL_VALUE"]),
        "latest_stocks_value": float(latest["STOCKS"]),
        "latest_cash_value": float(latest["CASH"]),
        "currency": str(latest["CURRENCY"]),
        "summary_df": daily_portfolio
    }
    
    if records_to_store:
        with engine.connect() as conn:
            conn.execute(text(upsert_query), records_to_store)
            conn.commit()
        
    latest = daily_portfolio.iloc[-1]
    return {
        "records_stored": len(records_to_store),
        "total_historical_records": len(all_records),
        "latest_date": str(latest["DATE"]),
        "latest_total_value": float(latest["TOTAL_VALUE"]),
        "latest_stocks_value": float(latest["STOCKS"]),
        "latest_cash_value": float(latest["CASH"]),
        "currency": str(latest["CURRENCY"]),
        "summary_df": daily_portfolio
    }


def fetch_historical_prices_gbp(asof_date=None, engine=None) -> pd.DataFrame:
    """
    Retrieves all historical asset prices from ASSET_PRICES, joins with daily FX rates
    from FX_RATES (accounting for 1.0 GBP and 0.01 GBp pence rates), and returns a clean
    date x ticker DataFrame in GBP.
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        prices_df = pd.read_sql(
            text("SELECT `DATE`, `TICKER`, `CLOSE`, `CURRENCY` FROM `ASSET_PRICES` ORDER BY `DATE` ASC"),
            conn
        )
        fx_df = pd.read_sql(
            text("SELECT `DATE`, `FROM_CURRENCY`, `RATE` FROM `FX_RATES` WHERE `TO_CURRENCY` = 'GBP' ORDER BY `DATE` ASC"),
            conn
        )

    if prices_df.empty:
        return pd.DataFrame()

    prices_df["DATE"] = pd.to_datetime(prices_df["DATE"]).dt.strftime("%Y-%m-%d")
    fx_df["DATE"] = pd.to_datetime(fx_df["DATE"]).dt.strftime("%Y-%m-%d")

    all_dates = pd.DataFrame({"DATE": prices_df["DATE"].unique()})
    gbp_fx = all_dates.copy()
    gbp_fx["FROM_CURRENCY"] = "GBP"
    gbp_fx["RATE"] = 1.0
    all_fx = pd.concat([fx_df, gbp_fx], ignore_index=True).drop_duplicates(subset=["DATE", "FROM_CURRENCY"])

    merged = pd.merge(prices_df, all_fx, left_on=["DATE", "CURRENCY"], right_on=["DATE", "FROM_CURRENCY"], how="left")
    merged.loc[merged["CURRENCY"].isin(["GBp", "GBX", "GBp_PENCE"]) & merged["RATE"].isna(), "RATE"] = 0.01
    merged.loc[(merged["CURRENCY"] == "GBP") & merged["RATE"].isna(), "RATE"] = 1.0
    merged["CLOSE_GBP"] = merged["CLOSE"].astype(float) * merged["RATE"].astype(float)

    if asof_date:
        asof_str = pd.to_datetime(asof_date).strftime("%Y-%m-%d")
        merged = merged[merged["DATE"] <= asof_str]

    price_matrix = merged.pivot_table(
        index="DATE",
        columns="TICKER",
        values="CLOSE_GBP"
    ).sort_index().ffill().dropna(how="all")

    return price_matrix


def fetch_portfolio_positions_grid(engine=None):
    """
    Builds the cumulative share holdings grid across all dates available in ASSET_PRICES.
    Does NOT depend on PORTFOLIO_VALUES.
    Returns (holdings_grid_df, all_price_dates_list).
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        tx_df = pd.read_sql(
            text("SELECT `TICKER`, `TRANSACTION_DATE`, `QUANTITY` FROM `TRANSACTIONS` ORDER BY `TRANSACTION_DATE` ASC"),
            conn
        )
        dates_df = pd.read_sql(
            text("SELECT DISTINCT `DATE` FROM `ASSET_PRICES` ORDER BY `DATE` ASC"),
            conn
        )

    if tx_df.empty or dates_df.empty:
        return pd.DataFrame(), []

    tx_df["TRANSACTION_DATE"] = pd.to_datetime(tx_df["TRANSACTION_DATE"]).dt.strftime("%Y-%m-%d")
    all_dates = sorted(pd.to_datetime(dates_df["DATE"]).dt.strftime("%Y-%m-%d").unique())

    tx_pivot = tx_df.pivot_table(index="TRANSACTION_DATE", columns="TICKER", values="QUANTITY", aggfunc="sum").fillna(0)
    all_union_dates = sorted(set(tx_df["TRANSACTION_DATE"]).union(all_dates))
    holdings_grid = tx_pivot.reindex(all_union_dates).fillna(0).cumsum().loc[all_dates]

    return holdings_grid, all_dates


def find_missing_risk_dates(
    min_lookback: int = 260,
    backfill_days: Optional[int] = None,
    required_methods: Optional[List[str]] = None,
    engine=None
):
    """
    Identifies dates with available price data in ASSET_PRICES that do not yet have
    risk records in PORTFOLIO_VAR or PORTFOLIO_RISK_CONTRIBUTIONS for all required models
    (by default: both 'Historical Simulation' and 'Vol-Scaled VaR (EWMA Volatility (λ=0.94))').
    Requires at least min_lookback (default: 260) historical price observations.
    Does NOT depend on PORTFOLIO_VALUES.
    """
    engine = engine or get_engine()
    create_all_tables(engine)
    with engine.connect() as conn:
        price_dates_df = pd.read_sql(
            text("SELECT DISTINCT `DATE` FROM `ASSET_PRICES` ORDER BY `DATE` ASC"),
            conn
        )
        var_df = pd.read_sql(
            text("SELECT DISTINCT `DATE`, `METHOD` FROM `PORTFOLIO_VAR`"),
            conn
        )
        contrib_df = pd.read_sql(
            text("SELECT DISTINCT `DATE`, `METHOD` FROM `PORTFOLIO_RISK_CONTRIBUTIONS`"),
            conn
        )

    if price_dates_df.empty:
        return []

    all_dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in price_dates_df["DATE"]]

    if required_methods is None:
        required_methods = [
            "Historical Simulation",
            "Vol-Scaled VaR (EWMA Volatility (λ=0.94))"
        ]

    # Date sets having complete coverage for all required methods
    complete_var_dates = set(all_dates)
    for m in required_methods:
        m_dates = set(pd.to_datetime(var_df[var_df["METHOD"] == m]["DATE"]).dt.strftime("%Y-%m-%d"))
        complete_var_dates = complete_var_dates.intersection(m_dates)

    complete_contrib_dates = set(all_dates)
    for m in required_methods:
        m_dates = set(pd.to_datetime(contrib_df[contrib_df["METHOD"] == m]["DATE"]).dt.strftime("%Y-%m-%d"))
        complete_contrib_dates = complete_contrib_dates.intersection(m_dates)

    eligible_dates = all_dates[min_lookback:] if len(all_dates) > min_lookback else []
    missing = [
        d for d in eligible_dates
        if d not in complete_var_dates or d not in complete_contrib_dates
    ]
    if backfill_days is not None and backfill_days > 0 and len(missing) > backfill_days:
        missing = missing[-backfill_days:]
    return sorted(missing)


def store_scenario_pnl_records(pnl_records, engine=None):
    """Stores position-level historical scenario P&L records in PORTFOLIO_SCENARIO_PNL."""
    if not pnl_records:
        return 0
    engine = engine or get_engine()
    create_all_tables(engine)

    sanitized = [
        {k: (None if pd.isna(v) else v) for k, v in r.items()}
        for r in pnl_records
    ]
    if engine.dialect.name == "sqlite":
        upsert_sql = """
        INSERT INTO `PORTFOLIO_SCENARIO_PNL` (
            `ASOF_DATE`, `SCENARIO_DATE`, `TICKER`, `METHOD`,
            `SHARES`, `PRICE_GBP`, `POSITION_VALUE_GBP`, `LOG_RETURN`, `SCENARIO_PNL_GBP`
        ) VALUES (
            :ASOF_DATE, :SCENARIO_DATE, :TICKER, :METHOD,
            :SHARES, :PRICE_GBP, :POSITION_VALUE_GBP, :LOG_RETURN, :SCENARIO_PNL_GBP
        ) ON CONFLICT(`ASOF_DATE`, `SCENARIO_DATE`, `TICKER`, `METHOD`) DO UPDATE SET
            `SHARES` = excluded.`SHARES`,
            `PRICE_GBP` = excluded.`PRICE_GBP`,
            `POSITION_VALUE_GBP` = excluded.`POSITION_VALUE_GBP`,
            `LOG_RETURN` = excluded.`LOG_RETURN`,
            `SCENARIO_PNL_GBP` = excluded.`SCENARIO_PNL_GBP`;
        """
    else:
        upsert_sql = """
        INSERT INTO `PORTFOLIO_SCENARIO_PNL` (
            `ASOF_DATE`, `SCENARIO_DATE`, `TICKER`, `METHOD`,
            `SHARES`, `PRICE_GBP`, `POSITION_VALUE_GBP`, `LOG_RETURN`, `SCENARIO_PNL_GBP`
        ) VALUES (
            :ASOF_DATE, :SCENARIO_DATE, :TICKER, :METHOD,
            :SHARES, :PRICE_GBP, :POSITION_VALUE_GBP, :LOG_RETURN, :SCENARIO_PNL_GBP
        ) ON DUPLICATE KEY UPDATE
            `SHARES` = VALUES(`SHARES`),
            `PRICE_GBP` = VALUES(`PRICE_GBP`),
            `POSITION_VALUE_GBP` = VALUES(`POSITION_VALUE_GBP`),
            `LOG_RETURN` = VALUES(`LOG_RETURN`),
            `SCENARIO_PNL_GBP` = VALUES(`SCENARIO_PNL_GBP`);
        """
    with engine.connect() as conn:
        conn.execute(text(upsert_sql), sanitized)
        conn.commit()
    return len(sanitized)


def store_var_records(var_records, engine=None, chunk_size: int = 2000):
    """Stores total portfolio Value-at-Risk and Expected Shortfall records in PORTFOLIO_VAR."""
    if not var_records:
        return 0
    engine = engine or get_engine()
    create_all_tables(engine)

    sanitized = [
        {k: (None if pd.isna(v) else v) for k, v in r.items()}
        for r in var_records
    ]
    if engine.dialect.name == "sqlite":
        upsert_sql = """
        INSERT INTO `PORTFOLIO_VAR` (
            `DATE`, `METHOD`, `CONFIDENCE_LEVEL`, `HORIZON_DAYS`,
            `PORTFOLIO_VALUE_GBP`, `VAR_GBP`, `VAR_PCT`, `CVAR_GBP`, `CVAR_PCT`, `LOOKBACK_OBSERVATIONS`
        ) VALUES (
            :DATE, :METHOD, :CONFIDENCE_LEVEL, :HORIZON_DAYS,
            :PORTFOLIO_VALUE_GBP, :VAR_GBP, :VAR_PCT, :CVAR_GBP, :CVAR_PCT, :LOOKBACK_OBSERVATIONS
        ) ON CONFLICT(`DATE`, `METHOD`, `CONFIDENCE_LEVEL`, `HORIZON_DAYS`) DO UPDATE SET
            `PORTFOLIO_VALUE_GBP` = excluded.`PORTFOLIO_VALUE_GBP`,
            `VAR_GBP` = excluded.`VAR_GBP`,
            `VAR_PCT` = excluded.`VAR_PCT`,
            `CVAR_GBP` = excluded.`CVAR_GBP`,
            `CVAR_PCT` = excluded.`CVAR_PCT`,
            `LOOKBACK_OBSERVATIONS` = excluded.`LOOKBACK_OBSERVATIONS`;
        """
    else:
        upsert_sql = """
        INSERT INTO `PORTFOLIO_VAR` (
            `DATE`, `METHOD`, `CONFIDENCE_LEVEL`, `HORIZON_DAYS`,
            `PORTFOLIO_VALUE_GBP`, `VAR_GBP`, `VAR_PCT`, `CVAR_GBP`, `CVAR_PCT`, `LOOKBACK_OBSERVATIONS`
        ) VALUES (
            :DATE, :METHOD, :CONFIDENCE_LEVEL, :HORIZON_DAYS,
            :PORTFOLIO_VALUE_GBP, :VAR_GBP, :VAR_PCT, :CVAR_GBP, :CVAR_PCT, :LOOKBACK_OBSERVATIONS
        ) ON DUPLICATE KEY UPDATE
            `PORTFOLIO_VALUE_GBP` = VALUES(`PORTFOLIO_VALUE_GBP`),
            `VAR_GBP` = VALUES(`VAR_GBP`),
            `VAR_PCT` = VALUES(`VAR_PCT`),
            `CVAR_GBP` = VALUES(`CVAR_GBP`),
            `CVAR_PCT` = VALUES(`CVAR_PCT`),
            `LOOKBACK_OBSERVATIONS` = VALUES(`LOOKBACK_OBSERVATIONS`);
        """
    with engine.connect() as conn:
        for i in range(0, len(sanitized), chunk_size):
            chunk = sanitized[i:i + chunk_size]
            conn.execute(text(upsert_sql), chunk)
        conn.commit()
    return len(sanitized)


def store_risk_contributions_records(contrib_records, engine=None, chunk_size: int = 2000):
    """Stores position-level Shapley Value risk contributions in PORTFOLIO_RISK_CONTRIBUTIONS."""
    if not contrib_records:
        return 0
    engine = engine or get_engine()
    create_all_tables(engine)

    sanitized = []
    for r in contrib_records:
        pos_val = r.get("POSITION_VALUE_GBP")
        if pos_val is None or pd.isna(pos_val):
            continue
        rec = {k: (None if pd.isna(v) else v) for k, v in r.items()}
        rec["POSITION_VALUE_GBP"] = float(pos_val)
        rec["WEIGHT_PCT"] = float(r.get("WEIGHT_PCT", 0.0)) if not pd.isna(r.get("WEIGHT_PCT")) else 0.0
        rec["SHAPLEY_VAR_GBP"] = float(r.get("SHAPLEY_VAR_GBP", 0.0)) if not pd.isna(r.get("SHAPLEY_VAR_GBP")) else 0.0
        rec["SHAPLEY_VAR_PCT"] = float(r.get("SHAPLEY_VAR_PCT", 0.0)) if not pd.isna(r.get("SHAPLEY_VAR_PCT")) else 0.0
        rec["SHAPLEY_CVAR_GBP"] = float(r.get("SHAPLEY_CVAR_GBP", 0.0)) if not pd.isna(r.get("SHAPLEY_CVAR_GBP")) else 0.0
        rec["SHAPLEY_CVAR_PCT"] = float(r.get("SHAPLEY_CVAR_PCT", 0.0)) if not pd.isna(r.get("SHAPLEY_CVAR_PCT")) else 0.0
        sanitized.append(rec)

    if not sanitized:
        return 0
    if engine.dialect.name == "sqlite":
        upsert_sql = """
        INSERT INTO `PORTFOLIO_RISK_CONTRIBUTIONS` (
            `DATE`, `TICKER`, `METHOD`, `CONFIDENCE_LEVEL`, `HORIZON_DAYS`,
            `POSITION_VALUE_GBP`, `WEIGHT_PCT`,
            `SHAPLEY_VAR_GBP`, `SHAPLEY_VAR_PCT`,
            `SHAPLEY_CVAR_GBP`, `SHAPLEY_CVAR_PCT`,
            `STANDALONE_VAR_GBP`, `DIVERSIFICATION_BENEFIT_GBP`
        ) VALUES (
            :DATE, :TICKER, :METHOD, :CONFIDENCE_LEVEL, :HORIZON_DAYS,
            :POSITION_VALUE_GBP, :WEIGHT_PCT,
            :SHAPLEY_VAR_GBP, :SHAPLEY_VAR_PCT,
            :SHAPLEY_CVAR_GBP, :SHAPLEY_CVAR_PCT,
            :STANDALONE_VAR_GBP, :DIVERSIFICATION_BENEFIT_GBP
        ) ON CONFLICT(`DATE`, `TICKER`, `METHOD`, `CONFIDENCE_LEVEL`, `HORIZON_DAYS`) DO UPDATE SET
            `POSITION_VALUE_GBP` = excluded.`POSITION_VALUE_GBP`,
            `WEIGHT_PCT` = excluded.`WEIGHT_PCT`,
            `SHAPLEY_VAR_GBP` = excluded.`SHAPLEY_VAR_GBP`,
            `SHAPLEY_VAR_PCT` = excluded.`SHAPLEY_VAR_PCT`,
            `SHAPLEY_CVAR_GBP` = excluded.`SHAPLEY_CVAR_GBP`,
            `SHAPLEY_CVAR_PCT` = excluded.`SHAPLEY_CVAR_PCT`,
            `STANDALONE_VAR_GBP` = excluded.`STANDALONE_VAR_GBP`,
            `DIVERSIFICATION_BENEFIT_GBP` = excluded.`DIVERSIFICATION_BENEFIT_GBP`;
        """
    else:
        upsert_sql = """
        INSERT INTO `PORTFOLIO_RISK_CONTRIBUTIONS` (
            `DATE`, `TICKER`, `METHOD`, `CONFIDENCE_LEVEL`, `HORIZON_DAYS`,
            `POSITION_VALUE_GBP`, `WEIGHT_PCT`,
            `SHAPLEY_VAR_GBP`, `SHAPLEY_VAR_PCT`,
            `SHAPLEY_CVAR_GBP`, `SHAPLEY_CVAR_PCT`,
            `STANDALONE_VAR_GBP`, `DIVERSIFICATION_BENEFIT_GBP`
        ) VALUES (
            :DATE, :TICKER, :METHOD, :CONFIDENCE_LEVEL, :HORIZON_DAYS,
            :POSITION_VALUE_GBP, :WEIGHT_PCT,
            :SHAPLEY_VAR_GBP, :SHAPLEY_VAR_PCT,
            :SHAPLEY_CVAR_GBP, :SHAPLEY_CVAR_PCT,
            :STANDALONE_VAR_GBP, :DIVERSIFICATION_BENEFIT_GBP
        ) ON DUPLICATE KEY UPDATE
            `POSITION_VALUE_GBP` = VALUES(`POSITION_VALUE_GBP`),
            `WEIGHT_PCT` = VALUES(`WEIGHT_PCT`),
            `SHAPLEY_VAR_GBP` = VALUES(`SHAPLEY_VAR_GBP`),
            `SHAPLEY_VAR_PCT` = VALUES(`SHAPLEY_VAR_PCT`),
            `SHAPLEY_CVAR_GBP` = VALUES(`SHAPLEY_CVAR_GBP`),
            `SHAPLEY_CVAR_PCT` = VALUES(`SHAPLEY_CVAR_PCT`),
            `STANDALONE_VAR_GBP` = VALUES(`STANDALONE_VAR_GBP`),
            `DIVERSIFICATION_BENEFIT_GBP` = VALUES(`DIVERSIFICATION_BENEFIT_GBP`);
        """
    with engine.connect() as conn:
        for i in range(0, len(sanitized), chunk_size):
            chunk = sanitized[i:i + chunk_size]
            conn.execute(text(upsert_sql), chunk)
        conn.commit()
    return len(sanitized)


def run_full_pipeline(dask_scheduler=None, engine=None):
    """
    Executes the entire data synchronization, backfilling, cash accounting,
    and portfolio valuation pipeline from start to finish.
    """
    engine = engine or get_engine()
    print("1. Ensuring all database tables exist...")
    create_all_tables(engine)
    
    print("2. Fetching portfolio positions from TRANSACTIONS...")
    positions = fetch_portfolio_positions(engine=engine)
    print(f"   Found {len(positions)} active positions: {list(positions.keys())}")
    
    print("3. Ingesting daily stock price and dividend data...")
    price_results = []
    if dask_scheduler:
        from dask.distributed import Client
        import dask
        client = Client(dask_scheduler)
        tasks = [dask.delayed(fetch_and_store_ticker)(ticker, shares, engine) for ticker, shares in positions.items()]
        price_results = dask.compute(*tasks)
        client.close()
    else:
        for ticker, shares in positions.items():
            res = fetch_and_store_ticker(ticker, shares, engine=engine)
            if res:
                price_results.append(res)
    print(f"   Processed {len(price_results)} tickers.")
    
    print("4. Backfilling missing asset price dates...")
    bf_prices = backfill_missing_prices(engine=engine)
    print(f"   Backfilled {bf_prices['rows_backfilled']} price records.")
    
    print("5. Ingesting foreign exchange rates...")
    foreign_currencies = get_foreign_currencies_from_prices(engine=engine)
    print(f"   Foreign currencies detected: {foreign_currencies}")
    fx_results = []
    if dask_scheduler and foreign_currencies:
        from dask.distributed import Client
        import dask
        client = Client(dask_scheduler)
        tasks = [dask.delayed(fetch_and_store_fx_rate)(curr, "GBP", engine) for curr in foreign_currencies]
        fx_results = dask.compute(*tasks)
        client.close()
    else:
        for curr in foreign_currencies:
            res = fetch_and_store_fx_rate(curr, "GBP", engine=engine)
            if res:
                fx_results.append(res)
    print(f"   Processed {len(fx_results)} FX pairs.")
    
    print("6. Backfilling missing FX rate dates...")
    bf_fx = backfill_missing_fx_rates(engine=engine)
    print(f"   Backfilled {bf_fx['rows_backfilled']} FX records.")
    
    print("7. Collecting dividend cashflows and updating CASHACCOUNT...")
    cash_res = collect_and_store_dividend_cashflows_and_cash_account(engine=engine)
    print(f"   Recorded cash balances: {cash_res['total_gbp']:.2f} GBP across {cash_res['records_stored']} currencies.")
    
    print("8. Calculating and storing daily portfolio values...")
    val_res = calculate_and_store_daily_portfolio_values(engine=engine)
    print(f"   Calculated {val_res['records_stored']} daily portfolio values.")
    print(f"   Latest ({val_res['latest_date']}): TOTAL=£{val_res['latest_total_value']:,.2f}, STOCKS=£{val_res['latest_stocks_value']:,.2f}, CASH=£{val_res['latest_cash_value']:,.2f}")
    
    return val_res


PIPELINE_TABLES_TO_CLEAR = [
    "PORTFOLIO_SCENARIO_PNL",
    "PORTFOLIO_RISK_CONTRIBUTIONS",
    "PORTFOLIO_VAR",
    "PORTFOLIO_VALUES",
    "CASHACCOUNT",
    "CASHFLOWS",
    "FX_RATES",
    "ASSET_PRICES",
]


def clear_all_tables_except_transactions(engine=None) -> Dict[str, int]:
    """
    Clears all pipeline-generated tables while strictly preserving the manually-filled TRANSACTIONS table.

    Tables cleared:
      - PORTFOLIO_SCENARIO_PNL
      - PORTFOLIO_RISK_CONTRIBUTIONS
      - PORTFOLIO_VAR
      - PORTFOLIO_VALUES
      - CASHACCOUNT
      - CASHFLOWS
      - FX_RATES
      - ASSET_PRICES

    Returns:
        Dict mapping table names to the number of rows deleted.
    """
    engine = engine or get_engine()
    create_all_tables(engine)

    deleted_counts = {}
    is_sqlite = (engine.dialect.name == "sqlite")

    with engine.begin() as conn:
        if not is_sqlite:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))

        for table in PIPELINE_TABLES_TO_CLEAR:
            try:
                count_res = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`")).scalar()
                deleted_counts[table] = int(count_res or 0)
                conn.execute(text(f"DELETE FROM `{table}`;"))
                if is_sqlite:
                    try:
                        conn.execute(text(f"DELETE FROM sqlite_sequence WHERE name = '{table}';"))
                    except Exception:
                        pass
            except Exception as e:
                deleted_counts[table] = 0

        if not is_sqlite:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))

    return deleted_counts


# Convenience alias
clear_all_pipeline_tables = clear_all_tables_except_transactions


if __name__ == "__main__":
    import sys
    if "--clear-tables" in sys.argv:
        print("Clearing all pipeline tables (preserving TRANSACTIONS)...")
        deleted = clear_all_tables_except_transactions()
        for tbl, cnt in deleted.items():
            print(f"  - {tbl}: {cnt:,} rows deleted")
        print("Done.")
    else:
        run_full_pipeline()



# =====================================================================
# Dashboard Query Methods
# =====================================================================

def fetch_available_tickers(engine: Optional[Engine] = None) -> List[str]:
    """Fetches list of all unique tickers present in the database."""
    eng = engine or get_engine()
    query = """
        SELECT DISTINCT `TICKER`
        FROM `ASSET_PRICES`
        ORDER BY `TICKER` ASC
    """
    with eng.connect() as conn:
        df = pd.read_sql(text(query), conn)
    return df["TICKER"].tolist() if not df.empty else []


def fetch_historical_prices_gbp(
    asof_date: Optional[str] = None,
    engine: Optional[Engine] = None
) -> pd.DataFrame:
    """
    Retrieves all historical asset prices from ASSET_PRICES, joins with daily FX rates
    from FX_RATES (accounting for 1.0 GBP and 0.01 GBp pence rates), and returns a clean
    date x ticker DataFrame in GBP.
    """
    eng = engine or get_engine()
    with eng.connect() as conn:
        prices_df = pd.read_sql(
            text("SELECT `DATE`, `TICKER`, `CLOSE`, `CURRENCY` FROM `ASSET_PRICES` ORDER BY `DATE` ASC"),
            conn
        )
        fx_df = pd.read_sql(
            text("SELECT `DATE`, `FROM_CURRENCY`, `RATE` FROM `FX_RATES` WHERE `TO_CURRENCY` = 'GBP' ORDER BY `DATE` ASC"),
            conn
        )

    if prices_df.empty:
        return pd.DataFrame()

    prices_df["DATE"] = pd.to_datetime(prices_df["DATE"]).dt.strftime("%Y-%m-%d")
    fx_df["DATE"] = pd.to_datetime(fx_df["DATE"]).dt.strftime("%Y-%m-%d")

    all_dates = pd.DataFrame({"DATE": prices_df["DATE"].unique()})
    gbp_fx = all_dates.copy()
    gbp_fx["FROM_CURRENCY"] = "GBP"
    gbp_fx["RATE"] = 1.0
    all_fx = pd.concat([fx_df, gbp_fx], ignore_index=True).drop_duplicates(subset=["DATE", "FROM_CURRENCY"])

    merged = pd.merge(prices_df, all_fx, left_on=["DATE", "CURRENCY"], right_on=["DATE", "FROM_CURRENCY"], how="left")
    merged.loc[merged["CURRENCY"].isin(["GBp", "GBX", "GBp_PENCE", "gbp", "gbx"]) & merged["RATE"].isna(), "RATE"] = 0.01
    merged.loc[(merged["CURRENCY"].isin(["GBP", "gbp"])) & merged["RATE"].isna(), "RATE"] = 1.0
    merged["CLOSE_GBP"] = merged["CLOSE"].astype(float) * merged["RATE"].astype(float)

    if asof_date:
        asof_str = pd.to_datetime(asof_date).strftime("%Y-%m-%d")
        merged = merged[merged["DATE"] <= asof_str]

    price_matrix = merged.pivot_table(
        index="DATE",
        columns="TICKER",
        values="CLOSE_GBP"
    ).sort_index().ffill().dropna(how="all")

    price_matrix.index = pd.to_datetime(price_matrix.index)
    return price_matrix


def fetch_raw_asset_prices(
    ticker_symbol: str,
    asof_date: Optional[str] = None,
    engine: Optional[Engine] = None
) -> pd.DataFrame:
    """
    Fetches full price data (Open, High, Low, Close, Volume, Currency) for a single ticker.
    Also calculates GBP equivalent close price.
    """
    eng = engine or get_engine()
    params: Dict[str, Any] = {"ticker": ticker_symbol}
    date_clause = ""
    if asof_date:
        date_clause = "AND ap.DATE <= :asof"
        params["asof"] = str(asof_date)

    query = f"""
        SELECT 
            ap.DATE,
            ap.TICKER,
            ap.OPEN,
            ap.HIGH,
            ap.LOW,
            ap.CLOSE,
            ap.VOLUME,
            ap.CURRENCY,
            CASE 
                WHEN ap.CURRENCY IN ('GBp', 'GBX', 'GBp_PENCE', 'gbp', 'gbx') 
                    THEN ap.CLOSE / 100.0
                WHEN ap.CURRENCY IN ('GBP', 'gbp') 
                    THEN ap.CLOSE
                WHEN fx.RATE IS NOT NULL AND fx.RATE > 0 
                    THEN ap.CLOSE * fx.RATE
                ELSE ap.CLOSE
            END AS CLOSE_GBP,
            COALESCE(fx.RATE, 1.0) AS FX_RATE_TO_GBP
        FROM ASSET_PRICES ap
        LEFT JOIN FX_RATES fx 
            ON ap.DATE = fx.DATE 
            AND fx.FROM_CURRENCY = ap.CURRENCY 
            AND fx.TO_CURRENCY = 'GBP'
        WHERE ap.TICKER = :ticker
        {date_clause}
        ORDER BY ap.DATE ASC
    """

    with eng.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    if not df.empty:
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.sort_values("DATE").reset_index(drop=True)
    return df




def fetch_portfolio_values_history(
    days: int = 365,
    asof_date: Optional[str] = None,
    engine: Optional[Engine] = None
) -> pd.DataFrame:
    """Fetches historical daily portfolio valuations from PORTFOLIO_VALUES."""
    eng = engine or get_engine()
    params: Dict[str, Any] = {"limit": max(1, days)}
    date_filter = ""
    if asof_date:
        date_filter = "WHERE `DATE` <= :asof"
        params["asof"] = str(asof_date)

    query = f"""
        SELECT `DATE`, `TOTAL_VALUE`, `STOCKS`, `CASH`, `CURRENCY`
        FROM `PORTFOLIO_VALUES`
        {date_filter}
        ORDER BY `DATE` DESC
        LIMIT :limit
    """

    with eng.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    if df.empty:
        return pd.DataFrame(columns=["DATE", "TOTAL_VALUE", "STOCKS", "CASH", "CURRENCY"])

    df["DATE"] = pd.to_datetime(df["DATE"])
    return df.sort_values("DATE").reset_index(drop=True)


def fetch_available_var_dates(engine: Optional[Engine] = None) -> List[str]:
    """Fetches list of distinct dates present in PORTFOLIO_VAR table."""
    eng = engine or get_engine()
    query = """
        SELECT DISTINCT `DATE`
        FROM `PORTFOLIO_VAR`
        ORDER BY `DATE` DESC
    """
    with eng.connect() as conn:
        df = pd.read_sql(text(query), conn)
    if df.empty:
        return []
    return [str(d)[:10] for d in df["DATE"].tolist()]


def fetch_stored_var_metrics(
    asof_date: Optional[str] = None,
    engine: Optional[Engine] = None
) -> pd.DataFrame:
    """
    Fetches all VaR records from PORTFOLIO_VAR for a specific date (or latest).
    Includes all percentiles (0.01 to 0.99) and models.
    """
    eng = engine or get_engine()
    with eng.connect() as conn:
        if not asof_date:
            asof_date = conn.execute(text("SELECT MAX(`DATE`) FROM `PORTFOLIO_VAR`")).scalar()
            if not asof_date:
                return pd.DataFrame()

        query = """
            SELECT `DATE`, `METHOD`, `CONFIDENCE_LEVEL`, `HORIZON_DAYS`,
                   `PORTFOLIO_VALUE_GBP`, `VAR_GBP`, `VAR_PCT`, `CVAR_GBP`, `CVAR_PCT`,
                   `LOOKBACK_OBSERVATIONS`
            FROM `PORTFOLIO_VAR`
            WHERE `DATE` = :d
            ORDER BY `METHOD` ASC, `CONFIDENCE_LEVEL` ASC
        """
        df = pd.read_sql(text(query), conn, params={"d": str(asof_date)})

    if not df.empty:
        df["DATE"] = pd.to_datetime(df["DATE"])
        df["CONFIDENCE_LEVEL"] = df["CONFIDENCE_LEVEL"].astype(float)
    return df


def fetch_stored_risk_contributions(
    asof_date: Optional[str] = None,
    confidence_level: Optional[float] = None,
    method: Optional[str] = None,
    engine: Optional[Engine] = None
) -> pd.DataFrame:
    """
    Fetches component Shapley risk contributions and standalone VaR from PORTFOLIO_RISK_CONTRIBUTIONS.
    """
    eng = engine or get_engine()
    with eng.connect() as conn:
        if not asof_date:
            asof_date = conn.execute(text("SELECT MAX(`DATE`) FROM `PORTFOLIO_RISK_CONTRIBUTIONS`")).scalar()
            if not asof_date:
                return pd.DataFrame()

        filters = ["`DATE` = :d"]
        params: Dict[str, Any] = {"d": str(asof_date)}

        if confidence_level is not None:
            filters.append("`CONFIDENCE_LEVEL` = :cl")
            params["cl"] = float(confidence_level)
        if method:
            filters.append("`METHOD` = :m")
            params["m"] = str(method)

        where_clause = " AND ".join(filters)
        query = f"""
            SELECT `DATE`, `TICKER`, `METHOD`, `CONFIDENCE_LEVEL`, `HORIZON_DAYS`,
                   `POSITION_VALUE_GBP`, `WEIGHT_PCT`, `SHAPLEY_VAR_GBP`, `SHAPLEY_VAR_PCT`,
                   `SHAPLEY_CVAR_GBP`, `SHAPLEY_CVAR_PCT`, `STANDALONE_VAR_GBP`,
                   `DIVERSIFICATION_BENEFIT_GBP`
            FROM `PORTFOLIO_RISK_CONTRIBUTIONS`
            WHERE {where_clause}
            ORDER BY ABS(`SHAPLEY_VAR_GBP`) DESC, `POSITION_VALUE_GBP` DESC
        """
        df = pd.read_sql(text(query), conn, params=params)

    return df


def fetch_stored_scenario_pnl(
    asof_date: Optional[str] = None,
    engine: Optional[Engine] = None
) -> pd.DataFrame:
    """
    Fetches materialized scenario P&L records from PORTFOLIO_SCENARIO_PNL table.
    """
    eng = engine or get_engine()
    with eng.connect() as conn:
        if not asof_date:
            asof_date = conn.execute(text("SELECT MAX(`ASOF_DATE`) FROM `PORTFOLIO_SCENARIO_PNL`")).scalar()
            if not asof_date:
                return pd.DataFrame()

        query = """
            SELECT `SCENARIO_DATE` AS `DATE`,
                   SUM(CASE WHEN `METHOD` = 'Historical Simulation' THEN `SCENARIO_PNL_GBP` ELSE 0.0 END) AS `HISTORICAL_PNL`,
                   SUM(CASE WHEN `METHOD` LIKE 'Vol-Scaled%' OR `METHOD` LIKE '%EWMA%' THEN `SCENARIO_PNL_GBP` ELSE 0.0 END) AS `VOL_SCALED_EWMA_PNL`
            FROM `PORTFOLIO_SCENARIO_PNL`
            WHERE `ASOF_DATE` = :d
            GROUP BY `SCENARIO_DATE`
            ORDER BY `SCENARIO_DATE` ASC
        """
        df = pd.read_sql(text(query), conn, params={"d": str(asof_date)})

    if not df.empty:
        df["DATE"] = pd.to_datetime(df["DATE"])
    return df
