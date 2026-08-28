"""
Dagster repository and pipeline orchestration for Portfolio-Analytics-Platform.
Defines assets, jobs, and daily schedules, consuming database interaction
functions from the standalone db module.
"""

from typing import Optional
from dagster import (
    asset,
    Config,
    ConfigurableResource,
    EnvVar,
    Definitions,
    get_dagster_logger,
    Output,
    MetadataValue,
    ScheduleDefinition,
    define_asset_job,
)
from dask.distributed import Client
import dask
import pandas as pd

from portfolio_core.db import (
    get_engine,
    get_connection_string,
    fetch_all_historical_tickers,
    fetch_portfolio_positions,
    fetch_and_store_ticker,
    backfill_missing_prices as db_backfill_missing_prices,
    get_foreign_currencies_from_prices,
    fetch_and_store_fx_rate,
    backfill_missing_fx_rates as db_backfill_missing_fx_rates,
    collect_and_store_dividend_cashflows_and_cash_account,
    calculate_and_store_daily_portfolio_values,
    calculate_and_store_daily_benchmark_values,
    generate_and_store_benchmark_transactions,
    fetch_benchmarks_info,
    fetch_benchmark_values_history,
    fetch_historical_prices_gbp,
    fetch_portfolio_positions_grid,
    find_missing_risk_dates,
    store_scenario_pnl_records,
    store_var_records,
    store_risk_contributions_records,
)
from portfolio_core.analytics.var import (
    HistoricalVaR,
    VolatilityScaledVaR,
    EWMAVolatility,
    SampleVolatility,
    ShapleyRiskAttributor,
    evaluate_portfolio_risk_models,
    compute_historical_risk_timeline,
)
from reporting import (
    fetch_recent_portfolio_values,
    fetch_recent_var_metrics,
    fetch_dividends_for_date,
    fetch_top_position_movers,
    generate_portfolio_report_chart,
    format_telegram_caption,
    send_telegram_report,
)

import os
import yaml

DASK_SCHEDULER_URI = os.getenv("DASK_SCHEDULER_URI", "tcp://dask-scheduler:8786")
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")


def get_dask_client(logger=None, timeout: str = "4s") -> Client:
    """
    Connects to the distributed Dask scheduler if available.
    If the remote scheduler is unreachable, seamlessly creates a local threaded Dask Client.
    """
    uri = os.getenv("DASK_SCHEDULER_URI", DASK_SCHEDULER_URI)
    try:
        client = Client(uri, timeout=timeout)
        if logger:
            logger.info(f"Connected to distributed Dask scheduler at {uri}")
        return client
    except Exception as e:
        if logger:
            logger.warning(f"Could not connect to Dask scheduler at {uri} ({e}). Falling back to local Dask client.")
        return Client(processes=False)


def load_pipeline_config(config_file: str = CONFIG_FILE) -> dict:
    """
    Loads and validates configuration from config.yaml.
    Supports both 'mariadb' and local 'sqlite' (.s3db) configurations.
    Raises FileNotFoundError or ValueError if configuration is missing or invalid.
    No hardcoded defaults are kept in code.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Configuration file '{config_file}' not found. "
            "Please ensure config.yaml is present with resources.db.config and ops settings."
        )
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f) or {}
    
    db_cfg = cfg.get("resources", {}).get("db", {}).get("config", {})
    if not db_cfg:
        raise ValueError(f"Missing 'resources.db.config' section in '{config_file}'.")
        
    db_type = str(db_cfg.get("type", "mariadb")).lower()
    if db_type == "sqlite":
        sqlite_path = db_cfg.get("sqlite_path")
        if not sqlite_path:
            raise ValueError(f"Missing 'sqlite_path' under resources.db.config in '{config_file}'.")
        if not str(sqlite_path).endswith(".s3db"):
            raise ValueError(
                f"SQLite database file path must end with '.s3db' (got '{sqlite_path}'). "
                "Ensure your local SQLite file uses the .s3db extension so it is gitignored."
            )
    elif db_type in ["mariadb", "mysql"]:
        for field in ["host", "port", "user", "database"]:
            if field not in db_cfg or db_cfg[field] is None:
                raise ValueError(
                    f"Missing required parameter '{field}' under resources.db.config in '{config_file}'."
                )
    else:
        raise ValueError(f"Unsupported database type '{db_type}' in '{config_file}'. Supported: 'mariadb', 'sqlite'.")

    return cfg


_app_cfg = load_pipeline_config()
_db_cfg = _app_cfg["resources"]["db"]["config"]
_db_type = str(_db_cfg.get("type", "mariadb")).lower()


class DatabaseResource(ConfigurableResource):
    """
    Dagster Configurable Resource for database connections.
    Supports both 'mariadb' and local SQLite (.s3db) backends loaded from config.yaml.
    Secret credentials (password) are loaded securely from the DB_PASSWORD environment variable for MariaDB.
    No defaults are hardcoded in the Python code.
    """
    type: str = "mariadb"
    # MariaDB fields (optional when type='sqlite')
    user: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    password: Optional[str] = None

    # SQLite fields (optional when type='mariadb')
    sqlite_path: Optional[str] = None

    def get_connection_string(self) -> str:
        return get_connection_string(
            db_type=self.type,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
            sqlite_path=self.sqlite_path
        )

    def get_engine(self):
        return get_engine(self.get_connection_string())


class StockDataConfig(Config):
    """
    Configuration schema for the portfolio_stock_data asset.
    - history_days: Number of historical trading days to maintain for all tickers ever seen in TRANSACTIONS (default: 520).
    """
    history_days: int = 520


@asset
def portfolio_stock_data(config: StockDataConfig, db: DatabaseResource):
    """Fetches daily stock prices and dividend data for ALL tickers ever seen in TRANSACTIONS with continuous backfilling."""
    logger = get_dagster_logger()
    engine = db.get_engine()
    conn_str = db.get_connection_string()
    
    logger.info("Loading all distinct tickers ever recorded in TRANSACTIONS...")
    all_tickers = fetch_all_historical_tickers(engine=engine)
    if not all_tickers:
        raise ValueError("No tickers found in the transactions table.")
        
    current_positions = fetch_portfolio_positions(engine=engine)
    logger.info(f"Discovered {len(all_tickers)} historical tickers across all transactions: {all_tickers}")
    logger.info(f"Ensuring at least {config.history_days} historical trading days of continuous price history...")
    
    logger.info("Connecting to Dask cluster...")
    client = get_dask_client(logger=logger)
    
    delayed_tasks = []
    for ticker in all_tickers:
        shares = current_positions.get(ticker, 0.0)
        task = dask.delayed(fetch_and_store_ticker)(ticker, shares, config.history_days, conn_str)
        delayed_tasks.append(task)
        
    logger.info(f"Concurrently fetching/backfilling incremental data for {len(delayed_tasks)} tickers...")
    results = dask.compute(*delayed_tasks)
    client.close()
    
    summary = [r for r in results if r is not None]
    if not summary:
        raise ValueError("No valid stock data could be fetched for any ticker.")
        
    summary_df = pd.DataFrame(summary)
    data_preview = summary_df.to_markdown(index=False)
    total_written = int(summary_df["Rows Written"].sum())
    
    return Output(
        value=summary_df,
        metadata={
            "Status": "Stock price ingestion and historical backfill complete",
            "Tickers Tracked (All Historical)": len(summary_df),
            "History Days Target": config.history_days,
            "New Rows Written": total_written,
            "Summary": MetadataValue.md(data_preview)
        }
    )


@asset(deps=[portfolio_stock_data])
def backfill_missing_prices(db: DatabaseResource):
    """Forward-fills missing dates across tickers in ASSET_PRICES to ensure a uniform price grid."""
    logger = get_dagster_logger()
    logger.info("Starting price backfill for missing dates across tickers...")
    
    res = db_backfill_missing_prices(engine=db.get_engine())
    total_backfilled = res["rows_backfilled"]
    summary_df = res["summary_df"]
    preview = summary_df.to_markdown(index=False)
    
    logger.info(f"Price backfill complete. Inserted {total_backfilled} rows.")
    
    return Output(
        value=total_backfilled,
        metadata={
            "Status": "Price backfill complete",
            "Total Rows Backfilled": total_backfilled,
            "Summary": MetadataValue.md(preview)
        }
    )


@asset(deps=[backfill_missing_prices])
def portfolio_fx_rates(db: DatabaseResource):
    """Fetches foreign exchange rates for non-GBP currencies against GBP and stores in FX_RATES."""
    logger = get_dagster_logger()
    logger.info("Discovering foreign currencies from ASSET_PRICES table...")
    
    engine = db.get_engine()
    conn_str = db.get_connection_string()
    currencies = get_foreign_currencies_from_prices(engine=engine)
    if not currencies:
        logger.info("No foreign currencies found in ASSET_PRICES to download FX rates for.")
        return Output(value=None, metadata={"Status": "No foreign currencies found in ASSET_PRICES"})
        
    logger.info(f"Foreign currencies to fetch FX rates against GBP: {currencies}")
    
    logger.info("Connecting to Dask cluster...")
    client = get_dask_client(logger=logger)
    
    delayed_tasks = []
    for curr in currencies:
        task = dask.delayed(fetch_and_store_fx_rate)(curr, "GBP", conn_str)
        delayed_tasks.append(task)
        
    logger.info(f"Concurrently fetching FX rates for {len(delayed_tasks)} currencies...")
    results = dask.compute(*delayed_tasks)
    client.close()
    
    summary = [r for r in results if r is not None]
    if not summary:
        logger.info("No FX rate tasks produced new rows.")
        summary_df = pd.DataFrame([{"Status": "All foreign FX rates up to date"}])
    else:
        summary_df = pd.DataFrame(summary)
        
    preview = summary_df.to_markdown(index=False)
    
    return Output(
        value=summary_df,
        metadata={
            "Status": "FX rates saved to remote MariaDB FX_RATES table",
            "Summary": MetadataValue.md(preview)
        }
    )


@asset(deps=[portfolio_fx_rates])
def backfill_missing_fx_rates(db: DatabaseResource):
    """Forward-fills missing FX dates in FX_RATES aligned with the master asset price date timeline."""
    logger = get_dagster_logger()
    logger.info("Starting FX rate backfill for missing dates...")
    
    res = db_backfill_missing_fx_rates(engine=db.get_engine())
    total_backfilled = res["rows_backfilled"]
    summary_df = res["summary_df"]
    preview = summary_df.to_markdown(index=False)
    
    logger.info(f"FX backfill complete. Inserted {total_backfilled} rows.")
    
    return Output(
        value=total_backfilled,
        metadata={
            "Status": "FX backfill complete",
            "Total Rows Backfilled": total_backfilled,
            "Summary": MetadataValue.md(preview)
        }
    )


@asset(deps=[backfill_missing_prices, backfill_missing_fx_rates])
def cash_account(db: DatabaseResource):
    """Identifies dividend payments on held positions, logs to CASHFLOWS, and maintains daily historized CASHACCOUNT balances."""
    logger = get_dagster_logger()
    logger.info("Maintaining historized daily dividend cashflows and CASHACCOUNT balances...")
    
    res = collect_and_store_dividend_cashflows_and_cash_account(engine=db.get_engine())
    records_stored = res["records_stored"]
    total_gbp = res["total_gbp"]
    
    if records_stored == 0:
        return Output(value=0, metadata={"Status": "No cashflow balances to record in CASHACCOUNT"})
        
    summary_df = res["summary_df"]
    preview = summary_df.to_markdown(index=False)
    
    logger.info(f"Updated CASHACCOUNT with {records_stored} historical records. Current cash balance: £{total_gbp:,.2f}.")
    
    return Output(
        value=records_stored,
        metadata={
            "Status": "Daily historized cash balances stored in remote MariaDB CASHACCOUNT table",
            "Records Stored / Updated": records_stored,
            "Total Cash Balance (GBP)": f"£{total_gbp:,.2f}",
            "Latest CASHACCOUNT Balances": MetadataValue.md(preview)
        }
    )


class PortfolioValuesConfig(Config):
    """
    Configuration schema for the portfolio_daily_values asset.
    - backfill_days:
        0  -> (Default) No backfilling; only calculate and store today/latest valuation.
        N  -> Backfills portfolio values and cash account for the last N historical days (e.g. 30, 90).
        -1 -> Backfills all available historical dates.
    """
    backfill_days: int = 0


@asset(deps=[cash_account])
def portfolio_daily_values(config: PortfolioValuesConfig, db: DatabaseResource):
    """Calculates daily total portfolio valuation in GBP broken down into TOTAL_VALUE, STOCKS, and CASH (with cash backfill)."""
    logger = get_dagster_logger()
    logger.info(f"Starting daily portfolio valuation & cash backfill (backfill_days={config.backfill_days})...")
    engine = db.get_engine()

    # Synchronize dividend cashflows for the same backfill_days
    if config.backfill_days != 0:
        collect_and_store_dividend_cashflows_and_cash_account(
            backfill_days=config.backfill_days,
            engine=engine
        )
    
    res = calculate_and_store_daily_portfolio_values(
        backfill_days=config.backfill_days,
        engine=engine
    )
    records_stored = res["records_stored"]

    # Calculate shadow transactions and daily benchmark valuations
    try:
        bm_res = calculate_and_store_daily_benchmark_values(engine=engine)
        bm_stored = bm_res.get("records_stored", 0)
        logger.info(f"Calculated and stored {bm_stored} benchmark daily valuation records across active benchmarks.")
    except Exception as e:
        logger.warning(f"Failed to calculate benchmark daily values: {e}")
        bm_stored = 0
    
    if records_stored == 0:
        return Output(value=0, metadata={"Status": "No records to calculate"})
        
    summary_df = res["summary_df"][["DATE", "TOTAL_VALUE", "STOCKS", "CASH", "CURRENCY"]]
    preview = summary_df.tail(10).to_markdown(index=False)
    
    latest_tot = res["latest_total_value"]
    latest_stk = res["latest_stocks_value"]
    latest_csh = res["latest_cash_value"]
    latest_date = res["latest_date"]
    currency = res["currency"]
    
    logger.info(
        f"Calculated and saved {records_stored} daily portfolio values ({bm_stored} benchmark valuations). "
        f"Latest ({latest_date}): TOTAL=£{latest_tot:,.2f}, STOCKS=£{latest_stk:,.2f}, CASH=£{latest_csh:,.2f}"
    )
    
    return Output(
        value=records_stored,
        metadata={
            "Status": "Daily portfolio values stored in remote MariaDB PORTFOLIO_VALUES table",
            "Records Stored": records_stored,
            "Benchmark Valuations Stored": bm_stored,
            "Backfill Days": config.backfill_days,
            "Latest Valuation Date": latest_date,
            "Total Portfolio Value": f"£{latest_tot:,.2f} {currency}",
            "Stock Holdings Value": f"£{latest_stk:,.2f} {currency}",
            "Cash Balance": f"£{latest_csh:,.2f} {currency}",
            "Currency": currency,
            "Recent Valuations Preview": MetadataValue.md(preview)
        }
    )


class RiskConfig(Config):
    """
    Configuration schema for the portfolio_value_at_risk asset.
    - backfill_days:
        0  -> (Default) No backfilling; only calculate risk metrics for today.
        N  -> Backfills the last N missing historical dates (e.g. 10, 30).
        -1 -> Backfills all missing historical dates.
    - min_lookback_days: Minimum historical days of return history required (default: 260).
    - lookback_days: Lookback window of historical return observations (default: 260).
    - num_permutations: Permutations for Shapley risk attribution (default: 100).
    - var_min_percentile: Lower percentile bound for VaR calculation (e.g. 0.01).
    - var_max_percentile: Upper percentile bound for VaR calculation (e.g. 0.99).
    - shapley_min_percentile: Lower percentile bound for Shapley risk attribution (e.g. 0.95).
    - shapley_max_percentile: Upper percentile bound for Shapley risk attribution (e.g. 0.95).
    """
    backfill_days: int = 0
    min_lookback_days: int = 260
    lookback_days: int = 260
    num_permutations: int = 100
    var_min_percentile: float = 0.01
    var_max_percentile: float = 0.99
    shapley_min_percentile: float = 0.99
    shapley_max_percentile: float = 0.99


def run_risk_pipeline(
    backfill_days: int = 0,
    min_lookback: int = 260,
    lookback_days: int = 260,
    num_permutations: int = 100,
    var_min_percentile: float = 0.01,
    var_max_percentile: float = 0.99,
    shapley_min_percentile: float = 0.99,
    shapley_max_percentile: float = 0.99,
    asof_date=None,
    engine=None
) -> dict:
    """
    Orchestrates the entire risk pipeline:
    1. If backfill_days != 0, backfills missing historical dates (independent of PORTFOLIO_VALUES).
    2. Ingests current prices & positions from db.
    3. Evaluates VaR models across var_percentiles & Shapley contributions across shapley_percentiles.
    4. Persists scenario P&L vectors, VaR metrics, and Shapley contributions into MariaDB.
    """
    engine = engine or get_engine()
    backfill_stats = None
    if backfill_days != 0:
        limit = None if backfill_days < 0 else backfill_days
        missing_dates = find_missing_risk_dates(
            min_lookback=min_lookback,
            backfill_days=limit,
            engine=engine
        )
        if missing_dates:
            prices_gbp = fetch_historical_prices_gbp(engine=engine)
            holdings_grid, _ = fetch_portfolio_positions_grid(engine=engine)
            var_rows, contrib_rows = compute_historical_risk_timeline(
                prices_gbp=prices_gbp,
                holdings_grid=holdings_grid,
                dates_to_compute=missing_dates,
                min_lookback=min_lookback,
                lookback_days=lookback_days,
                num_permutations=num_permutations,
                var_min_percentile=var_min_percentile,
                var_max_percentile=var_max_percentile,
                shapley_min_percentile=shapley_min_percentile,
                shapley_max_percentile=shapley_max_percentile,
            )
            store_var_records(var_rows, engine=engine)
            store_risk_contributions_records(contrib_rows, engine=engine)
            backfill_stats = {
                "missing_dates_count": len(missing_dates),
                "var_records_inserted": len(var_rows),
                "contrib_records_inserted": len(contrib_rows)
            }

    # Current / specific date risk evaluation
    positions = fetch_portfolio_positions(asof=asof_date, engine=engine)
    prices_gbp = fetch_historical_prices_gbp(asof_date=asof_date, engine=engine)
    if not positions or prices_gbp.empty:
        raise ValueError("Cannot calculate risk: positions or price history is empty.")

    attributor = ShapleyRiskAttributor(num_permutations=num_permutations)
    results, contrib_df, scenario_records = evaluate_portfolio_risk_models(
        positions=positions,
        prices_gbp=prices_gbp,
        asof_date=asof_date,
        models=None,
        attributor=attributor,
        var_min_percentile=var_min_percentile,
        var_max_percentile=var_max_percentile,
        shapley_min_percentile=shapley_min_percentile,
        shapley_max_percentile=shapley_max_percentile,
        lookback_days=lookback_days,
    )

    # Persist scenario records
    store_scenario_pnl_records(scenario_records, engine=engine)
    
    var_records = [
        {
            "DATE": r.asof_date,
            "METHOD": r.model_name,
            "CONFIDENCE_LEVEL": float(r.confidence_level),
            "HORIZON_DAYS": int(r.horizon_days),
            "PORTFOLIO_VALUE_GBP": float(r.portfolio_value_gbp),
            "VAR_GBP": float(r.var_gbp),
            "VAR_PCT": float(r.var_pct),
            "CVAR_GBP": float(r.cvar_gbp) if r.cvar_gbp is not None else None,
            "CVAR_PCT": float(r.cvar_pct) if r.cvar_pct is not None else None,
            "LOOKBACK_OBSERVATIONS": int(r.lookback_observations) if r.lookback_observations else None,
        }
        for r in results
    ]
    store_var_records(var_records, engine=engine)

    if contrib_df is not None and not contrib_df.empty:
        store_risk_contributions_records(contrib_df.to_dict(orient="records"), engine=engine)

    summary_df = pd.DataFrame([r.to_dict() for r in results])
    return {
        "asof_date": results[0].asof_date,
        "results": results,
        "summary_df": summary_df,
        "contributions_df": contrib_df,
        "backfill_stats": backfill_stats
    }


@asset(deps=[portfolio_daily_values])
def portfolio_value_at_risk(config: RiskConfig, db: DatabaseResource):
    """Calculates Value-at-Risk and Shapley risk contributions across configured percentile ranges and stores in PORTFOLIO_VAR."""
    logger = get_dagster_logger()
    logger.info(
        f"Computing daily Value-at-Risk (backfill_days={config.backfill_days}, "
        f"lookback_days={config.lookback_days}, "
        f"VaR percentiles [{config.var_min_percentile}..{config.var_max_percentile}], "
        f"Shapley percentiles [{config.shapley_min_percentile}..{config.shapley_max_percentile}])..."
    )
    
    res = run_risk_pipeline(
        backfill_days=config.backfill_days,
        min_lookback=config.min_lookback_days,
        lookback_days=config.lookback_days,
        num_permutations=config.num_permutations,
        var_min_percentile=config.var_min_percentile,
        var_max_percentile=config.var_max_percentile,
        shapley_min_percentile=config.shapley_min_percentile,
        shapley_max_percentile=config.shapley_max_percentile,
        engine=db.get_engine()
    )
    summary_df = res["summary_df"]
    asof = res["asof_date"]
    
    # Highlight key percentiles for preview (e.g. 0.90, 0.95, 0.99)
    key_cls = ["90.0%", "95.0%", "97.5%", "99.0%"]
    preview_df = summary_df[summary_df["Confidence Level"].isin(key_cls)] if "Confidence Level" in summary_df.columns else summary_df.tail(8)
    if preview_df.empty:
        preview_df = summary_df.tail(8)
    preview = preview_df.to_markdown(index=False)
    
    logger.info(f"Value-at-Risk calculated for {asof} ({len(summary_df)} percentile models evaluated).")
    
    metadata = {
        "Status": "VaR calculated and stored in remote MariaDB PORTFOLIO_VAR table",
        "Valuation Date": asof,
        "Backfill Days": config.backfill_days,
        "VaR Percentile Range": f"{config.var_min_percentile:.2f} - {config.var_max_percentile:.2f}",
        "Shapley Percentile Range": f"{config.shapley_min_percentile:.2f} - {config.shapley_max_percentile:.2f}",
        "Risk Models Evaluated": len(summary_df),
        "Key Percentiles Preview": MetadataValue.md(preview)
    }
    if res.get("backfill_stats"):
        metadata["Backfill Stats"] = str(res["backfill_stats"])
        
    return Output(
        value=len(summary_df),
        metadata=metadata
    )


class ReportingConfig(Config):
    """
    Configuration schema for the portfolio_telegram_report asset.
    - enabled: Set to True to generate and deliver the Telegram report (default: False).
    - recipients: List of Telegram chat IDs / user IDs / group channel IDs to receive the report.
    - bot_token: Optional Telegram Bot Token (reads TELEGRAM_BOT_TOKEN env var if omitted).
    - asof_date: Optional historical calculation date (YYYY-MM-DD) to generate the report as of.
    - portfolio_history_days: Number of historical days to plot on the valuation chart (default: 14).
    - var_history_days: Number of historical days to plot on the VaR chart (default: 130).
    - confidence_levels: Confidence levels to include in the VaR chart (default: [0.01, 0.05, 0.95, 0.99]).
    - top_risk_contributors_n: Number of top Vol-Scaled VaR risk contributors to include in the table (default: 5).
    - top_movers_n: Number of top daily position value movers (|ΔValue|) to include in caption (default: 5).
    """
    enabled: bool = False
    recipients: list[str] = []
    bot_token: Optional[str] = None
    asof_date: Optional[str] = None
    portfolio_history_days: int = 14
    var_history_days: int = 130
    confidence_levels: list[float] = [0.01, 0.05, 0.95, 0.99]
    top_risk_contributors_n: int = 5
    top_movers_n: int = 5


@asset(deps=[portfolio_value_at_risk])
def portfolio_telegram_report(config: ReportingConfig, db: DatabaseResource):
    """
    Final pipeline step: Generates dual-panel visual charts of recent portfolio values
    and Value-at-Risk metrics, delivering them with a summary caption to configured Telegram recipients.
    Only executes delivery when enabled in configuration.
    """
    logger = get_dagster_logger()
    if not config.enabled:
        logger.info("Portfolio Telegram report is disabled in configuration. Skipping.")
        return Output(
            value="Disabled",
            metadata={
                "Status": "Reporting disabled (set enabled: true under ops.portfolio_telegram_report.config in config.yaml to activate)",
                "Enabled": False
            }
        )

    recipients = [str(r).strip() for r in config.recipients if str(r).strip()]
    if not recipients:
        logger.warning("Reporting is enabled, but no Telegram recipients were configured. Skipping delivery.")
        return Output(
            value="No Recipients",
            metadata={
                "Status": "Skipped: No recipients configured under ops.portfolio_telegram_report.config.recipients",
                "Enabled": True,
                "Recipients Count": 0
            }
        )

    token = config.bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.warning("Reporting is enabled, but TELEGRAM_BOT_TOKEN environment variable is not set. Skipping delivery.")
        return Output(
            value="Missing Token",
            metadata={
                "Status": "Skipped: TELEGRAM_BOT_TOKEN not found in environment or configuration",
                "Enabled": True,
                "Recipients": recipients
            }
        )

    logger.info(f"Generating visual portfolio & risk report for {len(recipients)} Telegram recipient(s)...")
    engine = db.get_engine()
    
    result = send_telegram_report(
        recipients=recipients,
        token=token,
        asof_date=config.asof_date,
        portfolio_days=config.portfolio_history_days,
        var_days=config.var_history_days,
        confidence_levels=config.confidence_levels,
        top_risk_contributors_n=config.top_risk_contributors_n,
        top_movers_n=config.top_movers_n,
        engine=engine,
        output_chart_path="/tmp/portfolio_report_latest.png"
    )

    delivered_cnt = len(result["delivered_recipients"])
    failed_cnt = len(result["failed_recipients"])
    logger.info(f"Telegram report broadcast complete: {delivered_cnt} delivered, {failed_cnt} failed.")
    
    clean_md_caption = (
        result["caption"]
        .replace("<b>", "**")
        .replace("</b>", "**")
        .replace("<i>", "*")
        .replace("</i>", "*")
    )
    
    return Output(
        value=result,
        metadata={
            "Status": result["status"],
            "Recipients Total": result["recipients_total"],
            "Delivered Count": delivered_cnt,
            "Failed Count": failed_cnt,
            "Dividends Paid Today": f"£{result.get('dividends_total_gbp', 0.0):,.2f}",
            "Summary Caption": MetadataValue.md(clean_md_caption),
            "Chart Size (Bytes)": result["chart_size_bytes"]
        }
    )


_ops_cfg = _app_cfg.get("ops", {})

portfolio_job = define_asset_job(
    name="portfolio_job",
    selection=[
        portfolio_stock_data,
        backfill_missing_prices,
        portfolio_fx_rates,
        backfill_missing_fx_rates,
        cash_account,
        portfolio_daily_values,
        portfolio_value_at_risk,
        portfolio_telegram_report,
    ],
    config={"ops": _ops_cfg} if _ops_cfg else None,
)

daily_portfolio_schedule = ScheduleDefinition(
    name="daily_portfolio_schedule",
    job=portfolio_job,
    cron_schedule="30 21 * * 1-5",
    execution_timezone="Europe/London",
)

if _db_type == "sqlite":
    _db_resource = DatabaseResource(
        type="sqlite",
        sqlite_path=str(_db_cfg["sqlite_path"]),
    )
else:
    _db_resource = DatabaseResource(
        type="mariadb",
        user=str(_db_cfg["user"]),
        host=str(_db_cfg["host"]),
        port=int(_db_cfg["port"]),
        database=str(_db_cfg["database"]),
        password=EnvVar("DB_PASSWORD"),
    )

defs = Definitions(
    assets=[
        portfolio_stock_data,
        backfill_missing_prices,
        portfolio_fx_rates,
        backfill_missing_fx_rates,
        cash_account,
        portfolio_daily_values,
        portfolio_value_at_risk,
        portfolio_telegram_report,
    ],
    jobs=[portfolio_job],
    schedules=[daily_portfolio_schedule],
    resources={
        "db": _db_resource
    }
)
