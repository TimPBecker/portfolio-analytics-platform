"""
Tab 8: Market Data Admin & Universe Overview View.
Provides:
  1. An overview table of all assets (market data coverage, first/last date, records, currency, source, status).
     Includes assets from portfolio trades and benchmarks that lack market data.
  2. A form to add additional time series for specified start and end dates
     (pre-populated with 1 Jul 2024 and the last processed date).
  3. A button to refresh market data, where market data tables (ASSET_PRICES, FX_RATES)
     are cleared and repopulated across all assets, trades, and benchmark constituents.
"""

from datetime import date, datetime
from typing import Optional, Dict, Any, List
import pandas as pd
import streamlit as st
from sqlalchemy.engine import Engine

from portfolio_core.db import (
    get_engine,
    get_latest_processed_date,
    fetch_asset_market_data_overview,
    ingest_custom_asset_time_series,
    refresh_all_market_data
)

try:
    from src.ui.theme import PALETTE
except ImportError:
    from apps.dashboard.src.ui.theme import PALETTE


def render_tab_market_data(
    engine: Optional[Engine] = None,
    db_name: Optional[str] = None,
    asof_date: Optional[str] = None
):
    """Renders the Market Data Admin and Universe Overview interface."""
    st.markdown("### 📡 Market Data Administration & Universe Overview")
    st.caption(
        "Manage market data coverage across portfolio positions, benchmark constituents, and custom time series. "
        "Monitor historical coverage windows, ingest new asset histories, or refresh the entire market data universe."
    )

    eng = engine or get_engine(database=db_name)

    # Resolve latest processed date for prepopulation defaults
    latest_proc = get_latest_processed_date(engine=eng) or asof_date
    default_end_date = date.today()
    if latest_proc:
        try:
            default_end_date = datetime.strptime(str(latest_proc)[:10], "%Y-%m-%d").date()
        except Exception:
            default_end_date = date.today()

    default_start_date = date(2024, 7, 1)  # 1 Jul 2024 as requested

    # -------------------------------------------------------------------------
    # 1. Fetch Universe Overview Data
    # -------------------------------------------------------------------------
    overview_df = fetch_asset_market_data_overview(engine=eng)

    total_assets = len(overview_df)
    avail_count = int((overview_df["STATUS"] == "Available").sum()) if not overview_df.empty else 0
    missing_count = total_assets - avail_count

    # Determine earliest and latest dates across available assets
    earliest_date_str = "-"
    latest_date_str = "-"
    if not overview_df.empty:
        valid_first = overview_df["FIRST_DATE"].dropna()
        valid_last = overview_df["LAST_DATE"].dropna()
        if not valid_first.empty:
            earliest_date_str = str(valid_first.min())
        if not valid_last.empty:
            latest_date_str = str(valid_last.max())

    # Summary Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Tracked Assets", f"{total_assets} tickers")
    with m_col2:
        st.metric("Market Data Available", f"{avail_count} assets", delta=f"{avail_count}/{total_assets}" if total_assets else "0")
    with m_col3:
        st.metric("Missing Market Data", f"{missing_count} assets", delta=f"-{missing_count}" if missing_count > 0 else "0", delta_color="inverse")
    with m_col4:
        st.metric("Price Date Range", f"{earliest_date_str} to {latest_date_str}")

    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    # Missing Assets Warning Banner
    if missing_count > 0:
        missing_tickers = overview_df[overview_df["STATUS"] != "Available"]["TICKER"].tolist()
        st.warning(
            f"⚠️ **{missing_count} asset(s) with trades or benchmark constituents lack market data in ASSET_PRICES**: "
            f"`{', '.join(missing_tickers)}`. You can add their time series using the form below or trigger a universe refresh."
        )

    # -------------------------------------------------------------------------
    # 2. Asset Market Data Overview Table with Search & Filter
    # -------------------------------------------------------------------------
    st.markdown("#### 📋 Market Data Coverage by Asset")

    col_f1, col_f2 = st.columns([1.5, 1.0])
    with col_f1:
        search_kw = st.text_input(
            "🔍 Filter by Ticker / Source:",
            value="",
            placeholder="Type ticker symbol or source to filter...",
            key="market_data_search_input"
        ).strip().upper()
    with col_f2:
        status_filter = st.selectbox(
            "Filter Status:",
            options=["All Assets", "Available Only", "Missing Data Only"],
            index=0,
            key="market_data_status_filter"
        )

    display_df = overview_df.copy()
    if search_kw:
        display_df = display_df[
            display_df["TICKER"].str.contains(search_kw, case=False, na=False) |
            display_df["SOURCE"].str.contains(search_kw, case=False, na=False)
        ]
    if status_filter == "Available Only":
        display_df = display_df[display_df["STATUS"] == "Available"]
    elif status_filter == "Missing Data Only":
        display_df = display_df[display_df["STATUS"] != "Available"]

    if not display_df.empty:
        # Format table columns for presentation
        formatted_table = []
        for _, row in display_df.iterrows():
            status_indicator = "🟢 Available" if row["STATUS"] == "Available" else "🔴 Missing Market Data"
            first_d = row["FIRST_DATE"] if pd.notna(row["FIRST_DATE"]) and row["FIRST_DATE"] else "—"
            last_d = row["LAST_DATE"] if pd.notna(row["LAST_DATE"]) and row["LAST_DATE"] else "—"
            rec_cnt = f"{int(row['RECORDS']):,}" if row["RECORDS"] > 0 else "0"
            formatted_table.append({
                "Ticker": row["TICKER"],
                "First Date Available": first_d,
                "Last Date Available": last_d,
                "Price Observations": rec_cnt,
                "Currency": row["CURRENCY"] or "—",
                "Universe Source": row["SOURCE"],
                "Data Status": status_indicator,
            })
        st.dataframe(
            pd.DataFrame(formatted_table),
            use_container_width=True,
            hide_index=True,
            height=320
        )
        st.caption(f"Showing {len(formatted_table)} of {total_assets} total tracked assets.")
    else:
        st.info("No assets match the selected filter criteria.")

    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Actions: Add Additional Time Series & Refresh Market Data
    # -------------------------------------------------------------------------
    col_add, col_refresh = st.columns([1.1, 0.9])

    # -------------------------------------------------------------------------
    # Section A: Add Additional Time Series Form
    # -------------------------------------------------------------------------
    with col_add:
        st.markdown("#### ➕ Add Additional Time Series")
        st.caption(
            "Fetch and ingest historical price quotes and foreign exchange rates from Yahoo Finance "
            "for specific ticker symbols across a custom date window."
        )

        with st.container(border=True):
            new_ticker_input = st.text_input(
                "Ticker Symbol(s):",
                value="",
                placeholder="e.g. MSFT, AAPL, CSPX.L",
                help="Enter one or multiple comma-separated ticker symbols to download and ingest into ASSET_PRICES.",
                key="add_ts_ticker_input"
            ).strip().upper()

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                start_dt = st.date_input(
                    "Start Date:",
                    value=default_start_date,
                    help="Beginning of the historical window. Prepopulated with 1 Jul 2024.",
                    key="add_ts_start_date"
                )
            with col_d2:
                end_dt = st.date_input(
                    "End Date:",
                    value=default_end_date,
                    help="End of the historical window. Prepopulated with the latest processed date.",
                    key="add_ts_end_date"
                )

            ingest_btn = st.button("📥 Fetch & Ingest Time Series", type="primary", use_container_width=True, key="btn_ingest_ts")

            if ingest_btn:
                if not new_ticker_input:
                    st.error("Please enter at least one ticker symbol.")
                elif start_dt > end_dt:
                    st.error(f"Start date ({start_dt}) cannot be after end date ({end_dt}).")
                else:
                    with st.spinner(f"Ingesting time series for '{new_ticker_input}' from {start_dt} to {end_dt}..."):
                        try:
                            res = ingest_custom_asset_time_series(
                                tickers=new_ticker_input,
                                start_date=str(start_dt),
                                end_date=str(end_dt),
                                engine=eng
                            )
                            if res.get("status") == "success":
                                st.success(
                                    f"✅ **Time series ingested successfully!** "
                                    f"Processed: `{', '.join(res.get('tickers_processed', []))}` "
                                    f"({res.get('total_rows_added', 0)} rows added). Date range: {res.get('start_date')} to {res.get('end_date')}."
                                )
                                if res.get("tickers_failed"):
                                    st.warning(f"Could not retrieve data for: `{', '.join(res['tickers_failed'])}`.")
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.warning(res.get("message", "Ingestion completed with warnings."))
                        except Exception as ex:
                            st.error(f"Failed to ingest time series: {ex}")

    # -------------------------------------------------------------------------
    # Section B: Refresh Market Data (Clear & Repopulate)
    # -------------------------------------------------------------------------
    with col_refresh:
        st.markdown("#### 🔄 Refresh Market Data Universe")
        st.caption(
            "Clears the market data tables (`ASSET_PRICES` and `FX_RATES`) and repopulates them "
            "from Yahoo Finance for all tracked assets, portfolio trades, and benchmark constituents."
        )

        with st.container(border=True):
            st.markdown(
                "**Universe to Repopulate:**\n"
                f"- **{total_assets}** distinct assets identified across tables\n"
                f"- Historical window: from earliest transaction or `2024-01-01` up to **{latest_proc or default_end_date}**\n"
                "- Automatically recalculates dividend cashflows, portfolio values, and benchmark values"
            )

            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            confirm_chk = st.checkbox(
                "Confirm clearing & repopulating ASSET_PRICES and FX_RATES",
                value=False,
                key="chk_confirm_refresh_all"
            )

            refresh_btn = st.button(
                "🔄 Clear & Repopulate Market Data",
                type="primary",
                disabled=not confirm_chk,
                use_container_width=True,
                key="btn_refresh_all_market_data_tab"
            )

            if refresh_btn:
                with st.spinner("Clearing market data tables and repopulating historical prices and FX rates across all assets..."):
                    try:
                        res = refresh_all_market_data(
                            engine=eng,
                            end_date=str(latest_proc) if latest_proc else str(default_end_date)
                        )
                        if res.get("status") == "success":
                            st.success(
                                f"✅ **Market data refreshed successfully!** "
                                f"Refreshed {len(res.get('tickers_refreshed', []))} of {res.get('tickers_total', 0)} assets. "
                                f"Price records: {res.get('price_records_stored', 0):,}. "
                                f"FX records: {res.get('fx_records_stored', 0):,}."
                            )
                            if res.get("tickers_failed"):
                                st.caption(f"Note: Could not retrieve data for: `{', '.join(res['tickers_failed'])}`.")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.warning(res.get("message", "Market data refresh returned with warnings."))
                    except Exception as ex:
                        st.error(f"Failed to refresh market data: {ex}")
