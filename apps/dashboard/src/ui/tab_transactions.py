"""
Tab 5: Transaction Management & Trade Entry View.
Allows recording new buy/sell transactions with automatic ID incrementation
and live Yahoo Finance closing price lookup.
"""

from datetime import date
from typing import Optional, Dict, Any
import pandas as pd
import streamlit as st

from portfolio_core.db import (
    get_engine,
    get_next_transaction_id,
    record_transaction,
    fetch_all_transactions,
    fetch_portfolio_positions,
    query_yahoo_close_price,
    fetch_and_store_ticker,
    fetch_and_store_fx_rate,
    calculate_and_store_daily_portfolio_values,
    calculate_and_store_daily_benchmark_values
)

from sqlalchemy.engine import Engine

try:
    from src.ui.theme import PALETTE, get_plotly_layout_defaults
except ImportError:
    from apps.dashboard.src.ui.theme import PALETTE, get_plotly_layout_defaults


def render_tab_transactions(
    engine: Optional[Engine] = None,
    transactions_df: Optional[pd.DataFrame] = None,
    positions_dict: Optional[Dict[str, float]] = None
):
    """Renders the Transaction Management and Trade Entry interface."""
    st.markdown("### 📝 Record Portfolio Transactions")
    st.caption("Enter new stock trade transactions into the database with real-time Yahoo Finance price lookup and automatic ID incrementation.")

    engine = engine or get_engine()

    # -------------------------------------------------------------------------
    # 1. Transaction Entry Form & Yahoo Price Preview
    # -------------------------------------------------------------------------
    next_id = get_next_transaction_id(engine=engine)

    col_form, col_preview = st.columns([1.1, 1.0])

    with col_form:
        st.markdown("#### ➕ New Trade Entry")

        st.text_input(
            "Transaction ID (Auto-Incremented):",
            value=f"#{next_id}",
            disabled=True,
            help="The ID is automatically incremented from the highest existing ID in the database."
        )

        col_t1, col_t2 = st.columns([1.2, 1.0])
        with col_t1:
            ticker_input = st.text_input(
                "Ticker Symbol:",
                value="NVDA",
                placeholder="e.g. NVDA, STAN.L, AAPL, MSFT",
                help="Enter standard Yahoo Finance ticker symbol (e.g., NVDA for US stocks, STAN.L for London stocks)."
            ).strip().upper()
        with col_t2:
            tx_date = st.date_input(
                "Transaction Date:",
                value=date.today(),
                max_value=date.today(),
                help="Date on which the transaction occurred."
            )

        col_q1, col_q2 = st.columns([1.2, 1.0])
        with col_q1:
            qty_input = st.number_input(
                "Quantity / Shares:",
                min_value=0.0001,
                value=10.0,
                step=1.0,
                format="%.4f",
                help="Number of shares bought or sold."
            )
        with col_q2:
            action = st.radio(
                "Action:",
                options=["BUY (Add Shares)", "SELL (Reduce Shares)"],
                horizontal=False,
                help="BUY adds positive quantity; SELL deducts shares."
            )

        sync_prices = st.checkbox(
            "Sync latest historical prices for this ticker into ASSET_PRICES",
            value=True,
            help="Automatically fetches and backfills price history from Yahoo Finance for this asset."
        )

        submit_btn = st.button("💾 Record Transaction", type="primary", use_container_width=True)

    with col_preview:
        st.markdown("#### 🔍 Yahoo Finance Market Quote & FX Translation")
        if ticker_input:
            with st.spinner(f"Querying Yahoo Finance for {ticker_input} on {tx_date}..."):
                quote_info = query_yahoo_close_price(ticker=ticker_input, target_date=tx_date, engine=engine)

            if quote_info.get("status") == "success":
                px = float(quote_info["close_price"])
                curr = str(quote_info.get("currency", "USD"))
                found_dt = str(quote_info["found_date"])
                is_exact = bool(quote_info["is_exact_date"])
                fx_rate = float(quote_info.get("fx_rate_to_gbp", 1.0))
                fx_desc = str(quote_info.get("fx_description", ""))
                px_gbp = float(quote_info.get("close_price_gbp", px * fx_rate))

                total_val_orig = px * qty_input
                total_val_gbp = px_gbp * qty_input

                curr_sym = "£" if curr == "GBP" else ("p" if curr in ["GBp", "GBX"] else ("$" if curr == "USD" else ("€" if curr == "EUR" else f"{curr} ")))
                orig_quote_str = f"{px:,.2f} {curr}" if curr in ["GBp", "GBX"] else f"{curr_sym}{px:,.2f} {curr}"
                orig_val_str = f"{total_val_orig:,.2f} {curr}" if curr in ["GBp", "GBX"] else f"{curr_sym}{total_val_orig:,.2f} {curr}"

                with st.container(border=True):
                    # Top Metrics: Translated GBP Price & Original Quote
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.metric(
                            label=f"Price in GBP ({ticker_input})",
                            value=f"£{px_gbp:,.2f} GBP"
                        )
                    with col_p2:
                        st.metric(
                            label=f"Original Quote ({curr})",
                            value=orig_quote_str
                        )

                    # FX Rate Conversion Banner
                    if curr != "GBP":
                        st.info(f"💱 **FX Rate:** 1 {curr} = **£{fx_rate:,.4f} GBP**  \n*{fx_desc}*")
                    else:
                        st.success("💱 **Currency:** Native GBP (FX Rate: 1.00)")

                    # Trading Date Match Info
                    date_badge = "🟢 Exact Market Date Match" if is_exact else "🟡 Nearest Prior Market Day"
                    st.caption(f"📅 **Trading Date Found:** `{found_dt}` ({date_badge})")

                    st.divider()

                    # Estimated Total Trade Value
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        st.metric(
                            label="Est. Total Value (GBP)",
                            value=f"£{total_val_gbp:,.2f} GBP",
                            help=f"{qty_input:,.4f} shares @ £{px_gbp:,.2f} GBP"
                        )
                    with col_v2:
                        if curr != "GBP":
                            st.metric(
                                label=f"Est. Value ({curr})",
                                value=orig_val_str,
                                help=f"{qty_input:,.4f} shares @ {orig_quote_str}"
                            )
                        else:
                            st.metric(
                                label="Order Size",
                                value=f"{qty_input:,.4f} shares"
                            )
            else:
                st.warning(quote_info.get("message", f"Could not retrieve price for {ticker_input}."))
        else:
            st.info("Enter a ticker symbol to look up its market closing price and GBP translated quote.")

    # Handle submission
    if submit_btn:
        if not ticker_input:
            st.error("Please enter a valid ticker symbol.")
        else:
            signed_qty = qty_input if "BUY" in action else -abs(qty_input)
            try:
                rec_res = record_transaction(
                    ticker=ticker_input,
                    transaction_date=tx_date,
                    quantity=signed_qty,
                    transaction_id=next_id,
                    engine=engine
                )

                if sync_prices:
                    try:
                        fetch_and_store_ticker(ticker=ticker_input, engine=engine)
                        fetch_and_store_fx_rate(from_curr="USD", engine=engine)
                        calculate_and_store_daily_portfolio_values(backfill_days=30, engine=engine)
                        calculate_and_store_daily_benchmark_values(engine=engine)
                    except Exception as e:
                        st.caption(f"Price sync note: {e}")

                st.success(f"✅ Successfully recorded Transaction **#{rec_res['id']}**: {signed_qty:+,.4f} shares of **{ticker_input}** on **{tx_date}**!")
                st.cache_data.clear()
                st.rerun()
            except Exception as ex:
                st.error(f"Failed to record transaction: {ex}")

    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. Historical Transactions & Current Net Positions
    # -------------------------------------------------------------------------
    col_hist, col_pos = st.columns([1.3, 0.9])

    with col_hist:
        st.markdown("#### 📋 Recorded Transactions History")
        tx_df = transactions_df if transactions_df is not None else fetch_all_transactions(limit=100, engine=engine)
        if not tx_df.empty:
            formatted_tx = []
            for _, r in tx_df.iterrows():
                q = float(r["QUANTITY"])
                q_sign = "+" if q > 0 else ""
                q_str = f"{q_sign}{q:,.2f}" if not q.is_integer() else f"{q_sign}{q:,.0f}"
                tx_type = "BUY" if q > 0 else "SELL"
                formatted_tx.append({
                    "ID": int(r["ID"]),
                    "Ticker": str(r["TICKER"]),
                    "Date": str(r["TRANSACTION_DATE"]),
                    "Type": tx_type,
                    "Quantity": q_str,
                })
            st.dataframe(pd.DataFrame(formatted_tx), use_container_width=True, hide_index=True)
        else:
            st.info("No transaction records found in the database.")

    with col_pos:
        st.markdown("#### 📊 Current Net Share Positions")
        positions = positions_dict if positions_dict is not None else fetch_portfolio_positions(engine=engine)
        if positions:
            pos_rows = [
                {"Ticker": t, "Net Shares": f"{sh:,.2f}" if not sh.is_integer() else f"{sh:,.0f}"}
                for t, sh in positions.items()
            ]
            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No active positions currently held.")


