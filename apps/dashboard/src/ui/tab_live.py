"""
Live Tab: Real-time / 1-Minute Auto-Refreshing Position Valuation vs Database Last Value.
Queries live prices from Yahoo Finance every minute (60s) for all current portfolio positions,
compares each position's current value in GBP to its previous end value in the database,
and displays changes in GBP and percent along with the timestamp from the last query.
"""

from typing import Optional, Dict, Any
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sqlalchemy.engine import Engine

from portfolio_core.db import (
    fetch_live_positions_valuation,
    query_yahoo_live_price,
    fetch_live_fx_rate_to_gbp,
    fetch_latest_db_prices_for_tickers,
)

try:
    from src.ui.theme import PALETTE, get_plotly_layout_defaults
except ImportError:
    from apps.dashboard.src.ui.theme import PALETTE, get_plotly_layout_defaults


LIVE_REFRESH_INTERVAL_SECONDS = 60


def format_live_positions_table(pos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the raw live positions valuation DataFrame into a human-readable display table
    with current value at position level, previous end value, changes in GBP and percent,
    and timestamp from last query.
    """
    if pos_df is None or pos_df.empty:
        return pd.DataFrame(columns=[
            "Ticker",
            "Shares",
            "Currency",
            "Live Price (Native)",
            "Live Price (£)",
            "Prev DB Price (£)",
            "Current Value (£)",
            "Previous End Value (£)",
            "Change (£)",
            "Change (%)",
            "Weight (%)",
            "Last Query",
        ])

    display_rows = []
    for _, row in pos_df.iterrows():
        sh = float(row["SHARES"])
        sh_str = f"{sh:,.2f}" if not sh.is_integer() else f"{sh:,.0f}"
        curr = str(row.get("CURRENCY", "GBP"))
        live_native = float(row.get("LIVE_PRICE_NATIVE", 0.0))
        live_gbp = float(row.get("LIVE_PRICE_GBP", 0.0))
        prev_gbp = float(row.get("PREV_PRICE_GBP", 0.0))
        curr_val = float(row.get("CURRENT_VALUE_GBP", 0.0))
        prev_val = float(row.get("PREV_VALUE_GBP", 0.0))
        chg_gbp = float(row.get("CHANGE_GBP", 0.0))
        chg_pct = float(row.get("CHANGE_PCT", 0.0))
        wt_pct = float(row.get("WEIGHT_PCT", 0.0))
        query_ts = str(row.get("QUERY_TIMESTAMP", ""))

        chg_sign = "+" if chg_gbp > 0 else ("-" if chg_gbp < 0 else "")
        pct_sign = "+" if chg_pct > 0 else ("-" if chg_pct < 0 else "")

        display_rows.append({
            "Ticker": str(row["TICKER"]),
            "Shares": sh_str,
            "Currency": curr,
            "Live Price (Native)": f"{live_native:,.2f}",
            "Live Price (£)": f"£{live_gbp:,.2f}",
            "Prev DB Price (£)": f"£{prev_gbp:,.2f}",
            "Current Value (£)": f"£{curr_val:,.2f}",
            "Previous End Value (£)": f"£{prev_val:,.2f}",
            "Change (£)": f"{chg_sign}£{abs(chg_gbp):,.2f}",
            "Change (%)": f"{pct_sign}{abs(chg_pct):.2f}%",
            "Weight (%)": f"{wt_pct:.2f}%",
            "Last Query": query_ts,
        })

    return pd.DataFrame(display_rows)


def _render_live_positions_content(
    prices_gbp: Optional[pd.DataFrame] = None,
    positions: Optional[Dict[str, float]] = None,
    engine: Optional[Engine] = None,
    db_name: Optional[str] = None,
):
    """Core renderer for live position valuation metrics, table, and chart."""
    with st.spinner("Querying live prices from Yahoo Finance for all current positions..."):
        pos_df, meta = fetch_live_positions_valuation(
            positions=positions,
            prices_gbp=prices_gbp,
            engine=engine
        )

    query_ts = meta.get("query_timestamp", "N/A")
    prev_db_date = meta.get("prev_db_date", "N/A")
    total_curr = float(meta.get("total_current_value_gbp", 0.0))
    total_prev = float(meta.get("total_prev_value_gbp", 0.0))
    total_chg_gbp = float(meta.get("total_change_gbp", 0.0))
    total_chg_pct = float(meta.get("total_change_pct", 0.0))
    pos_count = int(meta.get("positions_count", 0))

    # Top status bar with Last Query timestamp & manual refresh trigger
    col_status, col_btn = st.columns([3.0, 1.0])
    with col_status:
        st.info(
            f"🕒 **Last Query Timestamp:** `{query_ts}`  |  "
            f"🗄️ **Compared Against Last DB Date:** `{prev_db_date}`  |  "
            f"⏱️ **Auto-Refresh:** Every {LIVE_REFRESH_INTERVAL_SECONDS}s (1 minute)"
        )
    with col_btn:
        if st.button("🔄 Refresh Now", key="btn_live_refresh_now", type="primary", use_container_width=True):
            st.rerun()

    if pos_df.empty:
        st.warning("No active portfolio positions found to query.")
        return

    # KPI Metrics Row
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)

    with col_k1:
        delta_str = f"{total_chg_gbp:+,.2f} ({total_chg_pct:+.2f}%)" if total_prev > 0 else None
        st.metric(
            label="Current Live Value (£)",
            value=f"£{total_curr:,.2f}",
            delta=delta_str,
            help="Total live market value in GBP across all current positions from Yahoo Finance."
        )

    with col_k2:
        st.metric(
            label=f"Previous End Value ({prev_db_date})",
            value=f"£{total_prev:,.2f}",
            delta=f"{pos_count} active positions",
            delta_color="off",
            help="Total position value in GBP using the last recorded closing prices in the database."
        )

    with col_k3:
        chg_sign = "+" if total_chg_gbp > 0 else ("-" if total_chg_gbp < 0 else "")
        st.metric(
            label="Live Change (£ / %)",
            value=f"{chg_sign}£{abs(total_chg_gbp):,.2f}",
            delta=f"{total_chg_pct:+.2f}% vs DB End Value",
            help="Aggregate change in GBP and percentage between current Yahoo live valuation and last DB valuation."
        )

    with col_k4:
        st.metric(
            label="Last Query Timestamp",
            value=query_ts.split(" ")[-1] if " " in query_ts else query_ts,
            delta=query_ts.split(" ")[0] if " " in query_ts else "Auto-refresh 60s",
            delta_color="off",
            help=f"Exact timestamp of the last Yahoo Finance price query ({query_ts}). Refreshes automatically every minute."
        )

    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

    # Position-Level Live Valuation Table
    st.markdown("#### 📋 Live Position Valuation vs Last Database Value")
    st.caption(
        f"Position-level comparison of current live value from Yahoo Finance (queried at `{query_ts}`) "
        f"against the previous end value in the database (`{prev_db_date}`)."
    )

    formatted_df = format_live_positions_table(pos_df)
    st.dataframe(formatted_df, use_container_width=True, hide_index=True)

    # Visual Breakdown of Position-Level Live Changes (£)
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📊 Live Position Value Changes (£)")

    chart_df = pos_df.sort_values("CHANGE_GBP", ascending=True)
    bar_colors = ["#10B981" if v >= 0 else "#DC2626" for v in chart_df["CHANGE_GBP"]]
    fig_live = go.Figure(
        data=[
            go.Bar(
                y=chart_df["TICKER"],
                x=chart_df["CHANGE_GBP"],
                orientation="h",
                marker=dict(color=bar_colors, line=dict(width=1, color="#CBD5E1")),
                text=[f"{'+' if v > 0 else ''}£{v:,.2f} ({p:+.2f}%)" for v, p in zip(chart_df["CHANGE_GBP"], chart_df["CHANGE_PCT"])],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>Live Change: £%{x:,.2f}<extra></extra>"
            )
        ]
    )
    layout_live = get_plotly_layout_defaults()
    layout_live.update(dict(
        title=dict(text=f"Live Position Change vs Last DB Value (£) — As of {query_ts}", font=dict(size=13, color="#0F172A")),
        xaxis=dict(title="Change in Value (£)", tickprefix="£"),
        yaxis=dict(title=""),
        height=max(320, len(chart_df) * 28),
        margin=dict(l=60, r=30, t=40, b=40)
    ))
    fig_live.update_layout(**layout_live)
    st.plotly_chart(fig_live, use_container_width=True)


def render_tab_live(
    prices_gbp: Optional[pd.DataFrame] = None,
    positions: Optional[Dict[str, float]] = None,
    engine: Optional[Engine] = None,
    db_name: Optional[str] = None,
):
    """
    Renders the Live Portfolio Valuation tab.
    Automatically re-queries Yahoo Finance every 60 seconds (1 minute) via st.fragment(run_every=60)
    when running inside an active Streamlit session.
    """
    st.markdown("### ⚡ Live Portfolio Valuation Monitor")
    st.caption(
        "Queries live market prices from Yahoo Finance for all current portfolio positions every minute (60s) "
        "and compares position-level valuations against the last value stored in the database."
    )

    has_ctx = False
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
        has_ctx = get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        has_ctx = False

    if has_ctx and hasattr(st, "fragment"):
        @st.fragment(run_every=LIVE_REFRESH_INTERVAL_SECONDS)
        def _auto_refresh_live_fragment():
            _render_live_positions_content(
                prices_gbp=prices_gbp,
                positions=positions,
                engine=engine,
                db_name=db_name
            )

        _auto_refresh_live_fragment()
    else:
        _render_live_positions_content(
            prices_gbp=prices_gbp,
            positions=positions,
            engine=engine,
            db_name=db_name
        )
