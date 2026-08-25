"""
Tab 1: Portfolio Holdings, Allocations, Top Movers & Historical Valuation View.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from portfolio_core.db import fetch_portfolio_values_history
from portfolio_core.analytics.statistics import compute_top_position_movers

try:
    from src.ui.theme import PALETTE, get_plotly_layout_defaults
except ImportError:
    from apps.dashboard.src.ui.theme import PALETTE, get_plotly_layout_defaults


def render_tab_portfolio(
    prices_gbp: pd.DataFrame,
    positions: Dict[str, float],
    asof_date: Optional[str] = None
):
    """Renders the Portfolio Stock Allocations, Top Movers, and Valuation History view."""
    st.markdown("### 💼 Portfolio Holdings & Historical Valuation")
    st.caption("Overview of stock holdings valuation, top daily movers, weight allocation breakdown, and historical trajectory.")

    if prices_gbp.empty or not positions:
        st.warning("Insufficient price data or active positions.")
        return

    # Slice prices up to asof_date if specified
    active_prices = prices_gbp.copy()
    if asof_date:
        asof_ts = pd.to_datetime(asof_date)
        if isinstance(active_prices.index, pd.DatetimeIndex):
            active_prices = active_prices.loc[active_prices.index <= asof_ts]
        else:
            asof_str = str(asof_date)[:10]
            active_prices = active_prices.loc[[str(idx)[:10] <= asof_str for idx in active_prices.index]]

    if active_prices.empty:
        st.warning("No price data available for the specified as-of date.")
        return

    latest_prices = active_prices.iloc[-1]
    active_pos = {t: float(sh) for t, sh in positions.items() if t in latest_prices and sh > 0}
    pos_values = pd.Series({t: active_pos[t] * float(latest_prices[t]) for t in active_pos})
    total_stock_value = float(pos_values.sum())

    # -------------------------------------------------------------------------
    # 1. Valuation KPIs at the Top: Current Value & Previous Value (Stock Holdings Only)
    # -------------------------------------------------------------------------
    pv_df = fetch_portfolio_values_history(days=365, asof_date=asof_date)

    if not pv_df.empty and len(pv_df) >= 1:
        curr_row = pv_df.iloc[-1]
        curr_stocks_val = float(curr_row["STOCKS"]) if "STOCKS" in curr_row else total_stock_value
        curr_date_str = pd.to_datetime(curr_row["DATE"]).strftime("%d %b %Y")

        if len(pv_df) >= 2:
            prev_row = pv_df.iloc[-2]
            prev_stocks_val = float(prev_row["STOCKS"]) if "STOCKS" in prev_row else None
            prev_date_str = pd.to_datetime(prev_row["DATE"]).strftime("%d %b %Y")
            diff_stocks_val = curr_stocks_val - prev_stocks_val if prev_stocks_val is not None else 0.0
            diff_stocks_pct = ((curr_stocks_val / prev_stocks_val) - 1.0) * 100.0 if prev_stocks_val and prev_stocks_val > 0 else 0.0
        else:
            prev_stocks_val = None
            prev_date_str = "N/A"
            diff_stocks_val = 0.0
            diff_stocks_pct = 0.0
    else:
        # Dynamic fallback from market prices
        curr_stocks_val = total_stock_value
        curr_date_str = pd.to_datetime(active_prices.index[-1]).strftime("%d %b %Y")

        if len(active_prices) >= 2:
            prev_prices = active_prices.iloc[-2]
            prev_stocks_val = float(sum(active_pos[t] * float(prev_prices[t]) for t in active_pos if t in prev_prices))
            prev_date_str = pd.to_datetime(active_prices.index[-2]).strftime("%d %b %Y")
            diff_stocks_val = curr_stocks_val - prev_stocks_val
            diff_stocks_pct = ((curr_stocks_val / prev_stocks_val) - 1.0) * 100.0 if prev_stocks_val > 0 else 0.0
        else:
            prev_stocks_val = None
            prev_date_str = "N/A"
            diff_stocks_val = 0.0
            diff_stocks_pct = 0.0

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    with col_kpi1:
        delta_str = f"{diff_stocks_val:+,.2f} ({diff_stocks_pct:+.2f}%)" if prev_stocks_val is not None else None
        st.metric(
            label=f"Current Value ({curr_date_str})",
            value=f"£{curr_stocks_val:,.2f}",
            delta=delta_str
        )

    with col_kpi2:
        st.metric(
            label=f"Previous Value ({prev_date_str})",
            value=f"£{prev_stocks_val:,.2f}" if prev_stocks_val is not None else "N/A"
        )

    with col_kpi3:
        diff_sign = "+" if diff_stocks_val > 0 else ("-" if diff_stocks_val < 0 else "")
        st.metric(
            label="Day-over-Day Change",
            value=f"{diff_sign}£{abs(diff_stocks_val):,.2f}" if prev_stocks_val is not None else "£0.00",
            delta=f"{diff_stocks_pct:+.2f}%" if prev_stocks_val is not None else None
        )

    with col_kpi4:
        st.metric(
            label="Active Positions",
            value=f"{len(active_pos)} assets"
        )

    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. Top 10 Movers in Absolute Terms
    # -------------------------------------------------------------------------
    st.markdown("#### 🚀 Top 10 Movers (Absolute Terms)")
    if len(active_prices) >= 2:
        p_d_str = pd.to_datetime(active_prices.index[-2]).strftime("%d %b %Y")
        c_d_str = pd.to_datetime(active_prices.index[-1]).strftime("%d %b %Y")
        st.caption(f"Top 10 holdings with the largest day-over-day value movement between {p_d_str} and {c_d_str}, ranked by absolute monetary movement (|Δ Value|).")
    else:
        st.caption("Top 10 holdings ranked by absolute value movement (|Δ Value|).")

    movers_df = compute_top_position_movers(
        prices_gbp=active_prices,
        positions=positions,
        asof_date=asof_date,
        top_n=10
    )

    if not movers_df.empty:
        col_m_table, col_m_chart = st.columns([1.35, 1.0])

        with col_m_table:
            movers_rows = []
            for _, r in movers_df.iterrows():
                sh = r["SHARES"]
                sh_str = f"{sh:,.2f}" if not sh.is_integer() else f"{sh:,.0f}"
                diff_sign = "+" if r["DIFF_GBP"] > 0 else ("-" if r["DIFF_GBP"] < 0 else "")
                pct_sign = "+" if r["DIFF_PCT"] > 0 else ("-" if r["DIFF_PCT"] < 0 else "")
                px_chg_sign = "+" if r["PRICE_CHG_PCT"] > 0 else ("-" if r["PRICE_CHG_PCT"] < 0 else "")

                movers_rows.append({
                    "Ticker": r["TICKER"],
                    "Shares": sh_str,
                    "Price (£)": f"£{r['PRICE_TODAY_GBP']:,.2f}",
                    "Prev Price (£)": f"£{r['PRICE_PREV_GBP']:,.2f}",
                    "Price Chg": f"{px_chg_sign}{r['PRICE_CHG_PCT']:.2f}%",
                    "Value (£)": f"£{r['VALUE_TODAY_GBP']:,.2f}",
                    "Prev Value (£)": f"£{r['VALUE_PREV_GBP']:,.2f}",
                    "Day Change (£)": f"{diff_sign}£{abs(r['DIFF_GBP']):,.2f}",
                    "Day Change (%)": f"{pct_sign}{abs(r['DIFF_PCT']):.2f}%",
                    "|Change| (£)": f"£{r['ABS_DIFF_GBP']:,.2f}"
                })
            st.dataframe(pd.DataFrame(movers_rows), use_container_width=True, hide_index=True)

        with col_m_chart:
            # Horizontal bar chart of top 10 movers
            chart_df = movers_df.iloc[::-1]
            bar_colors = ["#10B981" if v >= 0 else "#DC2626" for v in chart_df["DIFF_GBP"]]
            fig_movers = go.Figure(
                data=[
                    go.Bar(
                        y=chart_df["TICKER"],
                        x=chart_df["DIFF_GBP"],
                        orientation="h",
                        marker=dict(color=bar_colors, line=dict(width=1, color="#CBD5E1")),
                        text=[f"{'+' if v > 0 else ''}£{v:,.2f}" for v in chart_df["DIFF_GBP"]],
                        textposition="auto",
                        hovertemplate="<b>%{y}</b><br>Day Change: £%{x:,.2f}<extra></extra>"
                    )
                ]
            )
            layout_movers = get_plotly_layout_defaults()
            layout_movers.update(dict(
                title=dict(text="Top 10 Movers Value Impact (£)", font=dict(size=13, color="#0F172A")),
                xaxis=dict(title="Daily Value Change (£)", tickprefix="£"),
                yaxis=dict(title=""),
                height=380,
                margin=dict(l=60, r=30, t=40, b=40)
            ))
            fig_movers.update_layout(**layout_movers)
            st.plotly_chart(fig_movers, use_container_width=True)
    else:
        st.info("No movers data available for the selected dates.")

    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Allocation Donut & Holdings Table
    # -------------------------------------------------------------------------
    col_a1, col_a2 = st.columns([1.2, 1.8])

    with col_a1:
        st.markdown("#### 🥧 Asset Weight Allocation")
        top_holdings = pos_values.sort_values(ascending=False)

        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=top_holdings.index,
                    values=top_holdings.values,
                    hole=0.48,
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>Value: £%{value:,.2f}<br>Weight: %{percent}<extra></extra>"
                )
            ]
        )
        layout_donut = get_plotly_layout_defaults()
        layout_donut.update(dict(
            title=dict(text=f"Holdings Breakdown (Total: £{total_stock_value:,.2f})", font=dict(size=13, color="#0F172A")),
            height=380,
            showlegend=False
        ))
        fig_donut.update_layout(**layout_donut)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_a2:
        st.markdown("#### 📋 Current Portfolio Positions")
        holdings_rows = []
        for ticker, val in top_holdings.items():
            sh = active_pos[ticker]
            px_val = float(latest_prices[ticker])
            wt = (val / total_stock_value) * 100.0 if total_stock_value > 0 else 0.0
            holdings_rows.append({
                "Ticker": ticker,
                "Shares": f"{sh:,.2f}" if not sh.is_integer() else f"{sh:,.0f}",
                "Price (£)": f"£{px_val:,.2f}",
                "Value (£)": f"£{val:,.2f}",
                "Weight (%)": f"{wt:.2f}%"
            })
        st.dataframe(pd.DataFrame(holdings_rows), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # 4. Historical Portfolio Valuation Timeline (Stock Holdings Only)
    # -------------------------------------------------------------------------
    st.markdown("#### 📈 Historical Stock Holdings Valuation Timeline")

    if not pv_df.empty and len(pv_df) > 1:
        stocks_series = pv_df["STOCKS"] if "STOCKS" in pv_df.columns else pv_df["TOTAL_VALUE"]
        fig_pv = go.Figure()
        fig_pv.add_trace(
            go.Scatter(
                x=pv_df["DATE"], y=stocks_series,
                name="Stock Holdings Value (£)",
                line=dict(color="#1E3A8A", width=2.5),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Stock Holdings Value: <b>£%{y:,.2f}</b><extra></extra>"
            )
        )

        layout_pv = get_plotly_layout_defaults()
        layout_pv.update(dict(
            title=dict(text="Stock Holdings Value History (Last 12 Months)", font=dict(size=14, color="#0F172A")),
            height=360
        ))
        fig_pv.update_layout(**layout_pv)
        fig_pv.update_yaxes(title_text="Stock Holdings Value (£)", tickprefix="£")
        st.plotly_chart(fig_pv, use_container_width=True)
