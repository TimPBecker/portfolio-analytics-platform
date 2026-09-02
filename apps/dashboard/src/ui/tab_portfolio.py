"""
Tab 1: Portfolio Holdings, Allocations, Top Movers & Historical Valuation View.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from portfolio_core.db import (
    fetch_portfolio_values_history
)
from portfolio_core.analytics.statistics import compute_top_position_movers

from sqlalchemy.engine import Engine

try:
    from src.ui.theme import PALETTE, get_plotly_layout_defaults
    from src.services.report_generator import generate_portfolio_pdf_report
except ImportError:
    from apps.dashboard.src.ui.theme import PALETTE, get_plotly_layout_defaults
    from apps.dashboard.src.services.report_generator import generate_portfolio_pdf_report


def render_tab_portfolio(
    prices_gbp: pd.DataFrame,
    positions: Dict[str, float],
    asof_date: Optional[str] = None,
    engine: Optional[Engine] = None,
    db_name: Optional[str] = None
):
    """Renders the Portfolio Stock Allocations, Top Movers, and Valuation History view with PDF Export."""
    def _create_report_bytes() -> bytes:
        ok, path, pdf_bytes, err = generate_portfolio_pdf_report(asof_date=asof_date, db_name=db_name)
        if ok and pdf_bytes:
            return pdf_bytes
        st.error(f"Failed to generate PDF report: {err}")
        return b""

    report_date_str = str(asof_date or 'latest')[:10]
    col_header, col_pdf_action = st.columns([2.6, 1.4])
    with col_header:
        st.markdown("### 💼 Portfolio Holdings & Historical Valuation")
        st.caption("Overview of stock holdings valuation, top daily movers, weight allocation breakdown, and historical trajectory.")
    with col_pdf_action:
        st.markdown("<div style='padding-top: 6px;'>", unsafe_allow_html=True)
        st.download_button(
            label="📄 Export PDF Report",
            data=_create_report_bytes,
            file_name=f"Portfolio_Analytics_Report_{report_date_str}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
            help="Click to generate and download the executive PDF report in a single step."
        )
        st.markdown("</div>", unsafe_allow_html=True)

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
    pv_df = fetch_portfolio_values_history(days=None, asof_date=asof_date, engine=engine)

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
    # 4. Historical Portfolio Valuation Timeline
    # -------------------------------------------------------------------------
    col_chart_title, col_chart_ctrl = st.columns([2.6, 1.4])
    with col_chart_title:
        st.markdown("#### 📈 Portfolio Valuation Trajectory")
        st.caption("Historical valuation trajectory of stock holdings over time.")
    with col_chart_ctrl:
        history_choice = st.selectbox(
            "Select History Horizon:",
            options=["All", "1 Month", "3 Months", "6 Months", "1 Year"],
            index=0,
            key="tab_portfolio_history_horizon",
            help="Filter the valuation history chart by time horizon."
        )

    # Determine valuation history source: database records or dynamic calculation from prices
    chart_data = None
    if not pv_df.empty and len(pv_df) >= 1:
        pv_df_sorted = pv_df.sort_values("DATE").reset_index(drop=True)
        chart_data = pd.DataFrame({
            "DATE": pd.to_datetime(pv_df_sorted["DATE"]),
            "STOCKS": pv_df_sorted["STOCKS"] if "STOCKS" in pv_df_sorted.columns else pv_df_sorted["TOTAL_VALUE"]
        })
    elif not active_prices.empty and active_pos:
        valid_cols = [t for t in active_pos if t in active_prices.columns]
        if valid_cols:
            sh_series = pd.Series({t: active_pos[t] for t in valid_cols})
            computed_stocks = active_prices[valid_cols].dot(sh_series)
            chart_data = pd.DataFrame({
                "DATE": pd.to_datetime(active_prices.index),
                "STOCKS": computed_stocks.values
            })

    if chart_data is not None and not chart_data.empty:
        chart_data = chart_data.sort_values("DATE").reset_index(drop=True)
        max_date = chart_data["DATE"].max()

        if "1 Month" in history_choice or "1 month" in history_choice.lower() or history_choice == "1M":
            start_date = max_date - pd.DateOffset(months=1)
        elif "3 Month" in history_choice or "3month" in history_choice.lower() or history_choice == "3M":
            start_date = max_date - pd.DateOffset(months=3)
        elif "6 Month" in history_choice or "6month" in history_choice.lower() or history_choice == "6M":
            start_date = max_date - pd.DateOffset(months=6)
        elif "1 Year" in history_choice or "1Y" in history_choice or "1 year" in history_choice.lower():
            start_date = max_date - pd.DateOffset(years=1)
        else:
            start_date = chart_data["DATE"].min()

        filtered_chart_data = chart_data[chart_data["DATE"] >= start_date].copy()
        if filtered_chart_data.empty:
            filtered_chart_data = chart_data.tail(1)

        # Calculate dynamic y-axis framing for the selected time horizon
        y_vals = filtered_chart_data["STOCKS"].dropna()
        if not y_vals.empty:
            y_min = float(y_vals.min())
            y_max = float(y_vals.max())
            spread = y_max - y_min
            y_pad = max(spread * 0.08, y_max * 0.02) if y_max > 0 else 100.0
            y_range_min = max(0.0, y_min - y_pad)
            y_range_max = y_max + y_pad
        else:
            y_range_min = None
            y_range_max = None

        fig_pv = go.Figure()

        fig_pv.add_trace(
            go.Scatter(
                x=filtered_chart_data["DATE"],
                y=filtered_chart_data["STOCKS"],
                name="Stock Holdings Value (£)",
                line=dict(color="#1E3A8A", width=3.0),
                fill="tozeroy",
                fillcolor="rgba(30, 58, 138, 0.08)",
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Stock Holdings Value: <b>£%{y:,.2f}</b><extra></extra>"
            )
        )

        layout_pv = get_plotly_layout_defaults()
        layout_pv.update(dict(
            title=dict(text=f"Stock Holdings Valuation History (£) — {history_choice}", font=dict(size=14, color="#0F172A")),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        ))
        fig_pv.update_layout(**layout_pv)

        if y_range_min is not None and y_range_max is not None:
            fig_pv.update_yaxes(
                title_text="Stock Value (£)",
                tickprefix="£",
                range=[y_range_min, y_range_max],
                autorange=False
            )
        else:
            fig_pv.update_yaxes(title_text="Stock Value (£)", tickprefix="£", autorange=True)

        st.plotly_chart(fig_pv, use_container_width=True)
    else:
        st.info("No historical portfolio valuation data available.")
