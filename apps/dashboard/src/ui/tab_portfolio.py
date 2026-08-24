"""
Tab 1: Portfolio Holdings, Allocations & Historical Valuation View.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from portfolio_core.db import fetch_portfolio_values_history
from src.ui.theme import PALETTE, get_plotly_layout_defaults


def render_tab_portfolio(
    prices_gbp: pd.DataFrame,
    positions: Dict[str, float],
    asof_date: Optional[str] = None
):
    """Renders the Portfolio Allocations and Valuation History view."""
    st.markdown("### 💼 Portfolio Holdings & Historical Valuation")
    st.caption("Overview of current holdings, weight allocation breakdown, and portfolio valuation trajectory.")

    if prices_gbp.empty or not positions:
        st.warning("Insufficient price data or active positions.")
        return

    latest_prices = prices_gbp.iloc[-1]
    active_pos = {t: float(sh) for t, sh in positions.items() if t in latest_prices and sh > 0}
    pos_values = pd.Series({t: active_pos[t] * float(latest_prices[t]) for t in active_pos})
    total_stock_value = float(pos_values.sum())

    # -------------------------------------------------------------------------
    # 1. Allocation Donut & Holdings Table
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
    # 2. Historical Portfolio Valuation Timeline
    # -------------------------------------------------------------------------
    st.markdown("#### 📈 Historical Portfolio Valuation Timeline")
    pv_df = fetch_portfolio_values_history(days=365, asof_date=asof_date)

    if not pv_df.empty and len(pv_df) > 1:
        fig_pv = go.Figure()
        fig_pv.add_trace(
            go.Scatter(
                x=pv_df["DATE"], y=pv_df["TOTAL_VALUE"],
                name="Total Portfolio Value (£)",
                line=dict(color="#1E3A8A", width=2.5),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Total Value: <b>£%{y:,.2f}</b><extra></extra>"
            )
        )
        if "STOCKS" in pv_df.columns:
            fig_pv.add_trace(
                go.Scatter(
                    x=pv_df["DATE"], y=pv_df["STOCKS"],
                    name="Stock Holdings Value (£)",
                    line=dict(color="#3B82F6", width=1.8, dash="dot"),
                    hovertemplate="Stocks Value: £%{y:,.2f}<extra></extra>"
                )
            )
        if "CASH" in pv_df.columns:
            fig_pv.add_trace(
                go.Scatter(
                    x=pv_df["DATE"], y=pv_df["CASH"],
                    name="Cash Account (£)",
                    line=dict(color="#10B981", width=1.5, dash="dash"),
                    hovertemplate="Cash: £%{y:,.2f}<extra></extra>"
                )
            )

        layout_pv = get_plotly_layout_defaults()
        layout_pv.update(dict(
            title=dict(text="Portfolio Value History (Last 12 Months)", font=dict(size=14, color="#0F172A")),
            height=360
        ))
        fig_pv.update_layout(**layout_pv)
        fig_pv.update_yaxes(title_text="Portfolio Value (£)", tickprefix="£")
        st.plotly_chart(fig_pv, use_container_width=True)
