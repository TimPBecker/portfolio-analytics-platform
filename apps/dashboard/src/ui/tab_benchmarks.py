"""
Tab: Centralized Benchmarking & Performance Comparison View.
Centralizes all benchmark comparisons, historical valuations against actual portfolio,
value change / return distribution histograms, and linear combination benchmark management.
"""

from typing import List, Optional, Dict, Any
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from portfolio_core.db import (
    get_engine,
    fetch_portfolio_values_history,
    fetch_benchmarks_info,
    add_benchmark,
    delete_benchmark,
    calculate_and_store_daily_benchmark_values,
    fetch_benchmark_values_history,
    fetch_benchmark_transactions
)

try:
    from src.ui.theme import PALETTE, get_plotly_layout_defaults
except ImportError:
    from apps.dashboard.src.ui.theme import PALETTE, get_plotly_layout_defaults


def render_tab_benchmarks(
    prices_gbp: pd.DataFrame,
    asof_date: Optional[str] = None
):
    """Renders the Centralized Benchmarking, Comparative Performance, and Distribution view."""
    st.markdown("### 🎯 Centralized Benchmarking & Performance Analysis")
    st.caption(
        "Compare actual portfolio performance against market benchmarks and custom linear combinations. "
        "Each benchmark maintains an exact shadow portfolio matching the GBP capital deployed on every trade date."
    )

    engine = get_engine()

    # Load benchmark metadata & valuations
    bm_info_df = fetch_benchmarks_info(engine=engine)
    bm_history_df = fetch_benchmark_values_history(asof_date=asof_date, engine=engine)
    pv_df = fetch_portfolio_values_history(asof_date=asof_date, engine=engine)

    available_bms = bm_info_df["BENCHMARK_CODE"].tolist() if not bm_info_df.empty else []
    bm_name_map = dict(zip(bm_info_df["BENCHMARK_CODE"], bm_info_df["NAME"])) if not bm_info_df.empty else {}

    # -------------------------------------------------------------------------
    # 1. Benchmark Selection & Historical Valuation Trajectory
    # -------------------------------------------------------------------------
    st.markdown("#### 📈 Historical Valuation & Performance Comparison")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.5, 1.0, 1.0])
    with col_ctrl1:
        selected_bms = st.multiselect(
            "Select Benchmarks to Compare:",
            options=available_bms,
            default=available_bms,
            format_func=lambda b: f"{b} — {bm_name_map.get(b, b)}",
            help="Select one or more benchmark shadow portfolios to compare against your actual portfolio."
        )
    with col_ctrl2:
        chart_mode = st.radio(
            "Display Metric:",
            options=["Valuation (£)", "Indexed Growth (%)"],
            horizontal=True,
            help="Valuation (£) compares actual monetary value over time; Indexed Growth (%) normalizes values to 0% at inception."
        )
    with col_ctrl3:
        lookback_choice = st.selectbox(
            "Time Horizon:",
            options=["All Time", "3 Years", "2 Years", "1 Year", "6 Months"],
            index=0
        )

    if not pv_df.empty and len(pv_df) > 1:
        pv_df_sorted = pv_df.sort_values("DATE").reset_index(drop=True)
        # Total portfolio value including stock holdings and dividend cash account
        port_totals_series = pv_df_sorted["TOTAL_VALUE"] if "TOTAL_VALUE" in pv_df_sorted.columns else (
            pv_df_sorted["STOCKS"] + pv_df_sorted.get("CASH", 0.0)
        )

        # Date filtering based on lookback
        max_date = pv_df_sorted["DATE"].max()
        if lookback_choice == "6 Months":
            cutoff_date = max_date - pd.DateOffset(months=6)
        elif lookback_choice == "1 Year":
            cutoff_date = max_date - pd.DateOffset(years=1)
        elif lookback_choice == "2 Years":
            cutoff_date = max_date - pd.DateOffset(years=2)
        elif lookback_choice == "3 Years":
            cutoff_date = max_date - pd.DateOffset(years=3)
        else:
            cutoff_date = pv_df_sorted["DATE"].min()

        pv_filtered = pv_df_sorted[pv_df_sorted["DATE"] >= cutoff_date].reset_index(drop=True)
        port_vals_filtered = pv_filtered["TOTAL_VALUE"] if "TOTAL_VALUE" in pv_filtered.columns else (
            pv_filtered["STOCKS"] + pv_filtered.get("CASH", 0.0)
        )

        fig_pv = go.Figure()

        bm_colors = ["#F59E0B", "#8B5CF6", "#10B981", "#EC4899", "#06B6D4", "#E11D48", "#64748B"]
        bm_dash_styles = ["dash", "dot", "dashdot", "longdash", "longdashdot"]

        if chart_mode == "Valuation (£)":
            # Primary Portfolio Trace (Total Value: Stocks + Cash from Dividends)
            fig_pv.add_trace(
                go.Scatter(
                    x=pv_filtered["DATE"],
                    y=port_vals_filtered,
                    name="My Portfolio (Stocks + Cash)",
                    line=dict(color="#1E3A8A", width=3.2),
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>My Portfolio Total: <b>£%{y:,.2f}</b><extra></extra>"
                )
            )

            # Benchmark Shadow Portfolio Traces
            for idx, bm_code in enumerate(selected_bms):
                bm_sub = bm_history_df[bm_history_df["BENCHMARK_CODE"] == bm_code]
                if not bm_sub.empty:
                    bm_sub = bm_sub.sort_values("DATE").reset_index(drop=True)
                    bm_sub = bm_sub[bm_sub["DATE"] >= cutoff_date]
                    if not bm_sub.empty:
                        color = bm_colors[idx % len(bm_colors)]
                        dash = bm_dash_styles[idx % len(bm_dash_styles)]
                        bm_name = bm_name_map.get(bm_code, bm_code)

                        fig_pv.add_trace(
                            go.Scatter(
                                x=bm_sub["DATE"],
                                y=bm_sub["TOTAL_VALUE"],
                                name=f"{bm_code} (Stocks + Cash)",
                                line=dict(color=color, width=2.2, dash=dash),
                                hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{bm_code} Total: <b>£%{{y:,.2f}}</b><extra></extra>"
                            )
                        )

            layout_pv = get_plotly_layout_defaults()
            layout_pv.update(dict(
                title=dict(text="Total Valuation: Portfolio vs Benchmark Shadow Portfolios (£)", font=dict(size=14, color="#0F172A")),
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            ))
            fig_pv.update_layout(**layout_pv)
            fig_pv.update_yaxes(title_text="Total Valuation (£)", tickprefix="£")
            st.plotly_chart(fig_pv, use_container_width=True)

        else:
            # Indexed Growth (%) Mode (Total Return inclusive of dividend cash)
            base_port = port_vals_filtered.iloc[0] if port_vals_filtered.iloc[0] > 0 else 1.0
            port_growth = ((port_vals_filtered / base_port) - 1.0) * 100.0

            fig_pv.add_trace(
                go.Scatter(
                    x=pv_filtered["DATE"],
                    y=port_growth,
                    name="My Portfolio (Total Return)",
                    line=dict(color="#1E3A8A", width=3.2),
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>My Portfolio Total Return: <b>%{y:+.2f}%</b><extra></extra>"
                )
            )

            for idx, bm_code in enumerate(selected_bms):
                bm_sub = bm_history_df[bm_history_df["BENCHMARK_CODE"] == bm_code]
                if not bm_sub.empty:
                    bm_sub = bm_sub.sort_values("DATE").reset_index(drop=True)
                    bm_sub = bm_sub[bm_sub["DATE"] >= cutoff_date]
                    if not bm_sub.empty:
                        base_bm = bm_sub["TOTAL_VALUE"].iloc[0] if bm_sub["TOTAL_VALUE"].iloc[0] > 0 else 1.0
                        bm_growth = ((bm_sub["TOTAL_VALUE"] / base_bm) - 1.0) * 100.0
                        color = bm_colors[idx % len(bm_colors)]
                        dash = bm_dash_styles[idx % len(bm_dash_styles)]
                        bm_name = bm_name_map.get(bm_code, bm_code)

                        fig_pv.add_trace(
                            go.Scatter(
                                x=bm_sub["DATE"],
                                y=bm_growth,
                                name=f"{bm_code} (Total Return)",
                                line=dict(color=color, width=2.2, dash=dash),
                                hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{bm_code} Total Return: <b>%{{y:+.2f}}%</b><extra></extra>"
                            )
                        )

            layout_pv = get_plotly_layout_defaults()
            layout_pv.update(dict(
                title=dict(text="Total Return Growth (%) Comparison (Stocks + Dividend Cash)", font=dict(size=14, color="#0F172A")),
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            ))
            fig_pv.update_layout(**layout_pv)
            fig_pv.update_yaxes(title_text="Cumulative Total Return (%)", ticksuffix="%")
            st.plotly_chart(fig_pv, use_container_width=True)

        # ---------------------------------------------------------------------
        # Performance Summary Scorecard Table (with Stocks & Dividend Cash Breakdown)
        # ---------------------------------------------------------------------
        st.markdown("##### 📊 Comparative Total Performance & Asset Breakdown Scorecard")
        scorecard_rows = []

        # Portfolio metrics
        port_latest_tot = float(port_vals_filtered.iloc[-1])
        port_latest_stk = float(pv_filtered["STOCKS"].iloc[-1]) if "STOCKS" in pv_filtered.columns else port_latest_tot
        port_latest_csh = float(pv_filtered["CASH"].iloc[-1]) if "CASH" in pv_filtered.columns else 0.0
        port_init_val = float(port_vals_filtered.iloc[0]) if float(port_vals_filtered.iloc[0]) > 0 else 1.0
        port_total_ret = ((port_latest_tot / port_init_val) - 1.0) * 100.0

        # 1-day, 30-day, 1-year returns
        port_1d = ((port_vals_filtered.iloc[-1] / port_vals_filtered.iloc[-2]) - 1.0) * 100.0 if len(port_vals_filtered) > 1 else 0.0
        idx_30 = max(0, len(port_vals_filtered) - 22)
        port_30d = ((port_vals_filtered.iloc[-1] / port_vals_filtered.iloc[idx_30]) - 1.0) * 100.0 if len(port_vals_filtered) > idx_30 else 0.0
        idx_1y = max(0, len(port_vals_filtered) - 252)
        port_1y = ((port_vals_filtered.iloc[-1] / port_vals_filtered.iloc[idx_1y]) - 1.0) * 100.0 if len(port_vals_filtered) > idx_1y else 0.0

        # Volatility & Sharpe
        port_pct_changes = port_vals_filtered.pct_change().dropna()
        port_ann_vol = float(port_pct_changes.std() * np.sqrt(252) * 100.0) if len(port_pct_changes) > 1 else 0.0
        port_cagr = (((port_latest_tot / port_init_val) ** (252.0 / max(len(port_vals_filtered), 1))) - 1.0) * 100.0 if port_init_val > 0 else 0.0
        port_sharpe = (port_cagr / port_ann_vol) if port_ann_vol > 0 else 0.0

        scorecard_rows.append({
            "Asset / Benchmark": "💼 My Portfolio",
            "Total Value (£)": f"£{port_latest_tot:,.2f}",
            "Stock Holdings (£)": f"£{port_latest_stk:,.2f}",
            "Dividend Cash (£)": f"£{port_latest_csh:,.2f}",
            "1-Day Return": f"{'+' if port_1d >= 0 else ''}{port_1d:.2f}%",
            "30-Day Return": f"{'+' if port_30d >= 0 else ''}{port_30d:.2f}%",
            "1-Year Return": f"{'+' if port_1y >= 0 else ''}{port_1y:.2f}%",
            "Period Return": f"{'+' if port_total_ret >= 0 else ''}{port_total_ret:.2f}%",
            "Ann. Volatility": f"{port_ann_vol:.2f}%",
            "Return/Risk": f"{port_sharpe:.2f}"
        })

        for bm_code in selected_bms:
            bm_sub = bm_history_df[bm_history_df["BENCHMARK_CODE"] == bm_code]
            if not bm_sub.empty:
                bm_sub = bm_sub.sort_values("DATE").reset_index(drop=True)
                bm_sub = bm_sub[bm_sub["DATE"] >= cutoff_date]
                if not bm_sub.empty and len(bm_sub) > 1:
                    bm_latest_tot = float(bm_sub["TOTAL_VALUE"].iloc[-1])
                    bm_latest_stk = float(bm_sub["STOCKS"].iloc[-1]) if "STOCKS" in bm_sub.columns else bm_latest_tot
                    bm_latest_csh = float(bm_sub["CASH"].iloc[-1]) if "CASH" in bm_sub.columns else 0.0
                    bm_init_val = float(bm_sub["TOTAL_VALUE"].iloc[0]) if float(bm_sub["TOTAL_VALUE"].iloc[0]) > 0 else 1.0
                    bm_tot_ret = ((bm_latest_tot / bm_init_val) - 1.0) * 100.0

                    bm_1d = ((bm_sub["TOTAL_VALUE"].iloc[-1] / bm_sub["TOTAL_VALUE"].iloc[-2]) - 1.0) * 100.0 if len(bm_sub) > 1 else 0.0
                    bm_30_idx = max(0, len(bm_sub) - 22)
                    bm_30d = ((bm_sub["TOTAL_VALUE"].iloc[-1] / bm_sub["TOTAL_VALUE"].iloc[bm_30_idx]) - 1.0) * 100.0 if len(bm_sub) > bm_30_idx else 0.0
                    bm_1y_idx = max(0, len(bm_sub) - 252)
                    bm_1y = ((bm_sub["TOTAL_VALUE"].iloc[-1] / bm_sub["TOTAL_VALUE"].iloc[bm_1y_idx]) - 1.0) * 100.0 if len(bm_sub) > bm_1y_idx else 0.0

                    bm_pct_changes = bm_sub["TOTAL_VALUE"].pct_change().dropna()
                    bm_ann_vol = float(bm_pct_changes.std() * np.sqrt(252) * 100.0) if len(bm_pct_changes) > 1 else 0.0
                    bm_cagr = (((bm_latest_tot / bm_init_val) ** (252.0 / max(len(bm_sub), 1))) - 1.0) * 100.0 if bm_init_val > 0 else 0.0
                    bm_sharpe = (bm_cagr / bm_ann_vol) if bm_ann_vol > 0 else 0.0

                    bm_label = f"🎯 {bm_code} ({bm_name_map.get(bm_code, bm_code)})"
                    scorecard_rows.append({
                        "Asset / Benchmark": bm_label,
                        "Total Value (£)": f"£{bm_latest_tot:,.2f}",
                        "Stock Holdings (£)": f"£{bm_latest_stk:,.2f}",
                        "Dividend Cash (£)": f"£{bm_latest_csh:,.2f}",
                        "1-Day Return": f"{'+' if bm_1d >= 0 else ''}{bm_1d:.2f}%",
                        "30-Day Return": f"{'+' if bm_30d >= 0 else ''}{bm_30d:.2f}%",
                        "1-Year Return": f"{'+' if bm_1y >= 0 else ''}{bm_1y:.2f}%",
                        "Period Return": f"{'+' if bm_tot_ret >= 0 else ''}{bm_tot_ret:.2f}%",
                        "Ann. Volatility": f"{bm_ann_vol:.2f}%",
                        "Return/Risk": f"{bm_sharpe:.2f}"
                    })

        st.dataframe(pd.DataFrame(scorecard_rows), use_container_width=True, hide_index=True)

        st.markdown("<div style='margin-bottom: 2.5rem;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 2. Histogram of Value Changes & Daily Returns Distribution
        # ---------------------------------------------------------------------
        st.markdown("#### 📊 Value Changes & Daily Returns Distribution Histogram")
        st.caption("Inspect the historical distribution of daily valuation changes (£) and daily returns (%) for your actual portfolio alongside selected benchmarks.")

        col_h_metric, col_h_style, col_h_bins = st.columns([1.2, 1.0, 1.0])
        with col_h_metric:
            hist_metric = st.radio(
                "Distribution Variable:",
                options=["Daily Value Change (£)", "Daily Return (%)"],
                horizontal=True,
                help="Daily Value Change (£) shows monetary daily P&L fluctuations; Daily Return (%) shows proportional percentage changes."
            )
        with col_h_style:
            hist_barmode = st.radio(
                "Histogram Layout:",
                options=["Overlaid", "Side-by-Side (Subplots)"],
                horizontal=True
            )
        with col_h_bins:
            nbins = st.slider("Number of Bins:", min_value=15, max_value=80, value=35, step=5)

        # Prepare distribution data series
        dist_data: Dict[str, pd.Series] = {}

        if hist_metric == "Daily Value Change (£)":
            port_diff = port_vals_filtered.diff().dropna()
            dist_data["My Portfolio"] = port_diff

            for bm_code in selected_bms:
                bm_sub = bm_history_df[bm_history_df["BENCHMARK_CODE"] == bm_code]
                if not bm_sub.empty:
                    bm_sub = bm_sub.sort_values("DATE").reset_index(drop=True)
                    bm_sub = bm_sub[bm_sub["DATE"] >= cutoff_date]
                    if len(bm_sub) > 1:
                        dist_data[f"{bm_code}"] = bm_sub["TOTAL_VALUE"].diff().dropna()
        else:
            port_pct = port_vals_filtered.pct_change().dropna() * 100.0
            dist_data["My Portfolio"] = port_pct

            for bm_code in selected_bms:
                bm_sub = bm_history_df[bm_history_df["BENCHMARK_CODE"] == bm_code]
                if not bm_sub.empty:
                    bm_sub = bm_sub.sort_values("DATE").reset_index(drop=True)
                    bm_sub = bm_sub[bm_sub["DATE"] >= cutoff_date]
                    if len(bm_sub) > 1:
                        dist_data[f"{bm_code}"] = bm_sub["TOTAL_VALUE"].pct_change().dropna() * 100.0

        if dist_data:
            if hist_barmode == "Overlaid":
                fig_hist = go.Figure()

                for idx, (label, series) in enumerate(dist_data.items()):
                    is_port = "My Portfolio" in label
                    color = "#1E3A8A" if is_port else bm_colors[(idx - 1) % len(bm_colors)]
                    opacity = 0.65 if is_port else 0.50

                    fig_hist.add_trace(
                        go.Histogram(
                            x=series,
                            name=label,
                            nbinsx=nbins,
                            opacity=opacity,
                            marker=dict(color=color, line=dict(color="white", width=0.5)),
                            hovertemplate=f"<b>{label}</b><br>Bin: %{{x}}<br>Frequency: %{{y}}<extra></extra>"
                        )
                    )

                    # Vertical line for mean
                    mean_val = float(series.mean())
                    fig_hist.add_vline(
                        x=mean_val,
                        line_dash="dash" if not is_port else "solid",
                        line_color=color,
                        line_width=1.5,
                        annotation_text=f"Mean: {mean_val:+.2f}",
                        annotation_position="top"
                    )

                layout_hist = get_plotly_layout_defaults()
                layout_hist.update(dict(
                    barmode="overlay",
                    title=dict(text=f"Historical Distribution of {hist_metric}", font=dict(size=14, color="#0F172A")),
                    height=420,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                ))
                fig_hist.update_layout(**layout_hist)
                x_prefix = "£" if "£" in hist_metric else ""
                x_suffix = "%" if "%" in hist_metric else ""
                fig_hist.update_xaxes(title_text=hist_metric, tickprefix=x_prefix, ticksuffix=x_suffix)
                fig_hist.update_yaxes(title_text="Frequency (Days)")
                st.plotly_chart(fig_hist, use_container_width=True)

            else:
                # Subplots layout
                num_series = len(dist_data)
                cols_count = min(3, num_series)
                rows_count = int(np.ceil(num_series / cols_count))

                fig_sub = make_subplots(
                    rows=rows_count,
                    cols=cols_count,
                    subplot_titles=list(dist_data.keys()),
                    vertical_spacing=0.12,
                    horizontal_spacing=0.08
                )

                for idx, (label, series) in enumerate(dist_data.items()):
                    r = (idx // cols_count) + 1
                    c = (idx % cols_count) + 1
                    is_port = "My Portfolio" in label
                    color = "#1E3A8A" if is_port else bm_colors[(idx - 1) % len(bm_colors)]

                    fig_sub.add_trace(
                        go.Histogram(
                            x=series,
                            name=label,
                            nbinsx=nbins,
                            marker=dict(color=color, line=dict(color="white", width=0.5)),
                            showlegend=False
                        ),
                        row=r,
                        col=c
                    )

                layout_sub = get_plotly_layout_defaults()
                layout_sub.update(dict(
                    height=280 * rows_count,
                    title=dict(text=f"Individual Distributions: {hist_metric}", font=dict(size=14, color="#0F172A"))
                ))
                fig_sub.update_layout(**layout_sub)
                st.plotly_chart(fig_sub, use_container_width=True)

            # Distribution Summary Statistics Table
            st.markdown("##### 📋 Distribution Summary Statistics")
            stat_rows = []
            for label, series in dist_data.items():
                m = float(series.mean())
                s = float(series.std())
                med = float(series.median())
                mn = float(series.min())
                mx = float(series.max())
                pos_pct = float((series > 0).mean() * 100.0)
                unit_sym = "£" if "£" in hist_metric else ""
                unit_pct = "%" if "%" in hist_metric else ""

                stat_rows.append({
                    "Series": label,
                    "Mean Daily": f"{unit_sym}{m:+,.2f}{unit_pct}",
                    "Std Dev (Daily Vol)": f"{unit_sym}{s:,.2f}{unit_pct}",
                    "Median": f"{unit_sym}{med:+,.2f}{unit_pct}",
                    "Max Gain (Best Day)": f"{unit_sym}{mx:+,.2f}{unit_pct}",
                    "Max Loss (Worst Day)": f"{unit_sym}{mn:+,.2f}{unit_pct}",
                    "Positive Days (% Win Rate)": f"{pos_pct:.1f}%"
                })
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No historical portfolio valuation data available.")

    st.markdown("<div style='margin-bottom: 2.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Benchmark Assets & Linear Combination Management
    # -------------------------------------------------------------------------
    st.markdown("### ⚙️ Benchmark Asset & Blend Management")
    st.caption("Register new single-ticker benchmarks or weighted linear combinations, or remove existing benchmarks from the database.")

    col_bm_form, col_bm_table = st.columns([1.1, 1.2])

    with col_bm_form:
        with st.container(border=True):
            st.markdown("##### ➕ Register Benchmark (Single or Linear Combination)")
            bm_constituents_input = st.text_input(
                "Constituent Ticker(s) & Weights (%):",
                value="CSP1.L: 60, VUKE.L: 40",
                placeholder="e.g. CSP1.L: 60, VUKE.L: 40 or VWRL.L: 100",
                help="Enter a single ticker or comma-separated list of 'TICKER: WEIGHT' pairs. Weights will be normalized to 100%."
            ).strip()

            new_bm_name = st.text_input(
                "Benchmark Name (Optional):",
                value="",
                placeholder="Leave blank for fallback name (e.g. CSP1.L_60_VUKE.L_40)",
                help="If omitted, the benchmark name automatically falls back to 'TICKER1_PERCENT1_TICKER2_PERCENT2_...'."
            ).strip()

            new_bm_desc = st.text_input(
                "Description (Optional):",
                value="",
                placeholder="e.g. 60% S&P 500, 40% FTSE 100 Equity Blend"
            ).strip()

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                add_bm_btn = st.button("➕ Add Benchmark", type="primary", use_container_width=True)
            with col_btn2:
                recalc_bm_btn = st.button("🔄 Recalculate Shadow Trades", use_container_width=True)

            if add_bm_btn:
                if not bm_constituents_input:
                    st.error("Please enter benchmark constituent ticker(s).")
                else:
                    try:
                        res_add = add_benchmark(
                            constituents=bm_constituents_input,
                            name=new_bm_name if new_bm_name else None,
                            description=new_bm_desc if new_bm_desc else None,
                            engine=engine
                        )
                        calculate_and_store_daily_benchmark_values(engine=engine)
                        st.success(f"✅ Benchmark **{res_add['benchmark_code']}** ('{res_add['name']}') successfully registered!")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Failed to add benchmark: {ex}")

            if recalc_bm_btn:
                with st.spinner("Regenerating benchmark shadow transactions & daily valuations..."):
                    res_bm = calculate_and_store_daily_benchmark_values(engine=engine)
                    st.success(f"✅ Generated {res_bm.get('records_stored', 0)} benchmark valuation points!")
                    st.cache_data.clear()
                    st.rerun()

    with col_bm_table:
        with st.container(border=True):
            st.markdown("##### 📋 Registered Benchmarks (BENCHMARKS Table)")
            bm_df = fetch_benchmarks_info(engine=engine)
            if not bm_df.empty:
                display_cols = [c for c in ["BENCHMARK_CODE", "NAME", "CONSTITUENTS_DISPLAY", "DESCRIPTION"] if c in bm_df.columns]
                st.dataframe(bm_df[display_cols], use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("##### 🗑️ Remove Benchmark")
                bm_to_del = st.selectbox(
                    "Select Benchmark to Remove:",
                    options=bm_df["BENCHMARK_CODE"].tolist(),
                    index=None,
                    format_func=lambda b: f"{b} — {bm_name_map.get(b, b)}",
                    placeholder="Select a benchmark to remove from database..."
                )
                if bm_to_del:
                    del_name = bm_name_map.get(bm_to_del, bm_to_del)
                    st.caption(f"Selected: **{bm_to_del}** ({del_name})")
                    if st.button(f"🗑️ Delete Benchmark '{bm_to_del}'", type="secondary", use_container_width=True):
                        try:
                            delete_benchmark(bm_to_del, engine=engine)
                            st.success(f"✅ Deleted benchmark **{bm_to_del}** from the database.")
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Failed to delete benchmark: {ex}")
            else:
                st.info("No benchmark tickers registered.")

    # Expander for Benchmark Shadow Transactions Inspection
    with st.expander("🔍 View Benchmark Shadow Transactions (BENCHMARK_TRANSACTIONS Table)"):
        bm_tx_all = fetch_benchmark_transactions(engine=engine)
        if not bm_tx_all.empty:
            st.caption("Each row represents a shadow trade created with equivalent GBP invested value and constituent weight on the original transaction date.")
            st.dataframe(bm_tx_all, use_container_width=True, hide_index=True)
        else:
            st.info("No benchmark shadow transactions recorded yet.")
