"""
Tab 1: Rolling Volatility Analytics View.
Interactive exploration and comparison of dynamic volatility estimators:
- RiskMetrics EWMA (default λ=0.94, customizable/multi-parameter)
- Equally Weighted Rolling Sample Standard Deviations (default 60d, customizable/multi-window)
- Volatility Scaling Multipliers (σ_today / σ_t) across all estimators
- Dual-axis price overlays and volatility spread/ratio diagnostics.
"""

from typing import List, Optional, Dict, Any, Tuple
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from portfolio_core.analytics.volatility import (
    calculate_sample_volatility,
    calculate_ewma_volatility,
    calculate_scaling_factors,
    compute_volatility_summary_metrics
)
try:
    from src.ui.theme import PALETTE, STOCK_COLORS, get_plotly_layout_defaults
except ImportError:
    from apps.dashboard.src.ui.theme import PALETTE, STOCK_COLORS, get_plotly_layout_defaults


# Color palette for distinct estimators
ESTIMATOR_COLORS = [
    "#1E3A8A",  # Deep Navy Blue (Primary EWMA)
    "#059669",  # Emerald Green (Primary Rolling 60d)
    "#D97706",  # Amber Orange
    "#7C3AED",  # Royal Purple
    "#0891B2",  # Cyan / Teal
    "#DC2626",  # Crimson Red
    "#EC4899",  # Pink
    "#4F46E5",  # Indigo
    "#CA8A04",  # Yellow Gold
    "#16A34A",  # Green
]

LINE_STYLES = ["solid", "dash", "dot", "dashdot", "longdash"]


def render_tab_volatility(
    prices_gbp: pd.DataFrame,
    available_tickers: List[str],
    raw_prices_cache: Optional[Dict[str, pd.DataFrame]] = None
):
    """Renders the Rolling Volatilities view with multi-estimator scaling factor comparison."""
    st.markdown("### 📊 Rolling Volatility & Regime Analytics")
    st.caption("Compare dynamic EWMA and equally weighted rolling sample volatility estimators, volatility scaling multipliers, and price overlays.")

    if prices_gbp.empty:
        st.warning("No price history available in the database. Please check your database connection.")
        return

    # -------------------------------------------------------------------------
    # 1. Top Controls: Stock & Horizon Selection
    # -------------------------------------------------------------------------
    col_ctrl1, col_ctrl2 = st.columns([1.6, 1.4])

    with col_ctrl1:
        default_selected = [t for t in ["NVDA", "STAN.L"] if t in available_tickers]
        if not default_selected and available_tickers:
            default_selected = available_tickers[:2]

        selected_tickers = st.multiselect(
            "Select Stock(s) to Analyze:",
            options=available_tickers,
            default=default_selected,
            help="Choose one or multiple stocks to analyze dynamic volatility and scaling factors."
        )

    with col_ctrl2:
        horizon_options = ["1 Month", "3 Months", "6 Months", "1 Year (Default)", "2 Years", "All Available", "Custom Range"]
        horizon_choice = st.selectbox("Time Horizon:", horizon_options, index=3)

    if not selected_tickers:
        st.info("Please select at least one ticker from the dropdown above to display rolling volatility.")
        return

    # -------------------------------------------------------------------------
    # 2. Multi-Estimator Configuration (Defaults: EWMA λ=0.94 & Rolling 60d)
    # -------------------------------------------------------------------------
    with st.expander("⚡ Volatility Estimators Configuration (EWMA & Rolling Look-Back Periods)", expanded=True):
        st.caption("Compare the EWMA process with equally weighted rolling sample volatility across custom parameters.")
        
        col_est1, col_est2, col_est3 = st.columns([1.5, 1.5, 1.0])

        with col_est1:
            st.markdown("##### 📈 RiskMetrics EWMA Parameters")
            standard_lambdas = [0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99]
            selected_lambdas = st.multiselect(
                "Select EWMA Decay Factor(s) (λ):",
                options=standard_lambdas,
                default=[0.94],
                format_func=lambda x: f"λ = {x:.2f} {'(RiskMetrics Default)' if x == 0.94 else ''}",
                help="EWMA weights recent squared returns with weight (1 - λ). Default is 0.94."
            )
            # Optional custom lambda input
            add_custom_lambda = st.checkbox("Add Custom λ", value=False)
            if add_custom_lambda:
                custom_lam = st.number_input("Custom λ:", min_value=0.50, max_value=0.999, value=0.95, step=0.01, format="%.3f")
                if custom_lam not in selected_lambdas:
                    selected_lambdas = sorted(list(set(selected_lambdas + [custom_lam])))

        with col_est2:
            st.markdown("##### 📏 Equally Weighted Rolling Look-Back Windows")
            standard_windows = [10, 20, 30, 45, 60, 90, 120, 180, 252]
            selected_windows = st.multiselect(
                "Select Rolling Window(s) (Days):",
                options=standard_windows,
                default=[60],
                format_func=lambda x: f"{x} Days {'(Default 60d)' if x == 60 else ''}",
                help="Equally weighted historical rolling sample standard deviation window."
            )
            # Optional custom window input
            add_custom_window = st.checkbox("Add Custom Window (Days)", value=False)
            if add_custom_window:
                custom_w = st.number_input("Custom Window (Days):", min_value=3, max_value=504, value=40, step=5)
                if custom_w not in selected_windows:
                    selected_windows = sorted(list(set(selected_windows + [int(custom_w)])))

        with col_est3:
            st.markdown("##### 🎨 Display Options")
            overlay_price = st.checkbox("Overlay Share Price", value=True, help="Display stock price on secondary y-axis.")
            annualize_vol = st.checkbox("Annualize Volatility (×√252)", value=True, help="Scale daily volatility to annual percentage.")
            show_mean_line = st.checkbox("Show Horizon Baseline Mean", value=True)

    # Ensure at least one estimator is selected
    if not selected_lambdas and not selected_windows:
        st.warning("Please select at least one EWMA parameter or one Rolling window from the configuration panel above.")
        selected_lambdas = [0.94]
        selected_windows = [60]

    # -------------------------------------------------------------------------
    # 3. Date Slicing & Return Calculation with Warm-Up Protection
    # -------------------------------------------------------------------------
    max_date = prices_gbp.index.max()
    if horizon_choice == "1 Month":
        start_date = max_date - pd.Timedelta(days=31)
    elif horizon_choice == "3 Months":
        start_date = max_date - pd.Timedelta(days=92)
    elif horizon_choice == "6 Months":
        start_date = max_date - pd.Timedelta(days=183)
    elif horizon_choice == "1 Year (Default)":
        start_date = max_date - pd.Timedelta(days=365)
    elif horizon_choice == "2 Years":
        start_date = max_date - pd.Timedelta(days=730)
    elif horizon_choice == "All Available":
        start_date = prices_gbp.index.min()
    else:  # Custom Range
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input("Start Date:", value=max_date - pd.Timedelta(days=365))
        with col_d2:
            end_date = st.date_input("End Date:", value=max_date)
        start_date = pd.to_datetime(start_date)
        max_date = pd.to_datetime(end_date)

    # Full history returns for smooth warm-up calculation
    full_log_returns = np.log(prices_gbp[selected_tickers] / prices_gbp[selected_tickers].shift(1)).dropna(how="all")
    filtered_prices = prices_gbp.loc[start_date:max_date, selected_tickers].dropna(how="all")

    if filtered_prices.empty or len(filtered_prices) < 5:
        st.warning("Insufficient price observations in the selected date range.")
        return

    trading_days = 252 if annualize_vol else 1
    vol_unit_label = "Annualized Volatility (%)" if annualize_vol else "Daily Volatility (%)"

    # -------------------------------------------------------------------------
    # 4. Multi-Estimator Series Computation Engine
    # -------------------------------------------------------------------------
    # Structure: stock_data[ticker][estimator_name] = { 'vol': Series, 'scaling': Series, 'type': str, 'param': val }
    stock_data: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for ticker in selected_tickers:
        if ticker not in full_log_returns.columns:
            continue
        r_full = full_log_returns[ticker].dropna()
        if len(r_full) < 5:
            continue

        estimators_dict: Dict[str, Dict[str, Any]] = {}

        # 1. EWMA Estimators
        for lam in selected_lambdas:
            name = f"EWMA (λ={lam:.2f})"
            v_full = calculate_ewma_volatility(r_full, decay_factor=lam, annualize=annualize_vol, trading_days=trading_days)
            s_full = calculate_scaling_factors(v_full)
            estimators_dict[name] = {
                "vol": v_full.loc[start_date:max_date],
                "scaling": s_full.loc[start_date:max_date],
                "type": "EWMA",
                "param": lam,
                "label": name
            }

        # 2. Equally Weighted Rolling Sample Volatility Estimators
        for w in selected_windows:
            name = f"Rolling ({w}d)"
            v_full = calculate_sample_volatility(r_full, window=w, annualize=annualize_vol, trading_days=trading_days)
            s_full = calculate_scaling_factors(v_full)
            estimators_dict[name] = {
                "vol": v_full.loc[start_date:max_date],
                "scaling": s_full.loc[start_date:max_date],
                "type": "Rolling",
                "param": w,
                "label": name
            }

        stock_data[ticker] = estimators_dict

    if not stock_data:
        st.warning("No valid volatility calculations available for the selected stocks.")
        return

    # -------------------------------------------------------------------------
    # 5. KPI Summary Cards: EWMA vs Rolling 60d Comparison Snapshot
    # -------------------------------------------------------------------------
    st.markdown("#### 📌 Current Volatility Snapshot & Estimator Comparison")

    kpi_cols = st.columns(len(selected_tickers))

    for idx, ticker in enumerate(selected_tickers):
        if ticker not in stock_data:
            continue
        est_dict = stock_data[ticker]

        # Primary EWMA (0.94 or first available)
        primary_ewma_name = next((k for k in est_dict if "0.94" in k), next((k for k in est_dict if "EWMA" in k), list(est_dict.keys())[0]))
        # Primary Rolling (60d or first available)
        primary_roll_name = next((k for k in est_dict if "60d" in k), next((k for k in est_dict if "Rolling" in k), list(est_dict.keys())[0]))

        v_ewma = est_dict[primary_ewma_name]["vol"]
        v_roll = est_dict[primary_roll_name]["vol"]

        latest_ewma_val = float(v_ewma.iloc[-1]) * 100.0
        latest_roll_val = float(v_roll.iloc[-1]) * 100.0
        ewma_pct_rank = float((v_ewma <= v_ewma.iloc[-1]).mean() * 100.0)

        # Volatility ratio (EWMA / Rolling 60d)
        vol_ratio = latest_ewma_val / latest_roll_val if latest_roll_val > 0 else 1.0
        ratio_label = "Calmed (< 60d Mean)" if vol_ratio < 0.95 else ("Spike (> 60d Mean)" if vol_ratio > 1.05 else "In-Line with 60d")

        with kpi_cols[idx % len(kpi_cols)]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{ticker} — Volatility Estimator Comparison</div>
                    <div class="metric-value">{latest_ewma_val:.2f}% <span style="font-size:0.95rem; font-weight:600; color:#059669;">vs {latest_roll_val:.2f}% (60d)</span></div>
                    <div class="metric-delta">
                        EWMA ({primary_ewma_name}): <b>{latest_ewma_val:.2f}%</b> (Rank: <b>{ewma_pct_rank:.1f}%</b>)<br>
                        Ratio (EWMA / 60d): <b>{vol_ratio:.2f}x</b> ({ratio_label})
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # -------------------------------------------------------------------------
    # 6. Main Volatility Trajectory Chart (All Estimators Overlaid)
    # -------------------------------------------------------------------------
    st.markdown("#### 📈 Dynamic Volatility Trajectory: EWMA vs Rolling Sample Estimators")
    st.caption("Compares the time-series paths of RiskMetrics EWMA and equally weighted rolling standard deviations on the institutional grey background.")

    for idx, ticker in enumerate(selected_tickers):
        if ticker not in stock_data:
            continue
        est_dict = stock_data[ticker]
        prices = filtered_prices[ticker].dropna()

        fig_vol = make_subplots(specs=[[{"secondary_y": overlay_price}]])

        # Add traces for all configured estimators
        color_idx = 0
        for name, data in est_dict.items():
            v_series = data["vol"]
            est_type = data["type"]
            param = data["param"]
            
            c = ESTIMATOR_COLORS[color_idx % len(ESTIMATOR_COLORS)]
            dash_style = "solid" if (est_type == "EWMA" and param == 0.94) else ("dash" if (est_type == "Rolling" and param == 60) else "dot")
            lw = 2.6 if (param in [0.94, 60]) else 1.8

            fig_vol.add_trace(
                go.Scatter(
                    x=v_series.index,
                    y=v_series.values * 100.0,
                    name=name,
                    line=dict(color=c, width=lw, dash=dash_style),
                    hovertemplate=f"<b>{name}</b>: <b>%{{y:.2f}}%</b><extra></extra>"
                ),
                secondary_y=False
            )
            color_idx += 1

        # Period baseline mean line
        if show_mean_line:
            primary_v = est_dict[primary_ewma_name]["vol"]
            mean_val = float(primary_v.mean()) * 100.0
            fig_vol.add_trace(
                go.Scatter(
                    x=[primary_v.index[0], primary_v.index[-1]],
                    y=[mean_val, mean_val],
                    name=f"1Y Mean EWMA Baseline ({mean_val:.1f}%)",
                    line=dict(color="#64748B", width=1.3, dash="dash"),
                    hoverinfo="skip"
                ),
                secondary_y=False
            )

        # Secondary Price Overlay
        if overlay_price and not prices.empty:
            fig_vol.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=prices.values,
                    name=f"{ticker} Share Price (GBP)",
                    line=dict(color="#64748B", width=1.4, dash="dashdot"),
                    opacity=0.6,
                    hovertemplate="Price: £%{y:,.2f}<extra></extra>"
                ),
                secondary_y=True
            )

        layout_vol = get_plotly_layout_defaults()
        layout_vol.update(dict(
            title=dict(text=f"<b>{ticker}</b> — Volatility Estimator Comparison ({horizon_choice})", font=dict(size=14, color="#0F172A")),
            height=430,
        ))
        fig_vol.update_layout(**layout_vol)
        fig_vol.update_yaxes(title_text=vol_unit_label, ticksuffix="%", secondary_y=False)
        if overlay_price:
            fig_vol.update_yaxes(title_text="Share Price (£)", tickprefix="£", secondary_y=True, showgrid=False)

        st.plotly_chart(fig_vol, use_container_width=True)

    # -------------------------------------------------------------------------
    # 7. Volatility Scaling Factors Multiplier Comparison Chart
    # -------------------------------------------------------------------------
    st.markdown(r"#### ⚡ Volatility Scaling Multipliers Comparison ($\sigma_{\text{today}} / \sigma_t$)")
    st.caption("Compares the historical return scaling multipliers generated by EWMA vs Rolling Sample Volatilities. Multipliers < 1.0 (Green) damp historical tail shocks; multipliers > 1.0 (Red) amplify historical tail shocks.")

    for idx, ticker in enumerate(selected_tickers):
        if ticker not in stock_data:
            continue
        est_dict = stock_data[ticker]

        fig_scale = go.Figure()
        color_idx = 0

        for name, data in est_dict.items():
            s_series = data["scaling"]
            est_type = data["type"]
            param = data["param"]

            c = ESTIMATOR_COLORS[color_idx % len(ESTIMATOR_COLORS)]
            dash_style = "solid" if (est_type == "EWMA" and param == 0.94) else ("dash" if (est_type == "Rolling" and param == 60) else "dot")
            lw = 2.4 if (param in [0.94, 60]) else 1.8

            fig_scale.add_trace(
                go.Scatter(
                    x=s_series.index,
                    y=s_series.values,
                    name=f"{name} Scaling Ratio",
                    line=dict(color=c, width=lw, dash=dash_style),
                    hovertemplate=f"<b>{name}</b> Multiplier: <b>%{{y:.2f}}x</b><extra></extra>"
                )
            )
            color_idx += 1

        # Neutral 1.0x line
        fig_scale.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="#DC2626",
            line_width=1.5,
            annotation_text="Neutral 1.0x",
            annotation_position="top right"
        )

        # Shaded Dampened and Amplified Zones
        fig_scale.add_hrect(
            y0=0.0, y1=1.0,
            fillcolor="#DCFCE7",
            opacity=0.4,
            line_width=0,
            annotation_text="Dampened Shock Region (< 1.0x)",
            annotation_position="bottom left"
        )
        fig_scale.add_hrect(
            y0=1.0, y1=2.5,
            fillcolor="#FEE2E2",
            opacity=0.3,
            line_width=0,
            annotation_text="Amplified Shock Region (> 1.0x)",
            annotation_position="top left"
        )

        layout_scale = get_plotly_layout_defaults()
        layout_scale.update(dict(
            title=dict(text=f"<b>{ticker}</b> — Scaling Factors (σ_today / σ_t) Comparison Across Estimators", font=dict(size=14, color="#0F172A")),
            height=400
        ))
        fig_scale.update_layout(**layout_scale)
        fig_scale.update_yaxes(title_text="Scaling Multiplier (σ_today / σ_t)", tickformat=".2f")
        st.plotly_chart(fig_scale, use_container_width=True)

    # -------------------------------------------------------------------------
    # 8. Estimator Spread & Ratio Analysis (EWMA vs Rolling 60d)
    # -------------------------------------------------------------------------
    with st.expander("📊 View Estimator Ratio & Spread Dynamics (EWMA / Rolling 60d)", expanded=False):
        st.caption("Analyzes the relative divergence between short-memory EWMA volatility and medium-memory 60-day rolling sample volatility.")
        
        for ticker in selected_tickers:
            if ticker not in stock_data:
                continue
            est_dict = stock_data[ticker]
            
            p_ewma = next((k for k in est_dict if "0.94" in k), next((k for k in est_dict if "EWMA" in k), None))
            p_roll = next((k for k in est_dict if "60d" in k), next((k for k in est_dict if "Rolling" in k), None))

            if p_ewma and p_roll:
                v_ew = est_dict[p_ewma]["vol"]
                v_ro = est_dict[p_roll]["vol"]
                ratio_series = (v_ew / v_ro).dropna()
                spread_series = (v_ew - v_ro) * 100.0

                fig_ratio = go.Figure()
                fig_ratio.add_trace(
                    go.Scatter(
                        x=ratio_series.index,
                        y=ratio_series.values,
                        name=f"Volatility Ratio ({p_ewma} / {p_roll})",
                        line=dict(color="#1E3A8A", width=2.2),
                        hovertemplate="Ratio: <b>%{y:.2f}x</b><extra></extra>"
                    )
                )
                fig_ratio.add_hline(y=1.0, line_dash="dash", line_color="#64748B", line_width=1.3, annotation_text="Parity 1.0x")
                fig_ratio.add_hrect(y0=0.0, y1=1.0, fillcolor="#DCFCE7", opacity=0.35, line_width=0, annotation_text="Calming Regime (EWMA < 60d)")
                fig_ratio.add_hrect(y0=1.0, y1=2.5, fillcolor="#FEE2E2", opacity=0.25, line_width=0, annotation_text="Volatility Spike Regime (EWMA > 60d)")

                layout_r = get_plotly_layout_defaults()
                layout_r.update(dict(
                    title=dict(text=f"<b>{ticker}</b> — Volatility Ratio ({p_ewma} / {p_roll})", font=dict(size=13, color="#0F172A")),
                    height=330
                ))
                fig_ratio.update_layout(**layout_r)
                fig_ratio.update_yaxes(title_text="Ratio (EWMA / Rolling 60d)", tickformat=".2f")
                st.plotly_chart(fig_ratio, use_container_width=True)

    # -------------------------------------------------------------------------
    # 9. Comprehensive Multi-Estimator Statistics Comparison Table
    # -------------------------------------------------------------------------
    st.markdown("#### 📋 Detailed Estimator Summary Statistics & Scaling Metrics")
    table_rows = []

    for ticker in selected_tickers:
        if ticker not in stock_data:
            continue
        est_dict = stock_data[ticker]

        for name, data in est_dict.items():
            v_s = data["vol"]
            s_s = data["scaling"]

            latest_v = float(v_s.iloc[-1]) * 100.0
            mean_v = float(v_s.mean()) * 100.0
            med_v = float(v_s.median()) * 100.0
            min_v = float(v_s.min()) * 100.0
            max_v = float(v_s.max()) * 100.0
            pct_rank = float((v_s <= v_s.iloc[-1]).mean() * 100.0)

            mean_scale = float(s_s.mean())
            min_scale = float(s_s.min())
            max_scale = float(s_s.max())
            trough_damp_pct = (1.0 - min_scale) * 100.0 if min_scale < 1.0 else 0.0

            table_rows.append({
                "Ticker": ticker,
                "Estimator": name,
                "Latest Vol": f"{latest_v:.2f}%",
                "Horizon Mean": f"{mean_v:.2f}%",
                "Horizon Median": f"{med_v:.2f}%",
                "Horizon Min": f"{min_v:.2f}%",
                "Horizon Max": f"{max_v:.2f}%",
                "Percentile Rank": f"{pct_rank:.1f}%",
                "Mean Scaling (σ_t / σ)": f"{mean_scale:.2f}x",
                "Min Scaling (Max Damp)": f"{min_scale:.2f}x (-{trough_damp_pct:.1f}%)",
                "Max Scaling (Max Amp)": f"{max_scale:.2f}x"
            })

    if table_rows:
        summary_df = pd.DataFrame(table_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
