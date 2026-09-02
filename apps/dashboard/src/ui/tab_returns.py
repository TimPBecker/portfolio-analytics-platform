"""
Tab 3: Stock Price Levels, Returns & Distribution Histogram View.
Interactive querying of individual asset price trajectories, daily return series,
empirical histograms with fitted KDE/Normal/Student-t densities, and statistical normality diagnostics.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from portfolio_core.analytics.statistics import (
    compute_asset_returns,
    compute_distribution_metrics,
    generate_density_curves,
    compute_qq_plot_data
)
from portfolio_core.db import fetch_raw_asset_prices
from sqlalchemy.engine import Engine

try:
    from src.ui.theme import PALETTE, get_plotly_layout_defaults
except ImportError:
    from apps.dashboard.src.ui.theme import PALETTE, get_plotly_layout_defaults


def render_tab_returns(
    prices_gbp: pd.DataFrame,
    available_tickers: List[str],
    engine: Optional[Engine] = None
):
    """Renders the Price Levels, Returns and Distribution Histogram view."""
    st.markdown("### 📊 Asset Levels, Returns & Distribution Diagnostics")
    st.caption("Inspect stock price levels, daily return time-series, empirical histograms with fitted probability density functions, and tail-fatness diagnostics.")

    if not available_tickers:
        st.warning("No tickers available in the database.")
        return

    # -------------------------------------------------------------------------
    # 1. Controls Bar
    # -------------------------------------------------------------------------
    col_c1, col_c2, col_c3, col_c4 = st.columns([1.5, 1.2, 1.2, 1.1])

    with col_c1:
        default_idx = available_tickers.index("NVDA") if "NVDA" in available_tickers else 0
        selected_ticker = st.selectbox("Select Stock Ticker:", options=available_tickers, index=default_idx)

    with col_c2:
        horizon_options = ["1 Month", "3 Months", "6 Months", "1 Year (Default)", "2 Years", "All Available", "Custom Range"]
        horizon_choice = st.selectbox("Time Horizon (Returns):", horizon_options, index=3)

    with col_c3:
        currency_mode = st.selectbox("Currency Display:", ["GBP Converted (£)", "Native Currency"], index=0)

    with col_c4:
        return_type = st.selectbox("Return Type:", ["Log Returns (ln)", "Simple Returns (%)"], index=0)

    # Fetch raw data for selected ticker
    raw_df = fetch_raw_asset_prices(selected_ticker, engine=engine)

    if raw_df.empty:
        st.warning(f"No price records found for ticker {selected_ticker}.")
        return

    raw_df = raw_df.set_index("DATE").sort_index()

    # Determine date slicing
    max_date = raw_df.index.max()
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
        start_date = raw_df.index.min()
    else:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input("Start Date (Asset):", value=max_date - pd.Timedelta(days=365))
        with col_d2:
            end_date = st.date_input("End Date (Asset):", value=max_date)
        start_date = pd.to_datetime(start_date)
        max_date = pd.to_datetime(end_date)

    filtered_df = raw_df.loc[start_date:max_date].copy()
    if len(filtered_df) < 5:
        st.warning("Insufficient price observations for the selected horizon.")
        return

    # Choose price series
    native_currency = filtered_df["CURRENCY"].iloc[-1] if "CURRENCY" in filtered_df.columns else "USD"
    if currency_mode.startswith("GBP"):
        price_series = filtered_df["CLOSE_GBP"]
        price_unit = "£"
        curr_label = "GBP"
    else:
        price_series = filtered_df["CLOSE"]
        price_unit = "" if native_currency in ["GBp", "GBX"] else ("$" if native_currency == "USD" else "€")
        curr_label = native_currency

    # Calculate returns
    method_key = "log" if "Log" in return_type else "simple"
    returns_series = compute_asset_returns(price_series, method=method_key)

    # Compute moments and diagnostics
    dist_metrics = compute_distribution_metrics(returns_series)

    # -------------------------------------------------------------------------
    # 2. Key Statistical Moments Cards
    # -------------------------------------------------------------------------
    st.markdown(f"#### 📌 Distribution & Moment Diagnostics for **{selected_ticker}** ({horizon_choice})")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Latest Price</div>
                <div class="metric-value">{price_unit}{price_series.iloc[-1]:,.2f} <span style="font-size:0.9rem; color:#64748B;">{curr_label}</span></div>
                <div class="metric-delta">Observations: <b>{dist_metrics.get('count', len(returns_series))} days</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi2:
        ann_vol = dist_metrics.get("vol_annualized_pct", 0.0)
        daily_vol = dist_metrics.get("std_daily_pct", 0.0)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Annualized Volatility</div>
                <div class="metric-value">{ann_vol:.2f}%</div>
                <div class="metric-delta">Daily Std: <b>{daily_vol:.2f}%</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi3:
        skew = dist_metrics.get("skewness", 0.0)
        skew_label = "Left-skewed (Negative)" if skew < -0.2 else ("Right-skewed (Positive)" if skew > 0.2 else "Symmetric")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Skewness</div>
                <div class="metric-value">{skew:+.2f}</div>
                <div class="metric-delta"><b>{skew_label}</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi4:
        kurt = dist_metrics.get("kurtosis_excess", 0.0)
        kurt_label = "Fat-Tailed (Leptokurtic)" if kurt > 0.5 else ("Thin-Tailed (Platykurtic)" if kurt < -0.5 else "Mesokurtic")
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Excess Kurtosis</div>
                <div class="metric-value">{kurt:+.2f}</div>
                <div class="metric-delta"><b>{kurt_label}</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi5:
        is_norm = dist_metrics.get("is_normal", False)
        jb_p = dist_metrics.get("jb_pvalue", 0.0)
        norm_label = "Normal (p > 0.05)" if is_norm else "Non-Normal (p < 0.01)"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Normality (Jarque-Bera)</div>
                <div class="metric-value">{"Pass" if is_norm else "Reject"}</div>
                <div class="metric-delta">p-value: <b>{jb_p:.2e}</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------------------------------
    # 3. Stock Level Chart with Moving Averages & Volume
    # -------------------------------------------------------------------------
    st.markdown("#### 📈 Price Level Trajectory & Moving Averages")

    has_volume = "VOLUME" in filtered_df.columns and filtered_df["VOLUME"].sum() > 0

    if has_volume:
        fig_price = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.75, 0.25]
        )
    else:
        fig_price = go.Figure()

    # Calculate Moving Averages
    sma_20 = price_series.rolling(20, min_periods=5).mean()
    sma_50 = price_series.rolling(50, min_periods=10).mean()
    sma_200 = price_series.rolling(200, min_periods=20).mean()

    trace_price = go.Scatter(
        x=price_series.index, y=price_series.values,
        name=f"{selected_ticker} Close ({curr_label})",
        line=dict(color="#1E3A8A", width=2.4),
        hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>Price: {price_unit}%{{y:,.2f}} {curr_label}<extra></extra>"
    )
    trace_sma20 = go.Scatter(x=sma_20.index, y=sma_20.values, name="SMA 20d", line=dict(color="#3B82F6", width=1.4, dash="dot"))
    trace_sma50 = go.Scatter(x=sma_50.index, y=sma_50.values, name="SMA 50d", line=dict(color="#D97706", width=1.4, dash="dash"))
    trace_sma200 = go.Scatter(x=sma_200.index, y=sma_200.values, name="SMA 200d", line=dict(color="#DC2626", width=1.4, dash="solid"))

    if has_volume:
        fig_price.add_trace(trace_price, row=1, col=1)
        fig_price.add_trace(trace_sma20, row=1, col=1)
        fig_price.add_trace(trace_sma50, row=1, col=1)
        fig_price.add_trace(trace_sma200, row=1, col=1)

        # Volume bars
        colors_vol = ["#10B981" if r >= 0 else "#EF4444" for r in returns_series]
        fig_price.add_trace(
            go.Bar(
                x=filtered_df.index[1:], y=filtered_df["VOLUME"].iloc[1:],
                name="Trading Volume",
                marker_color="#94A3B8",
                opacity=0.6
            ),
            row=2, col=1
        )
    else:
        fig_price.add_trace(trace_price)
        fig_price.add_trace(trace_sma20)
        fig_price.add_trace(trace_sma50)
        fig_price.add_trace(trace_sma200)

    layout_p = get_plotly_layout_defaults()
    layout_p.update(dict(
        title=dict(text=f"<b>{selected_ticker}</b> — Price History ({curr_label})", font=dict(size=14, color="#0F172A")),
        height=450
    ))
    fig_price.update_layout(**layout_p)
    if has_volume:
        fig_price.update_yaxes(title_text=f"Price ({curr_label})", row=1, col=1)
        fig_price.update_yaxes(title_text="Volume", row=2, col=1)
    else:
        fig_price.update_yaxes(title_text=f"Price ({curr_label})")

    st.plotly_chart(fig_price, use_container_width=True)

    # -------------------------------------------------------------------------
    # 4. Daily Returns Series & Volatility Confidence Bands
    # -------------------------------------------------------------------------
    st.markdown("#### 🌊 Daily Return Time Series & Confidence Bands")

    fig_rets = go.Figure()

    std_val = float(returns_series.std())
    mean_val = float(returns_series.mean())

    # Return bars / line
    fig_rets.add_trace(
        go.Bar(
            x=returns_series.index,
            y=returns_series.values * 100.0,
            name="Daily Return (%)",
            marker_color=["#10B981" if r >= 0 else "#EF4444" for r in returns_series],
            opacity=0.85,
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Daily Return: <b>%{y:+.2f}%</b><extra></extra>"
        )
    )

    # Confidence bands (+/- 2 std dev)
    fig_rets.add_hline(y=(mean_val + 2 * std_val) * 100.0, line_dash="dash", line_color="#D97706", line_width=1.2, annotation_text="+2σ (+{:.1f}%)".format((mean_val + 2 * std_val)*100), annotation_position="top right")
    fig_rets.add_hline(y=(mean_val - 2 * std_val) * 100.0, line_dash="dash", line_color="#D97706", line_width=1.2, annotation_text="-2σ ({:.1f}%)".format((mean_val - 2 * std_val)*100), annotation_position="bottom right")
    fig_rets.add_hline(y=0.0, line_color="#64748B", line_width=1.0)

    layout_rets = get_plotly_layout_defaults()
    layout_rets.update(dict(
        title=dict(text=f"<b>{selected_ticker}</b> — Daily Returns Time Series ({'Log' if method_key == 'log' else 'Percentage'})", font=dict(size=14, color="#0F172A")),
        height=350
    ))
    fig_rets.update_layout(**layout_rets)
    fig_rets.update_yaxes(title_text="Daily Return (%)", ticksuffix="%")
    st.plotly_chart(fig_rets, use_container_width=True)

    # -------------------------------------------------------------------------
    # 5. Return Distribution Histogram & Fitted Probability Densities
    # -------------------------------------------------------------------------
    st.markdown("#### 📊 Empirical Return Distribution & Density Fitting")
    st.caption("Inspect the return frequency distribution with overlaid Kernel Density Estimation (KDE), Gaussian Normal fit, Student-t fit, and VaR cutoff thresholds.")

    col_h_ctrl1, col_h_ctrl2 = st.columns([2, 2])
    with col_h_ctrl1:
        num_bins = st.slider("Histogram Bins:", min_value=15, max_value=80, value=35, step=5)
    with col_h_ctrl2:
        density_overlays = st.multiselect(
            "Overlaid Density Curves:",
            options=["Kernel Density (KDE)", "Fitted Normal PDF", "Fitted Student-t PDF"],
            default=["Kernel Density (KDE)", "Fitted Normal PDF", "Fitted Student-t PDF"]
        )

    # Generate densities
    x_grid, kde_y, norm_y, t_y = generate_density_curves(returns_series)

    fig_density = go.Figure()

    # 1. Histogram (Probability Density normalized)
    fig_density.add_trace(
        go.Histogram(
            x=returns_series.values * 100.0,
            histnorm="probability density",
            name="Empirical Returns Histogram",
            nbinsx=num_bins,
            marker_color="#93C5FD",
            marker_line=dict(color="#1D4ED8", width=1.0),
            opacity=0.65,
            hovertemplate="Return Bin: %{x:.2f}%<br>Density: %{y:.4f}<extra></extra>"
        )
    )

    # 2. KDE Overlay
    if "Kernel Density (KDE)" in density_overlays and len(x_grid) > 0:
        fig_density.add_trace(
            go.Scatter(
                x=x_grid * 100.0, y=kde_y / 100.0,
                name="Kernel Density (KDE)",
                line=dict(color="#1E3A8A", width=2.6),
                hovertemplate="KDE Density: %{y:.4f}<extra></extra>"
            )
        )

    # 3. Fitted Normal PDF
    if "Fitted Normal PDF" in density_overlays and len(x_grid) > 0:
        fig_density.add_trace(
            go.Scatter(
                x=x_grid * 100.0, y=norm_y / 100.0,
                name=f"Fitted Normal (μ={dist_metrics.get('mean_daily_pct', 0):.2f}%, σ={dist_metrics.get('std_daily_pct', 0):.2f}%)",
                line=dict(color="#DC2626", width=2.2, dash="dash"),
                hovertemplate="Normal Fit Density: %{y:.4f}<extra></extra>"
            )
        )

    # 4. Fitted Student-t PDF
    if "Fitted Student-t PDF" in density_overlays and t_y is not None and len(x_grid) > 0:
        df_t = dist_metrics.get("student_t_df")
        df_t_str = f"ν={df_t:.1f}" if df_t else ""
        fig_density.add_trace(
            go.Scatter(
                x=x_grid * 100.0, y=t_y / 100.0,
                name=f"Fitted Student-t ({df_t_str})",
                line=dict(color="#10B981", width=2.2, dash="dot"),
                hovertemplate="Student-t Fit Density: %{y:.4f}<extra></extra>"
            )
        )

    # VaR Cutoffs lines
    p05 = dist_metrics.get("p05_pct", 0.0)
    p01 = dist_metrics.get("p01_pct", 0.0)

    fig_density.add_vline(x=p05, line_dash="dashdot", line_color="#F59E0B", line_width=1.8, annotation_text=f"95% VaR ({p05:.2f}%)", annotation_position="top left")
    fig_density.add_vline(x=p01, line_dash="dashdot", line_color="#DC2626", line_width=1.8, annotation_text=f"99% VaR ({p01:.2f}%)", annotation_position="top left")

    layout_dens = get_plotly_layout_defaults()
    layout_dens.update(dict(
        title=dict(text=f"<b>{selected_ticker}</b> — Empirical Return Histogram & Probability Density Fits", font=dict(size=14, color="#0F172A")),
        height=460,
    ))
    fig_density.update_layout(**layout_dens)
    fig_density.update_xaxes(title_text="Daily Return (%)", ticksuffix="%")
    fig_density.update_yaxes(title_text="Probability Density")
    st.plotly_chart(fig_density, use_container_width=True)

    # -------------------------------------------------------------------------
    # 6. Quantile-Quantile (Q-Q) Diagnostics Plot & Extreme Dates Table
    # -------------------------------------------------------------------------
    col_qq1, col_qq2 = st.columns([1.3, 1.7])

    with col_qq1:
        st.markdown("##### 🔬 Normal Quantile-Quantile (Q-Q) Plot")
        osm, osr, slope, intercept = compute_qq_plot_data(returns_series)

        if len(osm) > 0:
            fig_qq = go.Figure()
            fig_qq.add_trace(
                go.Scatter(
                    x=osm, y=osr * 100.0,
                    mode="markers",
                    name="Sample Quantiles",
                    marker=dict(color="#1E3A8A", size=5, opacity=0.75)
                )
            )
            # Reference 45-degree line
            x_line = np.linspace(osm.min(), osm.max(), 100)
            y_line = (slope * x_line + intercept) * 100.0
            fig_qq.add_trace(
                go.Scatter(
                    x=x_line, y=y_line,
                    mode="lines",
                    name="Normal Reference Line",
                    line=dict(color="#DC2626", width=1.5, dash="dash")
                )
            )
            layout_qq = get_plotly_layout_defaults()
            layout_qq.update(dict(
                title=dict(text="Normal Q-Q Plot (Tail Heaviness)", font=dict(size=12, color="#0F172A")),
                height=380,
                legend=dict(x=0.02, y=0.98, orientation="v")
            ))
            fig_qq.update_layout(**layout_qq)
            fig_qq.update_xaxes(title_text="Theoretical Normal Quantiles")
            fig_qq.update_yaxes(title_text="Sample Return Quantiles (%)", ticksuffix="%")
            st.plotly_chart(fig_qq, use_container_width=True)

    with col_qq2:
        st.markdown("##### ⚡ Top 5 Best & Worst Single-Day Return Shocks")
        best_5 = returns_series.sort_values(ascending=False).head(5)
        worst_5 = returns_series.sort_values(ascending=True).head(5)

        shock_records = []
        for d, r in best_5.items():
            d_str = pd.to_datetime(d).strftime("%Y-%m-%d")
            impact_str = f"{abs(r) / std_val:.1f}σ" if std_val > 0 else "-"
            shock_records.append({"Date": d_str, "Type": "Gain 🟢", "Daily Return": f"{r * 100.0:+.2f}%", "Magnitude": impact_str})
        for d, r in worst_5.items():
            d_str = pd.to_datetime(d).strftime("%Y-%m-%d")
            impact_str = f"{abs(r) / std_val:.1f}σ" if std_val > 0 else "-"
            shock_records.append({"Date": d_str, "Type": "Loss 🔴", "Daily Return": f"{r * 100.0:+.2f}%", "Magnitude": impact_str})

        st.dataframe(pd.DataFrame(shock_records), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # 7. Cross-Asset Return Correlation Matrix Heatmap
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 🔥 Cross-Asset Daily Return Correlation Heatmap")
    st.caption("Pearson correlation coefficients across multiple assets over the selected historical lookback window.")

    if not prices_gbp.empty and len(prices_gbp.columns) > 1:
        corr_col1, corr_col2 = st.columns([2.0, 1.0])
        with corr_col1:
            default_corr = [t for t in available_tickers if t in prices_gbp.columns][:15]
            corr_tickers = st.multiselect(
                "Select Assets for Correlation Matrix:",
                options=list(prices_gbp.columns),
                default=default_corr,
                key="tab_returns_corr_tickers"
            )
        with corr_col2:
            corr_window = st.selectbox(
                "Correlation Lookback Window:",
                options=["3 Months (63 Days)", "6 Months (126 Days)", "1 Year (260 Days)", "2 Years (520 Days)", "All Available"],
                index=2,
                key="tab_returns_corr_window"
            )

        if len(corr_tickers) >= 2:
            window_days = 260
            if "3 Months" in corr_window:
                window_days = 63
            elif "6 Months" in corr_window:
                window_days = 126
            elif "2 Years" in corr_window:
                window_days = 520
            elif "All Available" in corr_window:
                window_days = len(prices_gbp)

            p_sub = prices_gbp[corr_tickers].iloc[-window_days:]
            rets_sub = np.log(p_sub / p_sub.shift(1)).dropna()
            corr_matrix = rets_sub.corr()

            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Blues",
                labels=dict(x="Asset", y="Asset", color="Correlation")
            )
            layout_corr = get_plotly_layout_defaults()
            layout_corr.update(dict(
                height=min(650, max(380, len(corr_tickers) * 32 + 100)),
                margin=dict(l=40, r=40, t=30, b=40)
            ))
            fig_corr.update_layout(**layout_corr)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Please select at least 2 assets to generate the correlation matrix.")
    elif len(prices_gbp.columns) == 1:
        st.info("Single-asset mode active. Additional assets will appear here when more holdings or market tickers are loaded.")
