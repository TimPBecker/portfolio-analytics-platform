"""
Tab 7: VaR Backtesting, PIT eCDF Uniformity & Regulatory Coverage Diagnostics View.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sqlalchemy.engine import Engine

from portfolio_core.analytics.backtesting import (
    generate_portfolio_backtest_timeline,
    run_backtest_diagnostics,
    evaluate_binomial_outliers,
    evaluate_ecdf_uniformity,
    kupiec_pof_test,
    kupiec_independence_test,
    christoffersen_conditional_coverage_test
)
from portfolio_core.db import (
    fetch_available_var_dates,
    fetch_portfolio_positions
)

try:
    from src.ui.theme import PALETTE, get_plotly_layout_defaults
except ImportError:
    from apps.dashboard.src.ui.theme import PALETTE, get_plotly_layout_defaults


def render_tab_backtesting(
    prices_gbp: pd.DataFrame,
    positions: Dict[str, float],
    asof_date: Optional[str] = None,
    engine: Optional[Engine] = None
):
    """
    Renders the interactive VaR Backtesting & Model Validation view.
    Executes a 1-year rolling VaR and clean P&L generation on demand when triggered by the user.
    Provides PIT eCDF uniformity tests, binomial outlier evaluation with dynamic confidence levels,
    and Kupiec coverage and independence tests.
    """
    st.markdown("### 🔬 VaR Backtesting & Model Validation")
    st.caption(
        "Institutional backtesting suite: evaluate 1 year of daily Value-at-Risk against realized Clean P&L (zero position impact). "
        "Test Probability Integral Transform (PIT) uniformity, dynamic Binomial outlier distributions, and Kupiec coverage and independence."
    )

    if prices_gbp.empty or not positions:
        st.warning("Insufficient price history or active positions to run backtesting.")
        return

    # -------------------------------------------------------------------------
    # 1. On-Demand Simulation Configuration Controls
    # -------------------------------------------------------------------------
    col_c1, col_c2, col_c3 = st.columns([1.5, 1.2, 1.3])

    available_dates = fetch_available_var_dates(engine=engine)
    if not available_dates:
        available_dates = [str(d)[:10] for d in sorted(prices_gbp.index, reverse=True)]

    default_asof = str(asof_date)[:10] if asof_date else available_dates[0]
    date_idx = available_dates.index(default_asof) if default_asof in available_dates else 0

    with col_c1:
        selected_date = st.selectbox(
            "Portfolio As-Of Date:",
            options=available_dates,
            index=date_idx,
            help="Select the portfolio valuation date. Positions as of this date will be locked for the 1-year backtest."
        )

    with col_c2:
        backtest_window = st.selectbox(
            "Backtesting Window:",
            options=[252, 126, 504],
            index=0,
            format_func=lambda x: f"{x} days ({'1 Year' if x == 252 else ('6 Months' if x == 126 else '2 Years')})",
            help="Number of daily out-of-sample evaluation points."
        )

    with col_c3:
        lookback_scenarios = st.selectbox(
            "VaR Scenario Lookback:",
            options=[260, 520, 130],
            index=0,
            format_func=lambda x: f"{x} days ({'1 Market Year' if x == 260 else ('2 Years' if x == 520 else '6 Months')})",
            help="Number of historical daily return observations used in each rolling scenario simulation."
        )

    # Trigger Button for On-Demand Calculation
    col_btn, _ = st.columns([1.5, 2.5])
    with col_btn:
        run_clicked = st.button("🚀 Run 1-Year Backtest Simulation", type="primary", use_container_width=True)

    # Initialize or fetch session state cache
    cache_key = f"backtest_data_{selected_date}_{backtest_window}_{lookback_scenarios}"

    if run_clicked:
        with st.spinner(f"Generating {backtest_window} days of rolling VaRs, Clean P&L, and eCDFs on portfolio frozen at {selected_date}..."):
            # Reconstruct positions at selected date if available
            pos_target = None
            if engine is not None:
                try:
                    pos_target = fetch_portfolio_positions(asof_date=selected_date, engine=engine)
                except Exception:
                    pos_target = None
            if not pos_target:
                pos_target = dict(positions)

            timeline_df = generate_portfolio_backtest_timeline(
                positions=pos_target,
                prices_gbp=prices_gbp,
                asof_date=selected_date,
                backtest_days=backtest_window,
                lookback_days=lookback_scenarios,
                confidence_level=0.95,
                ewma_lambda=0.94
            )
            st.session_state["active_backtest_df"] = timeline_df
            st.session_state["active_backtest_key"] = cache_key
            st.session_state["active_backtest_date"] = selected_date
            st.session_state["active_backtest_positions"] = pos_target

    timeline_df = st.session_state.get("active_backtest_df")

    if timeline_df is None or timeline_df.empty:
        st.info("👆 Click **'Run 1-Year Backtest Simulation'** above to generate rolling VaRs, Clean P&L, and eCDF distributions on demand.")
        return

    active_btest_date = st.session_state.get("active_backtest_date", selected_date)
    st.markdown("<div style='margin-bottom: 1.0rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. Interactive Confidence Interval Slider & Model Selection
    # -------------------------------------------------------------------------
    st.markdown("#### ⚙️ Diagnostic Parameters")
    col_param1, col_param2 = st.columns([2.0, 2.0])

    with col_param1:
        target_conf = st.slider(
            "VaR Confidence Level (Recalculate Outliers & Coverage):",
            min_value=0.90,
            max_value=0.995,
            value=0.95,
            step=0.005,
            format="%.3f",
            help="Adjust the confidence level to instantly recompute outlier counts, binomial tests, and Kupiec coverage without re-simulating."
        )

    with col_param2:
        model_choice = st.radio(
            "Primary Risk Simulation Engine:",
            options=["Historical Simulation", "Vol-Scaled Simulation (EWMA λ=0.94)"],
            index=0,
            horizontal=True,
            help="Select which risk simulation model to evaluate across the diagnostic suite."
        )

    model_prefix = "HIST" if "Historical" in model_choice else "VOL_SCALED"

    # Fast on-the-fly diagnostic evaluation
    diagnostics = run_backtest_diagnostics(
        timeline_df=timeline_df,
        confidence_level=target_conf,
        model_column_prefix=model_prefix
    )

    uniformity_res = diagnostics["uniformity"]
    loss_stats = diagnostics.get("loss_tail", {})
    gain_stats = diagnostics.get("gain_tail", {})

    loss_binom = loss_stats.get("binomial", diagnostics.get("binomial"))
    loss_pof = loss_stats.get("kupiec_pof", diagnostics.get("kupiec_pof"))
    loss_ind = loss_stats.get("independence", diagnostics.get("independence"))
    loss_cc = loss_stats.get("conditional_coverage", diagnostics.get("conditional_coverage"))

    gain_binom = gain_stats.get("binomial", diagnostics.get("binomial"))
    gain_pof = gain_stats.get("kupiec_pof", diagnostics.get("kupiec_pof"))
    gain_ind = gain_stats.get("independence", diagnostics.get("independence"))
    gain_cc = gain_stats.get("conditional_coverage", diagnostics.get("conditional_coverage"))

    p_expected = 1.0 - target_conf

    # -------------------------------------------------------------------------
    # 3. High-Level Summary KPI Cards (Two-Tailed)
    # -------------------------------------------------------------------------
    st.markdown(
        f"#### 📊 Two-Tailed Diagnostic Summary (Confidence Level: **{target_conf * 100:.1f}%** | "
        f"Expected Tail Probability: **{p_expected * 100:.1f}%** each)"
    )
    st.caption(
        f"Portfolio Date: **{active_btest_date}** | Evaluation Window: **{loss_binom.total_observations} days** "
        f"({timeline_df['DATE'].min().strftime('%d %b %Y')} → {timeline_df['DATE'].max().strftime('%d %b %Y')}). "
        f"Both the downside loss tail (quantile {p_expected * 100:.1f}%) and upside gain tail (quantile {target_conf * 100:.1f}%) are backtested."
    )

    col_tail_loss, col_tail_gain = st.columns(2)

    with col_tail_loss:
        st.markdown("##### 🔻 Downside Risk / Loss Tail")
        c_l1, c_l2, c_l3, c_l4 = st.columns(4)
        with c_l1:
            st.metric(
                label="Loss Outliers",
                value=f"{loss_binom.observed_outliers} / {loss_binom.expected_outliers:.1f}",
                delta=f"{loss_binom.observed_failure_rate * 100:.2f}% observed",
                delta_color="inverse" if loss_binom.observed_outliers > loss_binom.expected_outliers * 1.5 else "normal",
                help=f"Days where realized Clean P&L breached Loss VaR (< £threshold)."
            )
        with c_l2:
            st.metric(
                label="Basel Zone",
                value=f"{loss_binom.basel_zone}",
                delta="Loss Tail Zone",
                help="Basel Committee traffic light classification for 250-day window."
            )
        with c_l3:
            p_status = "🟢 Pass" if loss_pof.is_accepted_5pct else "🔴 Fail"
            st.metric(
                label="Kupiec POF",
                value=p_status,
                delta=f"p: {loss_pof.p_value:.3f}",
                help=f"LR_uc: {loss_pof.lr_statistic:.2f} (Critical: 3.841)"
            )
        with c_l4:
            i_status = "🟢 Indep" if loss_ind.is_independent_5pct else "🔴 Clustered"
            st.metric(
                label="Independence",
                value=i_status,
                delta=f"p: {loss_ind.p_value:.3f}",
                help=f"LR_ind: {loss_ind.lr_statistic:.2f} (Critical: 3.841)"
            )

    with col_tail_gain:
        st.markdown("##### 🔺 Upside Surges / Gain Tail")
        c_g1, c_g2, c_g3, c_g4 = st.columns(4)
        with c_g1:
            st.metric(
                label="Gain Outliers",
                value=f"{gain_binom.observed_outliers} / {gain_binom.expected_outliers:.1f}",
                delta=f"{gain_binom.observed_failure_rate * 100:.2f}% observed",
                delta_color="normal" if gain_binom.is_acceptable_5pct else "inverse",
                help=f"Days where realized Clean P&L exceeded Gain VaR (> +£threshold)."
            )
        with c_g2:
            st.metric(
                label="Basel Zone",
                value=f"{gain_binom.basel_zone}",
                delta="Gain Tail Zone",
                help="Basel traffic light threshold applied to upside tail exceptions."
            )
        with c_g3:
            gp_status = "🟢 Pass" if gain_pof.is_accepted_5pct else "🔴 Fail"
            st.metric(
                label="Kupiec POF",
                value=gp_status,
                delta=f"p: {gain_pof.p_value:.3f}",
                help=f"LR_uc: {gain_pof.lr_statistic:.2f} (Critical: 3.841)"
            )
        with c_g4:
            gi_status = "🟢 Indep" if gain_ind.is_independent_5pct else "🔴 Clustered"
            st.metric(
                label="Independence",
                value=gi_status,
                delta=f"p: {gain_ind.p_value:.3f}",
                help=f"LR_ind: {gain_ind.lr_statistic:.2f} (Critical: 3.841)"
            )

    # Full distribution calibration banner
    u_hist = diagnostics.get("uniformity_hist") or (
        test_ecdf_uniformity(timeline_df["HIST_ECDF"].dropna()) if "HIST_ECDF" in timeline_df.columns else uniformity_res
    )
    u_vol = diagnostics.get("uniformity_vol") or (
        test_ecdf_uniformity(timeline_df["VOL_SCALED_ECDF"].dropna()) if "VOL_SCALED_ECDF" in timeline_df.columns else uniformity_res
    )

    u_hist_badge = "🟢 PASS (Uniform)" if u_hist.is_uniform_5pct else "🔴 BIAS DETECTED"
    u_vol_badge = "🟢 PASS (Uniform)" if u_vol.is_uniform_5pct else "🔴 BIAS DETECTED"
    st.info(
        f"**PIT Uniformity Test (Kolmogorov-Smirnov):** "
        f"🏛️ **Historical Simulation:** {u_hist_badge} ($D$ = **{u_hist.statistic:.4f}**, $p$ = **{u_hist.p_value:.4f}**) | "
        f"⚡ **Vol-Scaled VaR (EWMA):** {u_vol_badge} ($D$ = **{u_vol.statistic:.4f}**, $p$ = **{u_vol.p_value:.4f}**). "
        f"Evaluates whether daily empirical percentiles match the theoretical uniform distribution over [0, 1]."
    )

    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. Chart 1: Clean P&L vs Two-Tailed VaR Timeline with Outlier Markers
    # -------------------------------------------------------------------------
    st.markdown("#### 📈 Two-Tailed Value-at-Risk Timeline with Outliers")
    st.caption(
        "Realized Clean P&L (dots) plotted against both the Loss VaR threshold (lower tail) "
        "and Gain VaR threshold (upper tail). Outliers on both sides are highlighted with distinct markers."
    )

    var_col = f"{model_prefix}_VAR_GBP"
    gain_var_col = f"{model_prefix}_GAIN_VAR_GBP"

    # Loss VaR series
    dynamic_loss_var = loss_stats.get("dynamic_var")
    active_loss_var = dynamic_loss_var if dynamic_loss_var is not None else timeline_df[var_col]

    # Gain VaR series
    dynamic_gain_var = gain_stats.get("dynamic_var")
    if dynamic_gain_var is not None:
        active_gain_var = dynamic_gain_var
    elif gain_var_col in timeline_df.columns:
        active_gain_var = timeline_df[gain_var_col]
    else:
        active_gain_var = None

    # Exception indicator series
    loss_exceptions_series = loss_stats.get("exceptions_series")
    if loss_exceptions_series is not None:
        is_loss_breach = (loss_exceptions_series == 1)
    else:
        is_loss_breach = (timeline_df["CLEAN_PNL_GBP"] < active_loss_var)

    gain_exceptions_series = gain_stats.get("exceptions_series")
    if gain_exceptions_series is not None:
        is_gain_breach = (gain_exceptions_series == 1)
    elif active_gain_var is not None:
        is_gain_breach = (timeline_df["CLEAN_PNL_GBP"] > active_gain_var)
    else:
        is_gain_breach = pd.Series([False] * len(timeline_df), index=timeline_df.index)

    loss_breach_df = timeline_df[is_loss_breach]
    gain_breach_df = timeline_df[is_gain_breach]

    fig_timeline = go.Figure()

    # Clean P&L Dots (mode="markers")
    fig_timeline.add_trace(
        go.Scatter(
            x=timeline_df["DATE"],
            y=timeline_df["CLEAN_PNL_GBP"],
            name="Realized Clean P&L (£)",
            mode="markers",
            marker=dict(
                color=["#10B981" if v >= 0 else "#64748B" for v in timeline_df["CLEAN_PNL_GBP"]],
                size=6,
                opacity=0.85
            ),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Clean P&L: <b>£%{y:,.2f}</b><extra></extra>"
        )
    )

    # Loss VaR Line (Lower Tail)
    fig_timeline.add_trace(
        go.Scatter(
            x=timeline_df["DATE"],
            y=active_loss_var,
            name=f"Loss VaR (Lower Tail: {p_expected * 100:.1f}%)",
            line=dict(color="#DC2626", width=2.2, dash="solid"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Loss VaR Threshold: <b>£%{y:,.2f}</b><extra></extra>"
        )
    )

    # Gain VaR Line (Upper Tail)
    if active_gain_var is not None:
        fig_timeline.add_trace(
            go.Scatter(
                x=timeline_df["DATE"],
                y=active_gain_var,
                name=f"Gain VaR (Upper Tail: {target_conf * 100:.1f}%)",
                line=dict(color="#2563EB", width=2.2, dash="solid"),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Gain VaR Threshold: <b>£%{y:,.2f}</b><extra></extra>"
            )
        )

    # Zero P&L reference line
    fig_timeline.add_hline(y=0.0, line_dash="dot", line_color="#94A3B8", line_width=1.0)

    # Loss Outlier Markers (Breaches)
    if not loss_breach_df.empty:
        fig_timeline.add_trace(
            go.Scatter(
                x=loss_breach_df["DATE"],
                y=loss_breach_df["CLEAN_PNL_GBP"],
                name="Loss Outliers (< Loss VaR)",
                mode="markers",
                marker=dict(color="#DC2626", size=10, symbol="x", line=dict(width=2, color="#7F1D1D")),
                hovertemplate="<b>LOSS OUTLIER</b><br>%{x|%d %b %Y}<br>Loss: <b>£%{y:,.2f}</b><extra></extra>"
            )
        )

    # Gain Outlier Markers (Surges)
    if not gain_breach_df.empty:
        fig_timeline.add_trace(
            go.Scatter(
                x=gain_breach_df["DATE"],
                y=gain_breach_df["CLEAN_PNL_GBP"],
                name="Gain Outliers (> Gain VaR)",
                mode="markers",
                marker=dict(color="#2563EB", size=10, symbol="diamond", line=dict(width=2, color="#1E3A8A")),
                hovertemplate="<b>GAIN OUTLIER</b><br>%{x|%d %b %Y}<br>Gain: <b>£%{y:,.2f}</b><extra></extra>"
            )
        )

    layout_time = get_plotly_layout_defaults()
    layout_time.update(dict(
        title=dict(
            text=f"1-Year Two-Tailed Backtest: Clean P&L vs {p_expected * 100:.1f}% Loss VaR & {target_conf * 100:.1f}% Gain VaR (£)",
            font=dict(size=13, color="#0F172A")
        ),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    ))
    fig_timeline.update_layout(**layout_time)
    fig_timeline.update_yaxes(title_text="Clean P&L (£)", tickprefix="£")
    st.plotly_chart(fig_timeline, use_container_width=True)

    # -------------------------------------------------------------------------
    # 5. Separate PIT eCDF Diagnostics for Historical Simulation & Vol-Scaled VaR
    # -------------------------------------------------------------------------
    st.markdown("#### 🎯 Probability Integral Transform (PIT) Uniformity Tests")
    st.caption(
        "Side-by-side comparative diagnostics for both simulation models. "
        "A well-calibrated distribution produces a flat histogram (density = 1.0) and a cumulative CDF tracking the diagonal."
    )

    hist_ecdf_vals = timeline_df["HIST_ECDF"].dropna() if "HIST_ECDF" in timeline_df.columns else pd.Series(dtype=float)
    vol_ecdf_vals = timeline_df["VOL_SCALED_ECDF"].dropna() if "VOL_SCALED_ECDF" in timeline_df.columns else pd.Series(dtype=float)

    # Row 1: Two separate PIT Histograms side-by-side
    st.markdown("##### 📊 1. PIT Histograms: Historical Simulation vs Vol-Scaled VaR")
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        fig_h_hist = go.Figure()
        if not hist_ecdf_vals.empty:
            fig_h_hist.add_trace(
                go.Histogram(
                    x=hist_ecdf_vals,
                    nbinsx=10,
                    histnorm="probability density",
                    name="Historical Sim PIT",
                    marker=dict(color="#3B82F6", line=dict(color="#1E3A8A", width=1.0)),
                    opacity=0.75
                )
            )
        fig_h_hist.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="#DC2626",
            line_width=1.8,
            annotation_text="Theoretical Uniform Density = 1.0",
            annotation_position="top right"
        )
        l_h_hist = get_plotly_layout_defaults()
        l_h_hist.update(dict(
            title=dict(text=f"🏛️ Historical Sim PIT Histogram (KS p={u_hist.p_value:.4f})", font=dict(size=12, color="#0F172A")),
            height=320
        ))
        fig_h_hist.update_layout(**l_h_hist)
        fig_h_hist.update_xaxes(title_text="Empirical CDF Value (PIT)", range=[0.0, 1.0])
        fig_h_hist.update_yaxes(title_text="Density")
        st.plotly_chart(fig_h_hist, use_container_width=True)

    with col_h2:
        fig_v_hist = go.Figure()
        if not vol_ecdf_vals.empty:
            fig_v_hist.add_trace(
                go.Histogram(
                    x=vol_ecdf_vals,
                    nbinsx=10,
                    histnorm="probability density",
                    name="Vol-Scaled PIT",
                    marker=dict(color="#10B981", line=dict(color="#065F46", width=1.0)),
                    opacity=0.75
                )
            )
        fig_v_hist.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="#DC2626",
            line_width=1.8,
            annotation_text="Theoretical Uniform Density = 1.0",
            annotation_position="top right"
        )
        l_v_hist = get_plotly_layout_defaults()
        l_v_hist.update(dict(
            title=dict(text=f"⚡ Vol-Scaled Sim PIT Histogram (KS p={u_vol.p_value:.4f})", font=dict(size=12, color="#0F172A")),
            height=320
        ))
        fig_v_hist.update_layout(**l_v_hist)
        fig_v_hist.update_xaxes(title_text="Empirical CDF Value (PIT)", range=[0.0, 1.0])
        fig_v_hist.update_yaxes(title_text="Density")
        st.plotly_chart(fig_v_hist, use_container_width=True)

    # Row 2: Two separate Cumulative CDF Curves side-by-side
    st.markdown("##### 📐 2. Cumulative PIT Curves: Historical Simulation vs Vol-Scaled VaR")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        fig_h_cdf = go.Figure()
        if not hist_ecdf_vals.empty:
            sorted_h = np.sort(hist_ecdf_vals.values)
            emp_h_y = np.linspace(0.0, 1.0, len(sorted_h))
            fig_h_cdf.add_trace(
                go.Scatter(
                    x=sorted_h,
                    y=emp_h_y,
                    name="Empirical CDF (Hist)",
                    line=dict(color="#3B82F6", width=2.8),
                    hovertemplate="PIT: <b>%{x:.3f}</b><br>Cumulative: <b>%{y:.3f}</b><extra></extra>"
                )
            )
        fig_h_cdf.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                name="Theoretical Uniform(0,1)",
                line=dict(color="#64748B", width=2.0, dash="dash")
            )
        )
        l_h_cdf = get_plotly_layout_defaults()
        l_h_cdf.update(dict(
            title=dict(text=f"🏛️ Historical Sim Cumulative PIT (D={u_hist.statistic:.4f})", font=dict(size=12, color="#0F172A")),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        ))
        fig_h_cdf.update_layout(**l_h_cdf)
        fig_h_cdf.update_xaxes(title_text="Theoretical Quantile", range=[0.0, 1.0])
        fig_h_cdf.update_yaxes(title_text="Empirical Quantile", range=[0.0, 1.0])
        st.plotly_chart(fig_h_cdf, use_container_width=True)

    with col_c2:
        fig_v_cdf = go.Figure()
        if not vol_ecdf_vals.empty:
            sorted_v = np.sort(vol_ecdf_vals.values)
            emp_v_y = np.linspace(0.0, 1.0, len(sorted_v))
            fig_v_cdf.add_trace(
                go.Scatter(
                    x=sorted_v,
                    y=emp_v_y,
                    name="Empirical CDF (Vol-Scaled)",
                    line=dict(color="#10B981", width=2.8),
                    hovertemplate="PIT: <b>%{x:.3f}</b><br>Cumulative: <b>%{y:.3f}</b><extra></extra>"
                )
            )
        fig_v_cdf.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                name="Theoretical Uniform(0,1)",
                line=dict(color="#64748B", width=2.0, dash="dash")
            )
        )
        l_v_cdf = get_plotly_layout_defaults()
        l_v_cdf.update(dict(
            title=dict(text=f"⚡ Vol-Scaled Cumulative PIT (D={u_vol.statistic:.4f})", font=dict(size=12, color="#0F172A")),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        ))
        fig_v_cdf.update_layout(**l_v_cdf)
        fig_v_cdf.update_xaxes(title_text="Theoretical Quantile", range=[0.0, 1.0])
        fig_v_cdf.update_yaxes(title_text="Empirical Quantile", range=[0.0, 1.0])
        st.plotly_chart(fig_v_cdf, use_container_width=True)

    # -------------------------------------------------------------------------
    # 6. Detailed Analytics Tables (Two-Tailed Comparison & PIT Uniformity)
    # -------------------------------------------------------------------------
    st.markdown("#### 📋 Detailed Two-Tailed Backtesting Diagnostic Reports")

    col_t1, col_t2 = st.columns([1.5, 1.1])

    with col_t1:
        st.markdown("##### 1. Two-Tailed Comparative Backtesting Table")
        comp_data = [
            {"Metric / Test": "Evaluation Window (T)", "🔻 Loss Tail (Downside)": f"{loss_binom.total_observations} days", "🔺 Gain Tail (Upside)": f"{gain_binom.total_observations} days"},
            {"Metric / Test": "Target Quantile Threshold", "🔻 Loss Tail (Downside)": f"{p_expected * 100:.1f}% Lower Tail", "🔺 Gain Tail (Upside)": f"{target_conf * 100:.1f}% Upper Tail"},
            {"Metric / Test": "Expected Tail Probability (p)", "🔻 Loss Tail (Downside)": f"{p_expected * 100:.2f}%", "🔺 Gain Tail (Upside)": f"{p_expected * 100:.2f}%"},
            {"Metric / Test": "Expected Outlier Count", "🔻 Loss Tail (Downside)": f"{loss_binom.expected_outliers:.1f}", "🔺 Gain Tail (Upside)": f"{gain_binom.expected_outliers:.1f}"},
            {"Metric / Test": "Observed Outlier Count (x)", "🔻 Loss Tail (Downside)": f"{loss_binom.observed_outliers}", "🔺 Gain Tail (Upside)": f"{gain_binom.observed_outliers}"},
            {"Metric / Test": "Observed Failure Rate (p̂)", "🔻 Loss Tail (Downside)": f"{loss_binom.observed_failure_rate * 100:.2f}%", "🔺 Gain Tail (Upside)": f"{gain_binom.observed_failure_rate * 100:.2f}%"},
            {"Metric / Test": "Clopper-Pearson 95% CI", "🔻 Loss Tail (Downside)": f"[{loss_binom.ci_lower * 100:.2f}%, {loss_binom.ci_upper * 100:.2f}%]", "🔺 Gain Tail (Upside)": f"[{gain_binom.ci_lower * 100:.2f}%, {gain_binom.ci_upper * 100:.2f}%]"},
            {"Metric / Test": "Basel Traffic Light Zone", "🔻 Loss Tail (Downside)": f"{loss_binom.basel_zone}", "🔺 Gain Tail (Upside)": f"{gain_binom.basel_zone}"},
            {"Metric / Test": "Binomial Test p-value", "🔻 Loss Tail (Downside)": f"{loss_binom.p_value:.5f}", "🔺 Gain Tail (Upside)": f"{gain_binom.p_value:.5f}"},
            {"Metric / Test": "Kupiec POF LR_uc (Crit: 3.841)", "🔻 Loss Tail (Downside)": f"{loss_pof.lr_statistic:.3f} (p={loss_pof.p_value:.4f})", "🔺 Gain Tail (Upside)": f"{gain_pof.lr_statistic:.3f} (p={gain_pof.p_value:.4f})"},
            {"Metric / Test": "Kupiec POF Decision (5% Level)", "🔻 Loss Tail (Downside)": "ACCEPTED" if loss_pof.is_accepted_5pct else "REJECTED", "🔺 Gain Tail (Upside)": "ACCEPTED" if gain_pof.is_accepted_5pct else "REJECTED"},
            {"Metric / Test": "Independence LR_ind (Crit: 3.841)", "🔻 Loss Tail (Downside)": f"{loss_ind.lr_statistic:.3f} (p={loss_ind.p_value:.4f})", "🔺 Gain Tail (Upside)": f"{gain_ind.lr_statistic:.3f} (p={gain_ind.p_value:.4f})"},
            {"Metric / Test": "Independence Decision (No Clustering)", "🔻 Loss Tail (Downside)": "ACCEPTED" if loss_ind.is_independent_5pct else "REJECTED", "🔺 Gain Tail (Upside)": "ACCEPTED" if gain_ind.is_independent_5pct else "REJECTED"},
            {"Metric / Test": "Conditional Coverage LR_cc (Crit: 5.991)", "🔻 Loss Tail (Downside)": f"{loss_cc.lr_cc:.3f} (p={loss_cc.p_value:.4f})", "🔺 Gain Tail (Upside)": f"{gain_cc.lr_cc:.3f} (p={gain_cc.p_value:.4f})"},
            {"Metric / Test": "Conditional Coverage Decision", "🔻 Loss Tail (Downside)": "ACCEPTED" if loss_cc.is_accepted_5pct else "REJECTED", "🔺 Gain Tail (Upside)": "ACCEPTED" if gain_cc.is_accepted_5pct else "REJECTED"},
        ]
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    with col_t2:
        st.markdown("##### 2. PIT Uniformity Comparison (KS Test)")
        pit_comp_data = [
            {"Metric": "Sample Size (T)", "🏛️ Historical Sim": str(u_hist.sample_size), "⚡ Vol-Scaled Sim": str(u_vol.sample_size)},
            {"Metric": "KS Statistic (D)", "🏛️ Historical Sim": f"{u_hist.statistic:.4f}", "⚡ Vol-Scaled Sim": f"{u_vol.statistic:.4f}"},
            {"Metric": "Asymptotic p-value", "🏛️ Historical Sim": f"{u_hist.p_value:.5f}", "⚡ Vol-Scaled Sim": f"{u_vol.p_value:.5f}"},
            {"Metric": "Empirical Mean (Exp: 0.50)", "🏛️ Historical Sim": f"{u_hist.empirical_mean:.4f}", "⚡ Vol-Scaled Sim": f"{u_vol.empirical_mean:.4f}"},
            {"Metric": "Empirical Std (Exp: 0.289)", "🏛️ Historical Sim": f"{u_hist.empirical_std:.4f}", "⚡ Vol-Scaled Sim": f"{u_vol.empirical_std:.4f}"},
            {"Metric": "Uniformity (5% Level)", "🏛️ Historical Sim": "PASS (Uniform)" if u_hist.is_uniform_5pct else "REJECTED (Bias)", "⚡ Vol-Scaled Sim": "PASS (Uniform)" if u_vol.is_uniform_5pct else "REJECTED (Bias)"},
            {
                "Metric": "Calibration Assessment",
                "🏛️ Historical Sim": getattr(
                    u_hist,
                    "calibration_quality",
                    ("Acceptable Calibration" if getattr(u_hist, "is_uniform_5pct", True) else "Severe Miscalibration")
                ),
                "⚡ Vol-Scaled Sim": getattr(
                    u_vol,
                    "calibration_quality",
                    ("Acceptable Calibration" if getattr(u_vol, "is_uniform_5pct", True) else "Severe Miscalibration")
                ),
            },
        ]
        st.dataframe(pd.DataFrame(pit_comp_data), use_container_width=True, hide_index=True)

    with st.expander("ℹ️ Full Regulatory Methodology & Formulas", expanded=False):
        st.markdown(
            r"""
            ### Regulatory & Quantitative Foundations (Two-Tailed Backtesting):
            1. **Two-Tailed Risk Evaluation ($\alpha$ and $1 - \alpha$):**
               - **Loss Tail (Downside Risk):** Realized Clean P&L breaches the lower tail VaR threshold:
                 $$X_{\text{loss}, t} = \mathbf{1}_{\{\text{Clean P\&L}_t < \text{VaR}_{t}(1 - \alpha)\}} \iff U_t \le 1 - \alpha$$
               - **Gain Tail (Upside Surges / Model Symmetry):** Realized Clean P&L exceeds the upper tail VaR threshold:
                 $$X_{\text{gain}, t} = \mathbf{1}_{\{\text{Clean P\&L}_t > \text{VaR}_{t}(\alpha)\}} \iff U_t \ge \alpha$$
               - Under a well-calibrated distribution, both tails independently follow $X \sim \text{Binomial}(T, 1 - \alpha)$.
            2. **Probability Integral Transform (PIT) Uniformity:**
               If a model's scenario distribution $F_t$ matches the true data generating process, then $U_t = F_t(\text{Clean P\&L}_t)$ is i.i.d. $\text{Uniform}(0, 1)$.
               The **Kolmogorov-Smirnov (KS) test** evaluates the maximum vertical deviation $D = \sup_u |F_T(u) - u|$ against the theoretical uniform distribution.
            3. **Binomial Distribution Outlier Test:**
               Outliers $X$ follow a $\text{Binomial}(T, p)$ distribution, where $p = 1 - \text{confidence level}$ and $T$ is the number of evaluation days.
               Computes the exact two-sided Clopper-Pearson confidence interval and Basel Traffic Light classification (**Green**, **Yellow**, **Red**).
            4. **Kupiec Proportion of Failures (Unconditional Coverage, $LR_{uc}$):**
               $$LR_{uc} = 2 \left[ x \ln\left(\frac{\hat{p}}{p}\right) + (T - x) \ln\left(\frac{1 - \hat{p}}{1 - p}\right) \right] \sim \chi^2(1)$$
               Evaluated for both the loss tail and gain tail to test unconditional tail calibration.
            5. **Kupiec / Christoffersen Independence Test ($LR_{ind}$):**
               Models exceptions via a two-state Markov chain to test for temporal clustering:
               $$LR_{ind} = 2 \left[ \ln L_1(\hat{\Pi}) - \ln L_0(\hat{\pi}) \right] \sim \chi^2(1)$$
               Rejection indicates temporal clustering (volatility clustering or regime changes).
            6. **Christoffersen Conditional Coverage Test ($LR_{cc}$):**
               Omnibus test combining unconditional coverage and exception independence:
               $$LR_{cc} = LR_{uc} + LR_{ind} \sim \chi^2(2)$$
            """
        )
