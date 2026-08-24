"""
Tab 2: Value-at-Risk (VaR) & Tail Risk Spectrum View.
Database-First Architecture:
- By default, all VaR metrics, CVaR curves, and Shapley risk contributions
  are fetched directly from materialized database tables (PORTFOLIO_VAR, PORTFOLIO_RISK_CONTRIBUTIONS,
  PORTFOLIO_SCENARIO_PNL) with zero recalculation overhead.
- Live on-the-fly math is used seamlessly as a dynamic fallback only if data is not yet in the DB.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from portfolio_core.analytics.var import (
    compute_multi_model_var_spectrum,
    compute_standalone_asset_var,
    compute_shapley_risk_contributions,
    compute_portfolio_scenario_pnl
)
from portfolio_core.db import (
    fetch_stored_var_metrics,
    fetch_stored_risk_contributions,
    fetch_stored_scenario_pnl,
    fetch_available_var_dates
)
from src.ui.theme import PALETTE, get_plotly_layout_defaults


def render_tab_var(
    prices_gbp: pd.DataFrame,
    positions: Dict[str, float],
    asof_date: Optional[str] = None
):
    """Renders the Value-at-Risk and Risk Attribution view."""
    st.markdown("### 🛡️ Value-at-Risk (VaR) & Percentile Spectrum")
    st.caption("Inspect historical and volatility-scaled tail risk across every percentile (0.01 to 0.99) and component risk attribution.")

    if prices_gbp.empty or not positions:
        st.warning("Insufficient price data or active positions to compute Value-at-Risk.")
        return

    # -------------------------------------------------------------------------
    # 1. Top Controls
    # -------------------------------------------------------------------------
    col_c1, col_c2, col_c3 = st.columns([1.5, 1.2, 1.3])

    available_dates = fetch_available_var_dates()
    default_asof = asof_date or (available_dates[0] if available_dates else str(prices_gbp.index[-1])[:10])

    with col_c1:
        selected_asof = st.selectbox(
            "Reporting As-Of Date:",
            options=available_dates if available_dates else [default_asof],
            index=0,
            help="Select historical valuation date for risk calculations."
        )

    with col_c2:
        lookback_days = st.selectbox("Simulation Lookback:", [260, 520, 130, 780], index=0, help="Number of daily return observations (260 = 1 market year).")

    with col_c3:
        horizon_days = st.selectbox("Holding Horizon:", [1, 5, 10, 21], index=0, help="VaR time horizon in trading days.")

    # -------------------------------------------------------------------------
    # 2. Database-First VaR Spectrum Loading (Default: PORTFOLIO_VAR)
    # -------------------------------------------------------------------------
    stored_var_df = fetch_stored_var_metrics(asof_date=selected_asof)

    if not stored_var_df.empty:
        # --- PATH A: Materialized Database Records (No recalculation) ---
        spectrum_df = stored_var_df.copy()
        spectrum_df["METHOD"] = spectrum_df["METHOD"].replace({
            "Vol-Scaled VaR (EWMA Volatility (λ=0.94))": "Vol-Scaled VaR (EWMA λ=0.94)",
            "Vol-Scaled VaR (Sample 30d Volatility)": "Vol-Scaled VaR (Sample 30d)",
            "Vol-Scaled VaR (Sample 60d Volatility)": "Vol-Scaled VaR (Sample 60d)"
        })
        # Exclude parametric models
        spectrum_df = spectrum_df[~spectrum_df["METHOD"].str.startswith("Parametric")]
        total_portfolio_value = float(spectrum_df["PORTFOLIO_VALUE_GBP"].iloc[0])
        var_data_source = "🟢 Database Records (`PORTFOLIO_VAR`)"

        # Fetch materialized scenario PnL from DB
        scenario_pnl_df = fetch_stored_scenario_pnl(asof_date=selected_asof)
        if scenario_pnl_df.empty:
            _, scenario_pnl_df, _ = compute_multi_model_var_spectrum(
                price_history=prices_gbp, positions=positions, asof_date=selected_asof, lookback_days=lookback_days
            )
    else:
        # --- PATH B: Fallback Dynamic Live Math ---
        with st.spinner("Computing full percentile VaR spectrum..."):
            all_cls = [round(c, 2) for c in np.arange(0.01, 1.00, 0.01)]
            spectrum_df, scenario_pnl_df, pos_values = compute_multi_model_var_spectrum(
                price_history=prices_gbp,
                positions=positions,
                asof_date=selected_asof,
                lookback_days=lookback_days,
                ewma_lambda=0.94,
                horizon_days=horizon_days,
                confidence_levels=all_cls
            )
        total_portfolio_value = float(spectrum_df["PORTFOLIO_VALUE_GBP"].iloc[0]) if not spectrum_df.empty else 0.0
        var_data_source = "⚡ Live Computed (Dynamic Fallback)"

    if spectrum_df.empty:
        st.error("Failed to load or compute VaR spectrum. Check price history and active positions.")
        return

    # -------------------------------------------------------------------------
    # 3. Interactive Percentile Selector & Key Metrics
    # -------------------------------------------------------------------------
    st.markdown("#### 🎯 Quick Percentile Inspector")
    st.caption(f"Data Source: **{var_data_source}** | Valuation Date: **{selected_asof}**")

    col_p1, col_p2 = st.columns([2.5, 1.5])
    with col_p1:
        target_cl = st.slider(
            "Inspect Confidence Level (Percentile):",
            min_value=0.01,
            max_value=0.99,
            value=0.95,
            step=0.01,
            format="%.2f"
        )
    with col_p2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        btn_cols = st.columns(4)
        if btn_cols[0].button("90%"): target_cl = 0.90
        if btn_cols[1].button("95%"): target_cl = 0.95
        if btn_cols[2].button("99%"): target_cl = 0.99
        if btn_cols[3].button("99.5%"): target_cl = 0.995

    # Filter spectrum at target confidence level
    cl_match = round(target_cl, 2)
    current_metrics = spectrum_df[spectrum_df["CONFIDENCE_LEVEL"] == cl_match]

    # Render Metric Cards for target percentile
    st.markdown(f"##### Risk Snapshot at **{target_cl*100:.1f}% Confidence Level** (Portfolio Value: **£{total_portfolio_value:,.2f}**)")

    m_cols = st.columns(len(current_metrics))
    for idx, (_, row) in enumerate(current_metrics.iterrows()):
        m_name = row["METHOD"]
        v_gbp = row["VAR_GBP"]
        v_pct = row["VAR_PCT"]
        cv_gbp = row["CVAR_GBP"]
        cv_pct = row["CVAR_PCT"]

        with m_cols[idx % len(m_cols)]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{m_name}</div>
                    <div class="metric-value">£{v_gbp:,.2f}</div>
                    <div class="metric-delta delta-negative">
                        VaR: <b>{v_pct:+.2f}%</b> | Expected Shortfall (CVaR): <b>£{cv_gbp:,.2f}</b> ({cv_pct:+.2f}%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # -------------------------------------------------------------------------
    # 4. Full Percentile Term Structure Curves (0.01 to 0.99)
    # -------------------------------------------------------------------------
    st.markdown("#### 📈 Value-at-Risk Percentile Term Structure (1% to 99%)")
    st.caption("Continuous tail loss spectrum comparing standard Historical Simulation against Volatility-Scaled VaR (loaded directly from DB).")

    model_colors = {
        "Historical Simulation": "#EF4444",          # Crimson Red
        "Vol-Scaled VaR (EWMA λ=0.94)": "#3B82F6",   # Bright Navy Blue
        "Vol-Scaled VaR (Sample 30d)": "#10B981",    # Emerald Green
        "Vol-Scaled VaR (Sample 60d)": "#059669",
    }

    # Plot VaR GBP Spectrum
    fig_spectrum = go.Figure()
    for method, group in spectrum_df.groupby("METHOD"):
        c = model_colors.get(method, "#64748B")
        fig_spectrum.add_trace(
            go.Scatter(
                x=group["CONFIDENCE_LEVEL"] * 100.0,
                y=group["VAR_GBP"],
                name=method,
                mode="lines",
                line=dict(color=c, width=2.6),
                hovertemplate=f"<b>{method}</b><br>Confidence: <b>%{{x:.1f}}%</b><br>VaR: <b>£%{{y:,.2f}}</b><extra></extra>"
            )
        )

    # Add vertical cursor at inspected percentile
    fig_spectrum.add_vline(
        x=target_cl * 100.0,
        line_dash="dash",
        line_color="#0F172A",
        line_width=1.5,
        annotation_text=f"Selected: {target_cl*100:.1f}%",
        annotation_position="bottom right"
    )

    layout_spec = get_plotly_layout_defaults()
    layout_spec.update(dict(
        title=dict(text=f"Portfolio Value-at-Risk (£) vs Confidence Level ({horizon_days}-Day Horizon)", font=dict(size=14, color="#0F172A")),
        height=440
    ))
    fig_spectrum.update_layout(**layout_spec)
    fig_spectrum.update_xaxes(title_text="Confidence Level (%)", ticksuffix="%")
    fig_spectrum.update_yaxes(title_text="Value at Risk (£)", tickprefix="£")
    st.plotly_chart(fig_spectrum, use_container_width=True)

    # Expected Shortfall (CVaR) Spectrum Plot
    with st.expander("📊 View Expected Shortfall / CVaR Percentile Term Structure", expanded=False):
        fig_cvar = go.Figure()
        for method, group in spectrum_df.groupby("METHOD"):
            c = model_colors.get(method, "#64748B")
            fig_cvar.add_trace(
                go.Scatter(
                    x=group["CONFIDENCE_LEVEL"] * 100.0,
                    y=group["CVAR_GBP"],
                    name=f"{method} (CVaR)",
                    mode="lines",
                    line=dict(color=c, width=2.2, dash="dash"),
                    hovertemplate=f"<b>{method}</b><br>Confidence: <b>%{{x:.1f}}%</b><br>CVaR: <b>£%{{y:,.2f}}</b><extra></extra>"
                )
            )
        layout_cvar = get_plotly_layout_defaults()
        layout_cvar.update(dict(
            title=dict(text="Expected Shortfall / Conditional VaR (£) across Percentiles", font=dict(size=13, color="#0F172A")),
            height=380
        ))
        fig_cvar.update_layout(**layout_cvar)
        fig_cvar.update_xaxes(title_text="Confidence Level (%)", ticksuffix="%")
        fig_cvar.update_yaxes(title_text="Expected Shortfall (£)", tickprefix="£")
        st.plotly_chart(fig_cvar, use_container_width=True)

    # -------------------------------------------------------------------------
    # 5. Component Risk Breakdown: Dropdown (Shapley vs Standalone)
    # -------------------------------------------------------------------------
    st.markdown("#### 🧩 Component Risk Attribution & Asset Breakdown")

    col_view1, _ = st.columns([2.0, 1.0])
    with col_view1:
        component_metric_choice = st.selectbox(
            "Select Component Risk Measure:",
            options=["Shapley Risk Contributions (Game-Theoretic)", "Standalone VaR Figures (Isolated Risk)"],
            index=0,
            help="Toggle between game-theoretic Shapley marginal risk contributions (default) and standalone isolated asset risk."
        )

    # Check database for stored component risk contributions
    stored_risk_df = fetch_stored_risk_contributions(asof_date=selected_asof, confidence_level=target_cl)

    # --- Mode A: Shapley Risk Contributions (Default) ---
    if "Shapley" in component_metric_choice:
        st.caption(
            "**Game-Theoretic Shapley Value Risk Attribution (Euler Allocation):** "
            "Decomposes total portfolio tail risk among holdings based on marginal contributions. "
            "The sum of all Shapley contributions strictly equals **100% of Portfolio VaR** (£)."
        )

        if not stored_risk_df.empty:
            # --- Load directly from PORTFOLIO_RISK_CONTRIBUTIONS ---
            hist_df = stored_risk_df[stored_risk_df["METHOD"] == "Historical Simulation"].set_index("TICKER")
            vol_df = stored_risk_df[stored_risk_df["METHOD"].str.startswith("Vol-Scaled")].set_index("TICKER")
            tickers = hist_df.index.union(vol_df.index).tolist()
            combined = []
            for t in tickers:
                h_row = hist_df.loc[t] if t in hist_df.index else {}
                v_row = vol_df.loc[t] if t in vol_df.index else {}
                pos_val = float(h_row.get("POSITION_VALUE_GBP", v_row.get("POSITION_VALUE_GBP", 0.0)))
                wt_pct = float(h_row.get("WEIGHT_PCT", v_row.get("WEIGHT_PCT", 0.0)))
                sh_h = float(h_row.get("SHAPLEY_VAR_GBP", 0.0))
                sh_s = float(v_row.get("SHAPLEY_VAR_GBP", 0.0))
                sh_h_pct = float(h_row.get("SHAPLEY_VAR_PCT", 0.0))
                sh_s_pct = float(v_row.get("SHAPLEY_VAR_PCT", 0.0))
                st_h = float(h_row.get("STANDALONE_VAR_GBP", 0.0))
                st_s = float(v_row.get("STANDALONE_VAR_GBP", 0.0))
                div_h = float(h_row.get("DIVERSIFICATION_BENEFIT_GBP", 0.0))
                div_s = float(v_row.get("DIVERSIFICATION_BENEFIT_GBP", 0.0))
                combined.append({
                    "TICKER": t,
                    "POSITION_VALUE_GBP": pos_val,
                    "WEIGHT_PCT": wt_pct,
                    "HIST_SHAPLEY_VAR_GBP": sh_h,
                    "HIST_SHAPLEY_VAR_PCT": sh_h_pct,
                    "VOL_SCALED_SHAPLEY_VAR_GBP": sh_s,
                    "VOL_SCALED_SHAPLEY_VAR_PCT": sh_s_pct,
                    "HIST_STANDALONE_VAR_GBP": st_h,
                    "VOL_SCALED_STANDALONE_VAR_GBP": st_s,
                    "HIST_DIV_BENEFIT_GBP": div_h,
                    "VOL_SCALED_DIV_BENEFIT_GBP": div_s,
                    "PORTFOLIO_HIST_VAR_GBP": total_portfolio_value,
                    "PORTFOLIO_VOL_SCALED_VAR_GBP": total_portfolio_value
                })
            shapley_df = pd.DataFrame(combined).sort_values("HIST_SHAPLEY_VAR_GBP", ascending=True).reset_index(drop=True)
            st.caption("🟢 Source: Materialized Database Records (`PORTFOLIO_RISK_CONTRIBUTIONS`)")
        else:
            # --- Dynamic fallback live calculation ---
            with st.spinner("Computing Shapley Value risk contributions..."):
                shapley_df = compute_shapley_risk_contributions(
                    price_history=prices_gbp,
                    positions=positions,
                    asof_date=selected_asof,
                    lookback_days=lookback_days,
                    confidence_level=target_cl,
                    ewma_lambda=0.94
                )
            st.caption("⚡ Source: Live Computed (Percentile not in DB)")

        if not shapley_df.empty:
            # Shapley Bar Chart
            fig_shapley = go.Figure()
            fig_shapley.add_trace(
                go.Bar(
                    y=shapley_df["TICKER"],
                    x=shapley_df["HIST_SHAPLEY_VAR_GBP"],
                    name="Historical Shapley VaR",
                    orientation="h",
                    marker_color="#EF4444",
                    opacity=0.85,
                    hovertemplate="<b>%{y}</b> Hist Shapley: <b>£%{x:,.2f}</b><extra></extra>"
                )
            )
            fig_shapley.add_trace(
                go.Bar(
                    y=shapley_df["TICKER"],
                    x=shapley_df["VOL_SCALED_SHAPLEY_VAR_GBP"],
                    name="Vol-Scaled Shapley VaR (EWMA λ=0.94)",
                    orientation="h",
                    marker_color="#3B82F6",
                    opacity=0.85,
                    hovertemplate="<b>%{y}</b> Vol-Scaled Shapley: <b>£%{x:,.2f}</b><extra></extra>"
                )
            )

            layout_shapley = get_plotly_layout_defaults()
            layout_shapley.update(dict(
                title=dict(
                    text=f"Shapley Risk Contribution by Asset at {target_cl*100:.1f}% Confidence Level",
                    font=dict(size=14, color="#0F172A")
                ),
                height=max(400, len(shapley_df) * 30),
                barmode="group",
                yaxis=dict(autorange="reversed")
            ))
            fig_shapley.update_layout(**layout_shapley)
            fig_shapley.update_xaxes(title_text="Shapley VaR Contribution (£)", tickprefix="£")
            st.plotly_chart(fig_shapley, use_container_width=True)

            # Formatted Shapley Table
            with st.expander("📋 View Shapley Risk Attribution Data Table", expanded=True):
                disp_shapley = shapley_df[[
                    "TICKER", "POSITION_VALUE_GBP", "WEIGHT_PCT",
                    "HIST_SHAPLEY_VAR_GBP", "HIST_SHAPLEY_VAR_PCT",
                    "VOL_SCALED_SHAPLEY_VAR_GBP", "VOL_SCALED_SHAPLEY_VAR_PCT",
                    "VOL_SCALED_STANDALONE_VAR_GBP", "VOL_SCALED_DIV_BENEFIT_GBP"
                ]].copy()
                disp_shapley.columns = [
                    "Ticker", "Value (£)", "Weight (%)",
                    "Hist Shapley (£)", "Hist Share (%)",
                    "Vol-Scaled Shapley (£)", "Vol-Scaled Share (%)",
                    "Standalone VaR (£)", "Diversification Benefit (£)"
                ]
                st.dataframe(
                    disp_shapley.style.format({
                        "Value (£)": "£{:,.2f}",
                        "Weight (%)": "{:.2f}%",
                        "Hist Shapley (£)": "£{:,.2f}",
                        "Hist Share (%)": "{:.2f}%",
                        "Vol-Scaled Shapley (£)": "£{:,.2f}",
                        "Vol-Scaled Share (%)": "{:.2f}%",
                        "Standalone VaR (£)": "£{:,.2f}",
                        "Diversification Benefit (£)": "+£{:,.2f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

    # --- Mode B: Standalone VaR Figures ---
    else:
        st.caption(
            "**Standalone Asset VaR:** "
            "Evaluates each asset's standalone tail risk assuming it is held entirely in isolation (100% position allocation without diversification offsets)."
        )

        if not stored_risk_df.empty:
            # Load standalone directly from DB
            hist_df = stored_risk_df[stored_risk_df["METHOD"] == "Historical Simulation"].set_index("TICKER")
            vol_df = stored_risk_df[stored_risk_df["METHOD"].str.startswith("Vol-Scaled")].set_index("TICKER")
            tickers = hist_df.index.union(vol_df.index).tolist()
            combined = []
            for t in tickers:
                h_row = hist_df.loc[t] if t in hist_df.index else {}
                v_row = vol_df.loc[t] if t in vol_df.index else {}
                pos_val = float(h_row.get("POSITION_VALUE_GBP", v_row.get("POSITION_VALUE_GBP", 0.0)))
                wt_pct = float(h_row.get("WEIGHT_PCT", v_row.get("WEIGHT_PCT", 0.0)))
                st_h = float(h_row.get("STANDALONE_VAR_GBP", 0.0))
                st_s = float(v_row.get("STANDALONE_VAR_GBP", 0.0))
                diff_gbp = st_s - st_h
                diff_pct = ((abs(st_s) - abs(st_h)) / abs(st_h)) * 100.0 if abs(st_h) > 0 else 0.0
                combined.append({
                    "TICKER": t,
                    "POSITION_VALUE_GBP": pos_val,
                    "WEIGHT_PCT": wt_pct,
                    "HIST_VAR_GBP": st_h,
                    "HIST_VAR_PCT": (st_h / pos_val) * 100.0 if pos_val > 0 else 0.0,
                    "VOL_SCALED_VAR_GBP": st_s,
                    "VOL_SCALED_VAR_PCT": (st_s / pos_val) * 100.0 if pos_val > 0 else 0.0,
                    "DIFFERENCE_GBP": diff_gbp,
                    "DIFFERENCE_PCT": diff_pct,
                    "CURRENT_EWMA_VOL_ANN": 0.0
                })
            standalone_df = pd.DataFrame(combined).sort_values("POSITION_VALUE_GBP", ascending=False).reset_index(drop=True)
            st.caption("🟢 Source: Materialized Database Records (`PORTFOLIO_RISK_CONTRIBUTIONS`)")
        else:
            with st.spinner("Computing standalone asset VaR..."):
                standalone_df = compute_standalone_asset_var(
                    price_history=prices_gbp,
                    positions=positions,
                    asof_date=selected_asof,
                    lookback_days=lookback_days,
                    confidence_level=target_cl,
                    ewma_lambda=0.94
                )
            st.caption("⚡ Source: Live Computed (Percentile not in DB)")

        if not standalone_df.empty:
            # Standalone Bar Chart
            fig_bar = go.Figure()
            fig_bar.add_trace(
                go.Bar(
                    y=standalone_df["TICKER"],
                    x=standalone_df["HIST_VAR_GBP"],
                    name="Historical Standalone VaR",
                    orientation="h",
                    marker_color="#EF4444",
                    opacity=0.85,
                    hovertemplate="<b>%{y}</b> Hist Standalone VaR: £%{x:,.2f}<extra></extra>"
                )
            )
            fig_bar.add_trace(
                go.Bar(
                    y=standalone_df["TICKER"],
                    x=standalone_df["VOL_SCALED_VAR_GBP"],
                    name="Vol-Scaled Standalone VaR (EWMA λ=0.94)",
                    orientation="h",
                    marker_color="#3B82F6",
                    opacity=0.85,
                    hovertemplate="<b>%{y}</b> Vol-Scaled Standalone VaR: £%{x:,.2f}<extra></extra>"
                )
            )

            layout_bar = get_plotly_layout_defaults()
            layout_bar.update(dict(
                title=dict(text=f"Standalone VaR by Asset at {target_cl*100:.1f}% Confidence Level (Isolated Risk)", font=dict(size=14, color="#0F172A")),
                height=max(400, len(standalone_df) * 30),
                barmode="group",
                yaxis=dict(autorange="reversed")
            ))
            fig_bar.update_layout(**layout_bar)
            fig_bar.update_xaxes(title_text="Standalone VaR (£)", tickprefix="£")
            st.plotly_chart(fig_bar, use_container_width=True)

            # Formatted Standalone Table
            with st.expander("📋 View Standalone Asset VaR Data Table", expanded=True):
                disp_df = standalone_df[[
                    "TICKER", "POSITION_VALUE_GBP", "WEIGHT_PCT",
                    "HIST_VAR_GBP", "HIST_VAR_PCT",
                    "VOL_SCALED_VAR_GBP", "VOL_SCALED_VAR_PCT",
                    "DIFFERENCE_GBP", "DIFFERENCE_PCT"
                ]].copy()
                disp_df.columns = [
                    "Ticker", "Value (£)", "Weight (%)",
                    "Hist VaR (£)", "Hist VaR (%)",
                    "Vol-Scaled VaR (£)", "Vol-Scaled VaR (%)",
                    "Diff (£)", "Diff (%)"
                ]
                st.dataframe(
                    disp_df.style.format({
                        "Value (£)": "£{:,.2f}",
                        "Weight (%)": "{:.2f}%",
                        "Hist VaR (£)": "£{:,.2f}",
                        "Hist VaR (%)": "{:+.2f}%",
                        "Vol-Scaled VaR (£)": "£{:,.2f}",
                        "Vol-Scaled VaR (%)": "{:+.2f}%",
                        "Diff (£)": "£{:,.2f}",
                        "Diff (%)": "{:+.1f}%"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

    # -------------------------------------------------------------------------
    # 6. Empirical Scenario P&L Distribution & Worst Drawdown Days
    # -------------------------------------------------------------------------
    st.markdown("#### 📉 Empirical Scenario P&L Distribution & Worst Historical Dates")
    st.caption("Compares simulated portfolio scenario P&L under unscaled Historical Simulation vs Volatility-Scaled VaR (EWMA λ=0.94).")

    # Valuation Date Dropdown for Empirical Scenario P&L Section
    col_scen_ctrl1, col_scen_ctrl2 = st.columns([1.8, 1.2])
    with col_scen_ctrl1:
        default_scen_idx = available_dates.index(selected_asof) if (available_dates and selected_asof in available_dates) else 0
        scen_asof_date = st.selectbox(
            "Valuation As-Of Date (Empirical Scenario P&L):",
            options=available_dates if available_dates else [selected_asof],
            index=default_scen_idx,
            key="empirical_scen_asof_date",
            help="Select historical valuation date to compute both Historical Simulation and Volatility-Scaled scenario P&L distributions."
        )

    # Compute complete scenario P&L series (Historical & Vol-Scaled) for selected valuation date
    scen_df = compute_portfolio_scenario_pnl(
        price_history=prices_gbp,
        positions=positions,
        asof_date=scen_asof_date,
        lookback_days=lookback_days,
        ewma_lambda=0.94,
        horizon_days=horizon_days
    )

    if scen_df.empty:
        st.warning("Insufficient scenario data available for the selected valuation date.")
        return

    # Quantile metrics
    h_95 = float(np.percentile(scen_df["HISTORICAL_PNL"], 5))
    v_95 = float(np.percentile(scen_df["VOL_SCALED_EWMA_PNL"], 5))
    h_99 = float(np.percentile(scen_df["HISTORICAL_PNL"], 1))
    v_99 = float(np.percentile(scen_df["VOL_SCALED_EWMA_PNL"], 1))
    h_worst = float(scen_df["HISTORICAL_PNL"].min())
    v_worst = float(scen_df["VOL_SCALED_EWMA_PNL"].min())

    # Snapshot KPI Strip
    kpi_s1, kpi_s2, kpi_s3, kpi_s4 = st.columns(4)
    with kpi_s1:
        st.metric("Historical 95% 1d VaR", f"£{h_95:,.2f}")
    with kpi_s2:
        st.metric("Vol-Scaled 95% 1d VaR", f"£{v_95:,.2f}", delta=f"£{v_95 - h_95:+,.2f}", delta_color="inverse")
    with kpi_s3:
        st.metric("Worst Historical Shock", f"£{h_worst:,.2f}")
    with kpi_s4:
        st.metric("Worst Vol-Scaled Shock", f"£{v_worst:,.2f}", delta=f"£{v_worst - h_worst:+,.2f}", delta_color="inverse")

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    col_s1, col_s2 = st.columns([1.5, 1.5])

    with col_s1:
        st.markdown(f"##### Empirical Scenario P&L Distribution ({scen_asof_date})")
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=scen_df["HISTORICAL_PNL"],
                name="Historical Simulation P&L",
                marker_color="#EF4444",
                opacity=0.55,
                nbinsx=40,
                hovertemplate="Historical P&L: <b>£%{x:,.2f}</b><br>Count: %{y}<extra></extra>"
            )
        )
        fig_hist.add_trace(
            go.Histogram(
                x=scen_df["VOL_SCALED_EWMA_PNL"],
                name="Volatility-Scaled P&L (EWMA λ=0.94)",
                marker_color="#3B82F6",
                opacity=0.55,
                nbinsx=40,
                hovertemplate="Vol-Scaled P&L: <b>£%{x:,.2f}</b><br>Count: %{y}<extra></extra>"
            )
        )

        # Add vertical cutoffs for 95% VaR
        fig_hist.add_vline(
            x=h_95, line_dash="dash", line_color="#EF4444", line_width=1.5,
            annotation_text="Hist 95%", annotation_position="top left",
            annotation_font=dict(size=10, color="#EF4444")
        )
        fig_hist.add_vline(
            x=v_95, line_dash="dash", line_color="#3B82F6", line_width=1.5,
            annotation_text="Scaled 95%", annotation_position="top right",
            annotation_font=dict(size=10, color="#3B82F6")
        )

        layout_hist = get_plotly_layout_defaults()
        layout_hist.update(dict(
            barmode="overlay",
            height=400,
            margin=dict(l=45, r=30, t=35, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
                font=dict(size=10)
            )
        ))
        fig_hist.update_layout(**layout_hist)
        fig_hist.update_xaxes(title_text="Simulated Daily P&L (£)", tickprefix="£")
        fig_hist.update_yaxes(title_text="Scenario Frequency (Days)")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_s2:
        st.markdown(f"##### Worst 10 Historical Stress Scenarios ({scen_asof_date})")
        worst_df = scen_df.sort_values("HISTORICAL_PNL").head(10).copy()
        worst_df["DIFF_GBP"] = worst_df["VOL_SCALED_EWMA_PNL"] - worst_df["HISTORICAL_PNL"]
        worst_df["DIFF_PCT"] = ((worst_df["VOL_SCALED_EWMA_PNL"].abs() - worst_df["HISTORICAL_PNL"].abs()) / worst_df["HISTORICAL_PNL"].abs()) * 100.0

        disp_worst = worst_df[["DATE", "HISTORICAL_PNL", "VOL_SCALED_EWMA_PNL", "DIFF_GBP", "DIFF_PCT"]].copy()
        disp_worst["DATE"] = pd.to_datetime(disp_worst["DATE"]).dt.strftime("%Y-%m-%d")
        disp_worst.columns = ["Scenario Date", "Hist Loss (£)", "Scaled Loss (£)", "Diff (£)", "Scaling Impact (%)"]

        st.dataframe(
            disp_worst.style.format({
                "Hist Loss (£)": "£{:,.2f}",
                "Scaled Loss (£)": "£{:,.2f}",
                "Diff (£)": "£{:,.2f}",
                "Scaling Impact (%)": "{:+.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )
