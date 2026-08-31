"""
Tab 3: Value-at-Risk (VaR) & Tail Risk Spectrum View (Shiny Module).
Database-First Architecture:
- By default, all VaR metrics, CVaR curves, and Shapley risk contributions
  are fetched directly from materialized database tables (PORTFOLIO_VAR, PORTFOLIO_RISK_CONTRIBUTIONS,
  PORTFOLIO_SCENARIO_PNL) with zero recalculation overhead.
- Live on-the-fly math is used seamlessly as a dynamic fallback only if data is not yet in the DB.
"""

from typing import List, Optional, Dict, Any, Callable
import pandas as pd
import numpy as np
from shiny import module, ui, render, reactive
from shinywidgets import output_widget, render_plotly
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
try:
    from src.ui.theme import PALETTE, get_plotly_layout_defaults, render_metric_card
except ImportError:
    from apps.dashboard.src.ui.theme import PALETTE, get_plotly_layout_defaults, render_metric_card


@module.ui
def tab_var_ui():
    """UI layout for Value-at-Risk and Risk Attribution view."""
    return ui.TagList(
        ui.tags.div(
            ui.tags.h3("🛡️ Value-at-Risk (VaR) & Percentile Spectrum", style="margin-bottom: 4px;"),
            ui.tags.p(
                "Inspect historical and volatility-scaled tail risk across every percentile (0.01 to 0.99) and component risk attribution.",
                class_="text-muted",
                style="margin-bottom: 1.2rem;"
            )
        ),

        # 1. Top Controls Bar
        ui.row(
            ui.column(4, ui.output_ui("asof_select_ui")),
            ui.column(4, ui.input_select("lookback_days", "Simulation Lookback:", {"260": "260 Days (1 Year)", "520": "520 Days (2 Years)", "130": "130 Days (6 Months)", "780": "780 Days (3 Years)"}, selected="260")),
            ui.column(4, ui.input_select("horizon_days", "Holding Horizon:", {"1": "1-Day Horizon", "5": "5-Day Horizon", "10": "10-Day Horizon", "21": "21-Day Horizon (1 Month)"}, selected="1"))
        ),
        ui.tags.div(style="margin-bottom: 1.2rem;"),

        # 2. Interactive Percentile Inspector & Key Metrics
        ui.tags.div(
            ui.tags.h4("🎯 Quick Percentile Inspector", style="margin-bottom: 4px;"),
            ui.output_ui("var_data_source_badge"),
            ui.row(
                ui.column(8, ui.input_slider("target_cl", "Inspect Confidence Level (Percentile):", min=0.01, max=0.99, value=0.95, step=0.01)),
                ui.column(
                    4,
                    ui.tags.div(
                        ui.tags.label("Quick Presets:", style="font-size: 0.85rem; font-weight: 600; color: #64748B; margin-bottom: 6px;"),
                        ui.tags.div(
                            ui.input_action_button("btn_cl_90", "90%", class_="btn-sm btn-outline-primary"),
                            ui.input_action_button("btn_cl_95", "95%", class_="btn-sm btn-outline-primary"),
                            ui.input_action_button("btn_cl_99", "99%", class_="btn-sm btn-outline-primary"),
                            ui.input_action_button("btn_cl_995", "99.5%", class_="btn-sm btn-outline-primary"),
                            style="display: flex; gap: 6px;"
                        )
                    )
                )
            ),
            ui.tags.div(style="margin-top: 0.8rem;"),
            ui.output_ui("risk_snapshot_cards_ui"),
            style="margin-bottom: 2rem;"
        ),

        # 3. Full Percentile Term Structure Curves (0.01 to 0.99)
        ui.tags.div(
            ui.tags.h4("📈 Value-at-Risk Percentile Term Structure (1% to 99%)", style="margin-bottom: 4px;"),
            ui.tags.p("Continuous tail loss spectrum comparing standard Historical Simulation against Volatility-Scaled VaR.", class_="text-muted", style="font-size: 0.9rem;"),
            output_widget("var_spectrum_chart"),
            ui.accordion(
                ui.accordion_panel(
                    "📊 View Expected Shortfall / CVaR Percentile Term Structure",
                    output_widget("cvar_spectrum_chart")
                ),
                id="cvar_accordion",
                open=False
            ),
            style="margin-bottom: 2rem;"
        ),

        # 4. Component Risk Breakdown
        ui.tags.div(
            ui.tags.h4("🧩 Component Risk Attribution & Asset Breakdown", style="margin-bottom: 8px;"),
            ui.input_select(
                "component_metric_choice",
                "Select Component Risk Measure:",
                ["Shapley Risk Contributions", "Standalone VaR Figures"],
                selected="Shapley Risk Contributions"
            ),
            ui.output_ui("component_risk_caption_ui"),
            output_widget("component_risk_chart"),
            ui.tags.div(style="margin-top: 1rem;"),
            ui.accordion(
                ui.accordion_panel(
                    "📋 View Detailed Risk Attribution Data Table",
                    ui.output_ui("component_risk_table_ui")
                ),
                id="comp_table_accordion",
                open=True
            ),
            style="margin-bottom: 2rem;"
        ),

        # 5. Empirical Scenario P&L Distribution & Worst Historical Dates
        ui.tags.div(
            ui.tags.h4("📉 Empirical Scenario P&L Distribution & Worst Historical Dates", style="margin-bottom: 4px;"),
            ui.tags.p("Compares simulated portfolio scenario P&L under unscaled Historical Simulation vs Volatility-Scaled VaR (EWMA λ=0.94).", class_="text-muted", style="font-size: 0.9rem;"),
            ui.output_ui("scen_kpis_ui"),
            ui.tags.div(style="margin-top: 1rem;"),
            ui.row(
                ui.column(6, output_widget("scen_hist_chart")),
                ui.column(6, ui.output_ui("worst_scenarios_table_ui"))
            )
        )
    )


@module.server
def tab_var_server(input, output, session, shared_data: Callable[[], Dict[str, Any]]):
    """Server reactive logic for VaR tab."""

    # Quick Preset Button Handlers
    @reactive.effect
    @reactive.event(input.btn_cl_90)
    def _set_cl_90():
        ui.update_slider("target_cl", value=0.90)

    @reactive.effect
    @reactive.event(input.btn_cl_95)
    def _set_cl_95():
        ui.update_slider("target_cl", value=0.95)

    @reactive.effect
    @reactive.event(input.btn_cl_99)
    def _set_cl_99():
        ui.update_slider("target_cl", value=0.99)

    @reactive.effect
    @reactive.event(input.btn_cl_995)
    def _set_cl_995():
        ui.update_slider("target_cl", value=0.99)

    @render.ui
    def asof_select_ui():
        sdata = shared_data()
        prices_gbp = sdata.get("prices_gbp", pd.DataFrame())
        available_dates = fetch_available_var_dates()
        default_asof = sdata.get("asof_date") or (available_dates[0] if available_dates else (str(prices_gbp.index[-1])[:10] if not prices_gbp.empty else "latest"))
        opts = available_dates if available_dates else [default_asof]
        return ui.input_select(
            "selected_asof",
            "Reporting As-Of Date:",
            choices=opts,
            selected=default_asof
        )

    # Core VaR Spectrum Data Loader (Database-First with Live Fallback)
    @reactive.calc
    def var_spectrum_data():
        sdata = shared_data()
        prices_gbp = sdata.get("prices_gbp", pd.DataFrame())
        positions = sdata.get("positions", {})
        selected_asof = input.selected_asof() or sdata.get("asof_date")
        lookback_days = int(input.lookback_days() or 260)
        horizon_days = int(input.horizon_days() or 1)

        if prices_gbp.empty or not positions:
            return None

        stored_var_df = fetch_stored_var_metrics(asof_date=selected_asof)

        if not stored_var_df.empty:
            spectrum_df = stored_var_df.copy()
            spectrum_df["METHOD"] = spectrum_df["METHOD"].replace({
                "Vol-Scaled VaR (EWMA Volatility (λ=0.94))": "Vol-Scaled VaR (EWMA λ=0.94)",
                "Vol-Scaled VaR (Sample 30d Volatility)": "Vol-Scaled VaR (Sample 30d)",
                "Vol-Scaled VaR (Sample 60d Volatility)": "Vol-Scaled VaR (Sample 60d)"
            })
            spectrum_df = spectrum_df[~spectrum_df["METHOD"].str.startswith("Parametric")]
            total_portfolio_value = float(spectrum_df["PORTFOLIO_VALUE_GBP"].iloc[0])
            var_data_source = "🟢 Database Records (`PORTFOLIO_VAR`)"

            scenario_pnl_df = fetch_stored_scenario_pnl(asof_date=selected_asof)
            if scenario_pnl_df.empty:
                _, scenario_pnl_df, _ = compute_multi_model_var_spectrum(
                    price_history=prices_gbp, positions=positions, asof_date=selected_asof, lookback_days=lookback_days
                )
        else:
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

        return {
            "spectrum_df": spectrum_df,
            "scenario_pnl_df": scenario_pnl_df,
            "total_portfolio_value": total_portfolio_value,
            "var_data_source": var_data_source,
            "selected_asof": selected_asof,
            "lookback_days": lookback_days,
            "horizon_days": horizon_days,
            "prices_gbp": prices_gbp,
            "positions": positions
        }

    @render.ui
    def var_data_source_badge():
        vdata = var_spectrum_data()
        if not vdata:
            return ui.HTML("")
        return ui.HTML(f"""
        <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 10px;">
            Data Source: <b>{vdata['var_data_source']}</b> | Valuation Date: <b>{vdata['selected_asof']}</b>
        </div>
        """)

    # Metric Cards at Inspected Percentile
    @render.ui
    def risk_snapshot_cards_ui():
        vdata = var_spectrum_data()
        if not vdata or vdata["spectrum_df"].empty:
            return ui.HTML('<div class="alert alert-warning">No VaR data available.</div>')

        spectrum_df = vdata["spectrum_df"]
        total_portfolio_value = vdata["total_portfolio_value"]
        target_cl = round(float(input.target_cl() or 0.95), 2)

        current_metrics = spectrum_df[spectrum_df["CONFIDENCE_LEVEL"] == target_cl]
        if current_metrics.empty:
            # Nearest match
            nearest_idx = (spectrum_df["CONFIDENCE_LEVEL"] - target_cl).abs().idxmin()
            current_metrics = spectrum_df.loc[[nearest_idx]]

        cards_html = []
        for _, row in current_metrics.iterrows():
            m_name = row["METHOD"]
            v_gbp = row["VAR_GBP"]
            v_pct = row["VAR_PCT"]
            cv_gbp = row["CVAR_GBP"]
            cv_pct = row["CVAR_PCT"]

            cards_html.append(f"""
            <div class="col-md-3 col-sm-6">
                <div class="metric-card">
                    <div class="metric-label">{m_name}</div>
                    <div class="metric-value">£{v_gbp:,.2f}</div>
                    <div class="metric-delta delta-negative">
                        VaR: <b>{v_pct:+.2f}%</b> | Expected Shortfall (CVaR): <b>£{cv_gbp:,.2f}</b> ({cv_pct:+.2f}%)
                    </div>
                </div>
            </div>
            """)

        header_text = f"Risk Snapshot at <b>{target_cl*100:.1f}% Confidence Level</b> (Portfolio Value: <b>£{total_portfolio_value:,.2f}</b>)"
        return ui.HTML(f"""
        <div>
            <h6 style="margin-bottom: 10px; color: #0F172A;">{header_text}</h6>
            <div class="row g-3">
                {''.join(cards_html)}
            </div>
        </div>
        """)

    # Full Percentile Term Structure (VaR)
    @render_plotly
    def var_spectrum_chart():
        vdata = var_spectrum_data()
        if not vdata or vdata["spectrum_df"].empty:
            return go.Figure()

        spectrum_df = vdata["spectrum_df"]
        target_cl = float(input.target_cl() or 0.95)
        horizon_days = vdata["horizon_days"]

        model_colors = {
            "Historical Simulation": "#EF4444",
            "Vol-Scaled VaR (EWMA λ=0.94)": "#3B82F6",
            "Vol-Scaled VaR (Sample 30d)": "#10B981",
            "Vol-Scaled VaR (Sample 60d)": "#059669",
        }

        fig = go.Figure()
        for method, group in spectrum_df.groupby("METHOD"):
            c = model_colors.get(method, "#64748B")
            fig.add_trace(
                go.Scatter(
                    x=group["CONFIDENCE_LEVEL"] * 100.0,
                    y=group["VAR_GBP"],
                    name=method,
                    mode="lines",
                    line=dict(color=c, width=2.6),
                    hovertemplate=f"<b>{method}</b><br>Confidence: <b>%{{x:.1f}}%</b><br>VaR: <b>£%{{y:,.2f}}</b><extra></extra>"
                )
            )

        fig.add_vline(
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
        fig.update_layout(**layout_spec)
        fig.update_xaxes(title_text="Confidence Level (%)", ticksuffix="%")
        fig.update_yaxes(title_text="Value at Risk (£)", tickprefix="£")
        return fig

    # Expected Shortfall (CVaR) Spectrum
    @render_plotly
    def cvar_spectrum_chart():
        vdata = var_spectrum_data()
        if not vdata or vdata["spectrum_df"].empty:
            return go.Figure()

        spectrum_df = vdata["spectrum_df"]
        model_colors = {
            "Historical Simulation": "#EF4444",
            "Vol-Scaled VaR (EWMA λ=0.94)": "#3B82F6",
            "Vol-Scaled VaR (Sample 30d)": "#10B981",
            "Vol-Scaled VaR (Sample 60d)": "#059669",
        }

        fig = go.Figure()
        for method, group in spectrum_df.groupby("METHOD"):
            c = model_colors.get(method, "#64748B")
            fig.add_trace(
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
        fig.update_layout(**layout_cvar)
        fig.update_xaxes(title_text="Confidence Level (%)", ticksuffix="%")
        fig.update_yaxes(title_text="Expected Shortfall (£)", tickprefix="£")
        return fig

    # Component Risk Calculation & Table
    @render.ui
    def component_risk_caption_ui():
        choice = input.component_metric_choice()
        if "Shapley" in choice:
            return ui.HTML("""
            <div class="info-callout" style="font-size: 0.88rem; margin-top: 6px;">
                <b>Game-Theoretic Shapley Value Risk Attribution (Euler Allocation):</b>
                Decomposes total portfolio tail risk among holdings based on marginal contributions.
                The sum of all Shapley contributions strictly equals <b>100% of Portfolio VaR</b> (£).
            </div>
            """)
        else:
            return ui.HTML("""
            <div class="info-callout" style="font-size: 0.88rem; margin-top: 6px;">
                <b>Standalone Asset VaR:</b>
                Evaluates each asset's standalone tail risk assuming it is held entirely in isolation (100% position allocation without diversification offsets).
            </div>
            """)

    @reactive.calc
    def component_risk_data():
        vdata = var_spectrum_data()
        if not vdata:
            return None

        selected_asof = vdata["selected_asof"]
        target_cl = round(float(input.target_cl() or 0.95), 2)
        choice = input.component_metric_choice()
        prices_gbp = vdata["prices_gbp"]
        positions = vdata["positions"]
        lookback_days = vdata["lookback_days"]
        total_portfolio_value = vdata["total_portfolio_value"]

        stored_risk_df = fetch_stored_risk_contributions(asof_date=selected_asof, confidence_level=target_cl)

        if "Shapley" in choice:
            if not stored_risk_df.empty:
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
                    })
                df = pd.DataFrame(combined).sort_values("HIST_SHAPLEY_VAR_GBP", ascending=True).reset_index(drop=True)
            else:
                df = compute_shapley_risk_contributions(
                    price_history=prices_gbp,
                    positions=positions,
                    asof_date=selected_asof,
                    lookback_days=lookback_days,
                    confidence_level=target_cl,
                    ewma_lambda=0.94
                )
            return {"type": "shapley", "df": df, "target_cl": target_cl}
        else:
            if not stored_risk_df.empty:
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
                    })
                df = pd.DataFrame(combined).sort_values("POSITION_VALUE_GBP", ascending=False).reset_index(drop=True)
            else:
                df = compute_standalone_asset_var(
                    price_history=prices_gbp,
                    positions=positions,
                    asof_date=selected_asof,
                    lookback_days=lookback_days,
                    confidence_level=target_cl,
                    ewma_lambda=0.94
                )
            return {"type": "standalone", "df": df, "target_cl": target_cl}

    @render_plotly
    def component_risk_chart():
        cdata = component_risk_data()
        if not cdata or cdata["df"].empty:
            return go.Figure()

        df = cdata["df"]
        target_cl = cdata["target_cl"]

        if cdata["type"] == "shapley":
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    y=df["TICKER"],
                    x=df["HIST_SHAPLEY_VAR_GBP"],
                    name="Historical Shapley VaR",
                    orientation="h",
                    marker_color="#EF4444",
                    opacity=0.85,
                    hovertemplate="<b>%{y}</b> Hist Shapley: <b>£%{x:,.2f}</b><extra></extra>"
                )
            )
            fig.add_trace(
                go.Bar(
                    y=df["TICKER"],
                    x=df["VOL_SCALED_SHAPLEY_VAR_GBP"],
                    name="Vol-Scaled Shapley VaR (EWMA λ=0.94)",
                    orientation="h",
                    marker_color="#3B82F6",
                    opacity=0.85,
                    hovertemplate="<b>%{y}</b> Vol-Scaled Shapley: <b>£%{x:,.2f}</b><extra></extra>"
                )
            )
            layout = get_plotly_layout_defaults()
            layout.update(dict(
                title=dict(text=f"Shapley Risk Contribution by Asset at {target_cl*100:.1f}% Confidence Level", font=dict(size=14, color="#0F172A")),
                height=max(400, len(df) * 30),
                barmode="group",
                yaxis=dict(autorange="reversed")
            ))
            fig.update_layout(**layout)
            fig.update_xaxes(title_text="Shapley VaR Contribution (£)", tickprefix="£")
            return fig
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    y=df["TICKER"],
                    x=df["HIST_VAR_GBP"],
                    name="Historical Standalone VaR",
                    orientation="h",
                    marker_color="#EF4444",
                    opacity=0.85,
                    hovertemplate="<b>%{y}</b> Hist Standalone VaR: £%{x:,.2f}<extra></extra>"
                )
            )
            fig.add_trace(
                go.Bar(
                    y=df["TICKER"],
                    x=df["VOL_SCALED_VAR_GBP"],
                    name="Vol-Scaled Standalone VaR (EWMA λ=0.94)",
                    orientation="h",
                    marker_color="#3B82F6",
                    opacity=0.85,
                    hovertemplate="<b>%{y}</b> Vol-Scaled Standalone VaR: £%{x:,.2f}<extra></extra>"
                )
            )
            layout = get_plotly_layout_defaults()
            layout.update(dict(
                title=dict(text=f"Standalone VaR by Asset at {target_cl*100:.1f}% Confidence Level (Isolated Risk)", font=dict(size=14, color="#0F172A")),
                height=max(400, len(df) * 30),
                barmode="group",
                yaxis=dict(autorange="reversed")
            ))
            fig.update_layout(**layout)
            fig.update_xaxes(title_text="Standalone VaR (£)", tickprefix="£")
            return fig

    @render.ui
    def component_risk_table_ui():
        cdata = component_risk_data()
        if not cdata or cdata["df"].empty:
            return ui.HTML('<div class="text-muted">No data.</div>')

        df = cdata["df"]

        if cdata["type"] == "shapley":
            rows_html = "".join([
                f"""
                <tr>
                    <td><b>{r['TICKER']}</b></td>
                    <td>£{float(r['POSITION_VALUE_GBP']):,.2f}</td>
                    <td>{float(r['WEIGHT_PCT']):.2f}%</td>
                    <td>£{float(r['HIST_SHAPLEY_VAR_GBP']):,.2f}</td>
                    <td>{float(r.get('HIST_SHAPLEY_VAR_PCT', 0)):.2f}%</td>
                    <td><b>£{float(r['VOL_SCALED_SHAPLEY_VAR_GBP']):,.2f}</b></td>
                    <td>{float(r.get('VOL_SCALED_SHAPLEY_VAR_PCT', 0)):.2f}%</td>
                    <td>£{float(r.get('VOL_SCALED_STANDALONE_VAR_GBP', 0)):,.2f}</td>
                    <td><span class="text-success">+£{float(r.get('VOL_SCALED_DIV_BENEFIT_GBP', 0)):,.2f}</span></td>
                </tr>
                """ for _, r in df.iterrows()
            ])
            return ui.HTML(f"""
            <div style="overflow-x: auto; max-height: 400px;">
                <table class="custom-table">
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Value (£)</th>
                            <th>Weight (%)</th>
                            <th>Hist Shapley (£)</th>
                            <th>Hist Share (%)</th>
                            <th>Vol-Scaled Shapley (£)</th>
                            <th>Vol-Scaled Share (%)</th>
                            <th>Standalone VaR (£)</th>
                            <th>Diversification Benefit (£)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            """)
        else:
            rows_html = "".join([
                f"""
                <tr>
                    <td><b>{r['TICKER']}</b></td>
                    <td>£{float(r['POSITION_VALUE_GBP']):,.2f}</td>
                    <td>{float(r['WEIGHT_PCT']):.2f}%</td>
                    <td>£{float(r['HIST_VAR_GBP']):,.2f}</td>
                    <td>{float(r.get('HIST_VAR_PCT', 0)):+.2f}%</td>
                    <td><b>£{float(r['VOL_SCALED_VAR_GBP']):,.2f}</b></td>
                    <td>{float(r.get('VOL_SCALED_VAR_PCT', 0)):+.2f}%</td>
                    <td>£{float(r.get('DIFFERENCE_GBP', 0)):,.2f}</td>
                    <td>{float(r.get('DIFFERENCE_PCT', 0)):+.1f}%</td>
                </tr>
                """ for _, r in df.iterrows()
            ])
            return ui.HTML(f"""
            <div style="overflow-x: auto; max-height: 400px;">
                <table class="custom-table">
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Value (£)</th>
                            <th>Weight (%)</th>
                            <th>Hist VaR (£)</th>
                            <th>Hist VaR (%)</th>
                            <th>Vol-Scaled VaR (£)</th>
                            <th>Vol-Scaled VaR (%)</th>
                            <th>Diff (£)</th>
                            <th>Diff (%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            """)

    # Scenario P&L Series
    @reactive.calc
    def scenario_pnl_data():
        vdata = var_spectrum_data()
        if not vdata:
            return None

        prices_gbp = vdata["prices_gbp"]
        positions = vdata["positions"]
        selected_asof = vdata["selected_asof"]
        lookback_days = vdata["lookback_days"]
        horizon_days = vdata["horizon_days"]

        scen_df = compute_portfolio_scenario_pnl(
            price_history=prices_gbp,
            positions=positions,
            asof_date=selected_asof,
            lookback_days=lookback_days,
            ewma_lambda=0.94,
            horizon_days=horizon_days
        )
        return scen_df

    @render.ui
    def scen_kpis_ui():
        scen_df = scenario_pnl_data()
        if scen_df is None or scen_df.empty:
            return ui.HTML("")

        h_95 = float(np.percentile(scen_df["HISTORICAL_PNL"], 5))
        v_95 = float(np.percentile(scen_df["VOL_SCALED_EWMA_PNL"], 5))
        h_worst = float(scen_df["HISTORICAL_PNL"].min())
        v_worst = float(scen_df["VOL_SCALED_EWMA_PNL"].min())

        card1 = render_metric_card("Historical 95% 1d VaR", f"£{h_95:,.2f}")
        card2 = render_metric_card("Vol-Scaled 95% 1d VaR", f"£{v_95:,.2f}", f"£{v_95 - h_95:+,.2f}", "negative" if v_95 < h_95 else "positive")
        card3 = render_metric_card("Worst Historical Shock", f"£{h_worst:,.2f}")
        card4 = render_metric_card("Worst Vol-Scaled Shock", f"£{v_worst:,.2f}", f"£{v_worst - h_worst:+,.2f}", "negative" if v_worst < h_worst else "positive")

        return ui.HTML(f"""
        <div class="row g-3">
            <div class="col-md-3 col-sm-6">{card1}</div>
            <div class="col-md-3 col-sm-6">{card2}</div>
            <div class="col-md-3 col-sm-6">{card3}</div>
            <div class="col-md-3 col-sm-6">{card4}</div>
        </div>
        """)

    @render_plotly
    def scen_hist_chart():
        scen_df = scenario_pnl_data()
        if scen_df is None or scen_df.empty:
            return go.Figure()

        h_95 = float(np.percentile(scen_df["HISTORICAL_PNL"], 5))
        v_95 = float(np.percentile(scen_df["VOL_SCALED_EWMA_PNL"], 5))

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
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(size=10))
        ))
        fig_hist.update_layout(**layout_hist)
        fig_hist.update_xaxes(title_text="Simulated Daily P&L (£)", tickprefix="£")
        fig_hist.update_yaxes(title_text="Scenario Frequency (Days)")
        return fig_hist

    @render.ui
    def worst_scenarios_table_ui():
        scen_df = scenario_pnl_data()
        if scen_df is None or scen_df.empty:
            return ui.HTML('<div class="text-muted">No scenario data.</div>')

        worst_df = scen_df.sort_values("HISTORICAL_PNL").head(10).copy()
        worst_df["DIFF_GBP"] = worst_df["VOL_SCALED_EWMA_PNL"] - worst_df["HISTORICAL_PNL"]
        worst_df["DIFF_PCT"] = ((worst_df["VOL_SCALED_EWMA_PNL"].abs() - worst_df["HISTORICAL_PNL"].abs()) / worst_df["HISTORICAL_PNL"].abs()) * 100.0

        rows_html = "".join([
            f"""
            <tr>
                <td>{pd.to_datetime(r['DATE']).strftime('%Y-%m-%d')}</td>
                <td><span class="text-danger">£{float(r['HISTORICAL_PNL']):,.2f}</span></td>
                <td><span class="text-primary">£{float(r['VOL_SCALED_EWMA_PNL']):,.2f}</span></td>
                <td>£{float(r['DIFF_GBP']):,.2f}</td>
                <td>{float(r['DIFF_PCT']):+.1f}%</td>
            </tr>
            """ for _, r in worst_df.iterrows()
        ])

        return ui.HTML(f"""
        <div style="overflow-x: auto; max-height: 400px;">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Scenario Date</th>
                        <th>Hist Loss (£)</th>
                        <th>Scaled Loss (£)</th>
                        <th>Diff (£)</th>
                        <th>Scaling Impact (%)</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """)

