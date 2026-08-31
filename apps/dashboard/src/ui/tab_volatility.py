"""
Tab 5: Rolling Volatility Analytics View (Shiny Module).
Interactive exploration and comparison of dynamic volatility estimators:
- RiskMetrics EWMA (default λ=0.94, customizable/multi-parameter)
- Equally Weighted Rolling Sample Standard Deviations (default 60d, customizable/multi-window)
- Volatility Scaling Multipliers (σ_today / σ_t) across all estimators
- Dual-axis price overlays and volatility spread/ratio diagnostics.
"""

from typing import List, Optional, Dict, Any, Callable
import pandas as pd
import numpy as np
from shiny import module, ui, render, reactive
from shinywidgets import output_widget, render_plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from portfolio_core.analytics.volatility import (
    calculate_sample_volatility,
    calculate_ewma_volatility,
    calculate_scaling_factors,
    compute_volatility_summary_metrics
)
try:
    from src.ui.theme import PALETTE, get_plotly_layout_defaults, render_metric_card
except ImportError:
    from apps.dashboard.src.ui.theme import PALETTE, get_plotly_layout_defaults, render_metric_card


ESTIMATOR_COLORS = [
    "#1E3A8A",  # Deep Navy Blue
    "#059669",  # Emerald Green
    "#D97706",  # Amber Orange
    "#7C3AED",  # Royal Purple
    "#0891B2",  # Cyan / Teal
    "#DC2626",  # Crimson Red
    "#EC4899",  # Pink
    "#4F46E5",  # Indigo
    "#CA8A04",  # Yellow Gold
    "#16A34A",  # Green
]


@module.ui
def tab_volatility_ui():
    """UI layout for Rolling Volatility & Regime Analytics."""
    return ui.TagList(
        ui.tags.div(
            ui.tags.h3("📊 Rolling Volatility & Regime Analytics", style="margin-bottom: 4px;"),
            ui.tags.p(
                "Compare dynamic EWMA and equally weighted rolling sample volatility estimators, volatility scaling multipliers, and price overlays.",
                class_="text-muted",
                style="margin-bottom: 1.2rem;"
            )
        ),

        # 1. Top Controls Bar
        ui.row(
            ui.column(6, ui.output_ui("vol_tickers_select_ui")),
            ui.column(6, ui.input_select("horizon_choice", "Time Horizon:", ["1 Month", "3 Months", "6 Months", "1 Year (Default)", "2 Years", "All Available", "Custom Range"], selected="1 Year (Default)"))
        ),
        ui.output_ui("custom_date_range_ui"),
        ui.tags.div(style="margin-bottom: 1rem;"),

        # 2. Multi-Estimator Configuration
        ui.accordion(
            ui.accordion_panel(
                "⚡ Volatility Estimators Configuration (EWMA & Rolling Look-Back Periods)",
                ui.row(
                    ui.column(
                        4,
                        ui.tags.h6("📈 RiskMetrics EWMA Parameters"),
                        ui.input_selectize(
                            "selected_lambdas",
                            "Select EWMA Decay Factor(s) (λ):",
                            choices={"0.85": "λ = 0.85", "0.88": "λ = 0.88", "0.90": "λ = 0.90", "0.92": "λ = 0.92", "0.94": "λ = 0.94 (RiskMetrics Default)", "0.96": "λ = 0.96", "0.97": "λ = 0.97", "0.98": "λ = 0.98"},
                            selected=["0.94"],
                            multiple=True
                        )
                    ),
                    ui.column(
                        4,
                        ui.tags.h6("📏 Rolling Windows"),
                        ui.input_selectize(
                            "selected_windows",
                            "Select Rolling Window(s) (Days):",
                            choices={"10": "10 Days", "20": "20 Days", "30": "30 Days", "60": "60 Days (Default 60d)", "90": "90 Days", "120": "120 Days", "180": "180 Days", "252": "252 Days (1 Year)"},
                            selected=["60"],
                            multiple=True
                        )
                    ),
                    ui.column(
                        4,
                        ui.tags.h6("🎨 Display Options"),
                        ui.input_checkbox("overlay_price", "Overlay Share Price", value=True),
                        ui.input_checkbox("annualize_vol", "Annualize Volatility (×√252)", value=True),
                        ui.input_checkbox("show_mean_line", "Show Horizon Baseline Mean", value=True)
                    )
                )
            ),
            id="vol_config_accordion",
            open=True
        ),
        ui.tags.div(style="margin-bottom: 1.5rem;"),

        # 3. Snapshot KPI Cards
        ui.output_ui("vol_kpi_cards_ui"),
        ui.tags.div(style="margin-bottom: 1.8rem;"),

        # 4. Main Volatility Trajectory Chart
        ui.tags.div(
            ui.tags.h4("📈 Dynamic Volatility Trajectory: EWMA vs Rolling Sample Estimators", style="margin-bottom: 4px;"),
            ui.tags.p("Compares the time-series paths of RiskMetrics EWMA and equally weighted rolling standard deviations.", class_="text-muted", style="font-size: 0.9rem;"),
            output_widget("vol_trajectory_chart"),
            style="margin-bottom: 1.8rem;"
        ),

        # 5. Volatility Scaling Factors Multipliers Chart
        ui.tags.div(
            ui.tags.h4("⚡ Volatility Scaling Multipliers Comparison (σ_today / σ_t)", style="margin-bottom: 4px;"),
            ui.tags.p("Compares historical return scaling multipliers generated by EWMA vs Rolling Sample Volatilities. Multipliers < 1.0 (Green) damp historical tail shocks; multipliers > 1.0 (Red) amplify historical tail shocks.", class_="text-muted", style="font-size: 0.9rem;"),
            output_widget("scaling_chart"),
            style="margin-bottom: 1.8rem;"
        ),

        # 6. Estimator Ratio & Spread Dynamics
        ui.accordion(
            ui.accordion_panel(
                "📊 View Estimator Ratio & Spread Dynamics (EWMA / Rolling 60d)",
                output_widget("ratio_chart")
            ),
            id="ratio_accordion",
            open=False
        ),
        ui.tags.div(style="margin-bottom: 1.8rem;"),

        # 7. Summary Statistics Table
        ui.tags.div(
            ui.tags.h4("📋 Detailed Estimator Summary Statistics & Scaling Metrics", style="margin-bottom: 8px;"),
            ui.output_ui("vol_summary_table_ui")
        )
    )


@module.server
def tab_volatility_server(input, output, session, shared_data: Callable[[], Dict[str, Any]]):
    """Server reactive logic for Volatility tab."""

    @render.ui
    def vol_tickers_select_ui():
        sdata = shared_data()
        tickers = sdata.get("available_tickers", [])
        default_selected = [t for t in ["NVDA", "STAN.L"] if t in tickers]
        if not default_selected and tickers:
            default_selected = tickers[:2]
        return ui.input_selectize(
            "selected_tickers",
            "Select Stock(s) to Analyze:",
            choices=tickers,
            selected=default_selected,
            multiple=True
        )

    @render.ui
    def custom_date_range_ui():
        if input.horizon_choice() == "Custom Range":
            return ui.row(
                ui.column(6, ui.input_date("start_date", "Start Date:")),
                ui.column(6, ui.input_date("end_date", "End Date:"))
            )
        return ui.HTML("")

    # Core Calculation Reactive Calc
    @reactive.calc
    def volatility_computation():
        sdata = shared_data()
        prices_gbp = sdata.get("prices_gbp", pd.DataFrame())
        selected_tickers = input.selected_tickers() or []

        if prices_gbp.empty or not selected_tickers:
            return None

        # Filter valid tickers
        valid_tickers = [t for t in selected_tickers if t in prices_gbp.columns]
        if not valid_tickers:
            return None

        horizon_choice = input.horizon_choice() or "1 Year (Default)"
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
        elif horizon_choice == "Custom Range":
            start_date = pd.to_datetime(input.start_date() or (max_date - pd.Timedelta(days=365)))
            max_date = pd.to_datetime(input.end_date() or max_date)
        else:
            start_date = max_date - pd.Timedelta(days=365)

        full_log_returns = np.log(prices_gbp[valid_tickers] / prices_gbp[valid_tickers].shift(1)).dropna(how="all")
        filtered_prices = prices_gbp.loc[start_date:max_date, valid_tickers].dropna(how="all")

        if filtered_prices.empty or len(filtered_prices) < 5:
            return None

        annualize_vol = input.annualize_vol()
        trading_days = 252 if annualize_vol else 1
        vol_unit_label = "Annualized Volatility (%)" if annualize_vol else "Daily Volatility (%)"

        selected_lambdas = [float(x) for x in (input.selected_lambdas() or ["0.94"])]
        selected_windows = [int(x) for x in (input.selected_windows() or ["60"])]

        stock_data: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for ticker in valid_tickers:
            if ticker not in full_log_returns.columns:
                continue
            r_full = full_log_returns[ticker].dropna()
            if len(r_full) < 5:
                continue

            estimators_dict: Dict[str, Dict[str, Any]] = {}

            # EWMA
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

            # Rolling
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

        return {
            "stock_data": stock_data,
            "valid_tickers": valid_tickers,
            "filtered_prices": filtered_prices,
            "vol_unit_label": vol_unit_label,
            "horizon_choice": horizon_choice,
            "overlay_price": input.overlay_price(),
            "show_mean_line": input.show_mean_line()
        }

    # 3. Snapshot KPI Cards
    @render.ui
    def vol_kpi_cards_ui():
        vcomp = volatility_computation()
        if not vcomp or not vcomp["stock_data"]:
            return ui.HTML('<div class="alert alert-warning">Please select at least one valid ticker to compute volatility.</div>')

        stock_data = vcomp["stock_data"]
        valid_tickers = vcomp["valid_tickers"]

        cards_html = []
        for ticker in valid_tickers:
            if ticker not in stock_data:
                continue
            est_dict = stock_data[ticker]
            if not est_dict:
                continue

            primary_ewma_name = next((k for k in est_dict if "0.94" in k), next((k for k in est_dict if "EWMA" in k), list(est_dict.keys())[0]))
            primary_roll_name = next((k for k in est_dict if "60d" in k), next((k for k in est_dict if "Rolling" in k), list(est_dict.keys())[0]))

            v_ewma = est_dict[primary_ewma_name]["vol"]
            v_roll = est_dict[primary_roll_name]["vol"]

            if v_ewma.empty or v_roll.empty:
                continue

            latest_ewma_val = float(v_ewma.iloc[-1]) * 100.0
            latest_roll_val = float(v_roll.iloc[-1]) * 100.0
            ewma_pct_rank = float((v_ewma <= v_ewma.iloc[-1]).mean() * 100.0)

            vol_ratio = latest_ewma_val / latest_roll_val if latest_roll_val > 0 else 1.0
            ratio_label = "Calmed (< 60d Mean)" if vol_ratio < 0.95 else ("Spike (> 60d Mean)" if vol_ratio > 1.05 else "In-Line with 60d")

            cards_html.append(f"""
            <div class="col-md-6 col-lg-4">
                <div class="metric-card">
                    <div class="metric-label">{ticker} — Volatility Estimator Comparison</div>
                    <div class="metric-value">{latest_ewma_val:.2f}% <span style="font-size:0.95rem; font-weight:600; color:#059669;">vs {latest_roll_val:.2f}% (60d)</span></div>
                    <div class="metric-delta">
                        EWMA ({primary_ewma_name}): <b>{latest_ewma_val:.2f}%</b> (Rank: <b>{ewma_pct_rank:.1f}%</b>)<br>
                        Ratio (EWMA / 60d): <b>{vol_ratio:.2f}x</b> ({ratio_label})
                    </div>
                </div>
            </div>
            """)

        return ui.HTML(f'<div class="row g-3">{"".join(cards_html)}</div>')

    # 4. Main Volatility Trajectory Chart
    @render_plotly
    def vol_trajectory_chart():
        vcomp = volatility_computation()
        if not vcomp or not vcomp["stock_data"]:
            return go.Figure()

        stock_data = vcomp["stock_data"]
        valid_tickers = vcomp["valid_tickers"]
        filtered_prices = vcomp["filtered_prices"]
        vol_unit_label = vcomp["vol_unit_label"]
        horizon_choice = vcomp["horizon_choice"]
        overlay_price = vcomp["overlay_price"]
        show_mean_line = vcomp["show_mean_line"]

        fig = make_subplots(specs=[[{"secondary_y": overlay_price}]])
        color_idx = 0

        for ticker in valid_tickers:
            if ticker not in stock_data:
                continue
            est_dict = stock_data[ticker]
            prices = filtered_prices[ticker].dropna() if ticker in filtered_prices.columns else pd.Series()

            for name, data in est_dict.items():
                v_series = data["vol"]
                est_type = data["type"]
                param = data["param"]

                c = ESTIMATOR_COLORS[color_idx % len(ESTIMATOR_COLORS)]
                dash_style = "solid" if (est_type == "EWMA" and param == 0.94) else ("dash" if (est_type == "Rolling" and param == 60) else "dot")
                lw = 2.6 if (param in [0.94, 60]) else 1.8

                fig.add_trace(
                    go.Scatter(
                        x=v_series.index,
                        y=v_series.values * 100.0,
                        name=f"{ticker} {name}",
                        line=dict(color=c, width=lw, dash=dash_style),
                        hovertemplate=f"<b>{ticker} {name}</b>: <b>%{{y:.2f}}%</b><extra></extra>"
                    ),
                    secondary_y=False
                )
                color_idx += 1

            if show_mean_line and est_dict:
                first_key = list(est_dict.keys())[0]
                primary_v = est_dict[first_key]["vol"]
                if not primary_v.empty:
                    mean_val = float(primary_v.mean()) * 100.0
                    fig.add_trace(
                        go.Scatter(
                            x=[primary_v.index[0], primary_v.index[-1]],
                            y=[mean_val, mean_val],
                            name=f"{ticker} Mean Baseline ({mean_val:.1f}%)",
                            line=dict(color="#64748B", width=1.3, dash="dash"),
                            hoverinfo="skip"
                        ),
                        secondary_y=False
                    )

            if overlay_price and not prices.empty:
                fig.add_trace(
                    go.Scatter(
                        x=prices.index,
                        y=prices.values,
                        name=f"{ticker} Share Price (GBP)",
                        line=dict(color="#64748B", width=1.4, dash="dashdot"),
                        opacity=0.5,
                        hovertemplate=f"{ticker} Price: £%{{y:,.2f}}<extra></extra>"
                    ),
                    secondary_y=True
                )

        layout_vol = get_plotly_layout_defaults()
        layout_vol.update(dict(
            title=dict(text=f"Dynamic Volatility Trajectory Comparison ({horizon_choice})", font=dict(size=14, color="#0F172A")),
            height=440,
        ))
        fig.update_layout(**layout_vol)
        fig.update_yaxes(title_text=vol_unit_label, ticksuffix="%", secondary_y=False)
        if overlay_price:
            fig.update_yaxes(title_text="Share Price (£)", tickprefix="£", secondary_y=True, showgrid=False)
        return fig

    # 5. Volatility Scaling Multipliers Chart
    @render_plotly
    def scaling_chart():
        vcomp = volatility_computation()
        if not vcomp or not vcomp["stock_data"]:
            return go.Figure()

        stock_data = vcomp["stock_data"]
        valid_tickers = vcomp["valid_tickers"]

        fig_scale = go.Figure()
        color_idx = 0

        for ticker in valid_tickers:
            if ticker not in stock_data:
                continue
            est_dict = stock_data[ticker]

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
                        name=f"{ticker} {name} Scaling",
                        line=dict(color=c, width=lw, dash=dash_style),
                        hovertemplate=f"<b>{ticker} {name}</b> Multiplier: <b>%{{y:.2f}}x</b><extra></extra>"
                    )
                )
                color_idx += 1

        fig_scale.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="#DC2626",
            line_width=1.5,
            annotation_text="Neutral 1.0x",
            annotation_position="top right"
        )
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
            title=dict(text="Scaling Factors (σ_today / σ_t) Comparison Across Estimators", font=dict(size=14, color="#0F172A")),
            height=400
        ))
        fig_scale.update_layout(**layout_scale)
        fig_scale.update_yaxes(title_text="Scaling Multiplier (σ_today / σ_t)", tickformat=".2f")
        return fig_scale

    # 6. Estimator Ratio & Spread Dynamics
    @render_plotly
    def ratio_chart():
        vcomp = volatility_computation()
        if not vcomp or not vcomp["stock_data"]:
            return go.Figure()

        stock_data = vcomp["stock_data"]
        valid_tickers = vcomp["valid_tickers"]

        fig_ratio = go.Figure()

        for ticker in valid_tickers:
            if ticker not in stock_data:
                continue
            est_dict = stock_data[ticker]

            p_ewma = next((k for k in est_dict if "0.94" in k), next((k for k in est_dict if "EWMA" in k), None))
            p_roll = next((k for k in est_dict if "60d" in k), next((k for k in est_dict if "Rolling" in k), None))

            if p_ewma and p_roll:
                v_ew = est_dict[p_ewma]["vol"]
                v_ro = est_dict[p_roll]["vol"]
                ratio_series = (v_ew / v_ro).dropna()

                fig_ratio.add_trace(
                    go.Scatter(
                        x=ratio_series.index,
                        y=ratio_series.values,
                        name=f"{ticker} ({p_ewma} / {p_roll})",
                        line=dict(width=2.2),
                        hovertemplate=f"<b>{ticker}</b> Ratio: <b>%{{y:.2f}}x</b><extra></extra>"
                    )
                )

        fig_ratio.add_hline(y=1.0, line_dash="dash", line_color="#64748B", line_width=1.3, annotation_text="Parity 1.0x")
        fig_ratio.add_hrect(y0=0.0, y1=1.0, fillcolor="#DCFCE7", opacity=0.35, line_width=0, annotation_text="Calming Regime (EWMA < 60d)")
        fig_ratio.add_hrect(y0=1.0, y1=2.5, fillcolor="#FEE2E2", opacity=0.25, line_width=0, annotation_text="Volatility Spike Regime (EWMA > 60d)")

        layout_r = get_plotly_layout_defaults()
        layout_r.update(dict(
            title=dict(text="Volatility Ratio Dynamics (EWMA / Rolling 60d)", font=dict(size=13, color="#0F172A")),
            height=340
        ))
        fig_ratio.update_layout(**layout_r)
        fig_ratio.update_yaxes(title_text="Ratio (EWMA / Rolling 60d)", tickformat=".2f")
        return fig_ratio

    # 7. Summary Statistics Table
    @render.ui
    def vol_summary_table_ui():
        vcomp = volatility_computation()
        if not vcomp or not vcomp["stock_data"]:
            return ui.HTML('<div class="text-muted">No volatility calculations available.</div>')

        stock_data = vcomp["stock_data"]
        valid_tickers = vcomp["valid_tickers"]

        table_rows = []
        for ticker in valid_tickers:
            if ticker not in stock_data:
                continue
            est_dict = stock_data[ticker]

            for name, data in est_dict.items():
                v_s = data["vol"]
                s_s = data["scaling"]
                if v_s.empty:
                    continue

                latest_v = float(v_s.iloc[-1]) * 100.0
                mean_v = float(v_s.mean()) * 100.0
                med_v = float(v_s.median()) * 100.0
                min_v = float(v_s.min()) * 100.0
                max_v = float(v_s.max()) * 100.0
                pct_rank = float((v_s <= v_s.iloc[-1]).mean() * 100.0)

                mean_scale = float(s_s.mean()) if not s_s.empty else 1.0
                min_scale = float(s_s.min()) if not s_s.empty else 1.0
                max_scale = float(s_s.max()) if not s_s.empty else 1.0
                trough_damp_pct = (1.0 - min_scale) * 100.0 if min_scale < 1.0 else 0.0

                table_rows.append(f"""
                <tr>
                    <td><b>{ticker}</b></td>
                    <td>{name}</td>
                    <td><b>{latest_v:.2f}%</b></td>
                    <td>{mean_v:.2f}%</td>
                    <td>{med_v:.2f}%</td>
                    <td>{min_v:.2f}%</td>
                    <td>{max_v:.2f}%</td>
                    <td>{pct_rank:.1f}%</td>
                    <td>{mean_scale:.2f}x</td>
                    <td><span class="text-success">{min_scale:.2f}x (-{trough_damp_pct:.1f}%)</span></td>
                    <td><span class="text-danger">{max_scale:.2f}x</span></td>
                </tr>
                """)

        if not table_rows:
            return ui.HTML('<div class="text-muted">No data.</div>')

        return ui.HTML(f"""
        <div style="overflow-x: auto; max-height: 420px;">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Estimator</th>
                        <th>Latest Vol</th>
                        <th>Horizon Mean</th>
                        <th>Horizon Median</th>
                        <th>Horizon Min</th>
                        <th>Horizon Max</th>
                        <th>Percentile Rank</th>
                        <th>Mean Scaling</th>
                        <th>Min Scaling (Max Damp)</th>
                        <th>Max Scaling (Max Amp)</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
        </div>
        """)
