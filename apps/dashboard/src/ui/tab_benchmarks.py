"""
Tab 2: Centralized Benchmarking & Performance Comparison View (Shiny Module).
Centralizes all benchmark comparisons, historical valuations against actual portfolio,
value change / return distribution histograms, and linear combination benchmark management.
"""

from typing import List, Optional, Dict, Any, Callable
import json
import numpy as np
import pandas as pd
from shiny import module, ui, render, reactive
from shinywidgets import output_widget, render_plotly
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


@module.ui
def tab_benchmarks_ui():
    """UI layout for Centralized Benchmarking, Comparative Performance, and Distribution view."""
    return ui.TagList(
        ui.tags.div(
            ui.tags.h3("🎯 Centralized Benchmarking & Performance Analysis", style="margin-bottom: 4px;"),
            ui.tags.p(
                "Compare actual portfolio performance against market benchmarks and custom linear combinations. "
                "Each benchmark maintains an exact shadow portfolio matching the GBP capital deployed on every trade date.",
                class_="text-muted",
                style="margin-bottom: 1.2rem;"
            )
        ),

        # 1. Historical Valuation & Performance Comparison
        ui.tags.div(
            ui.tags.h4("📈 Historical Valuation & Performance Comparison", style="margin-bottom: 8px;"),
            ui.row(
                ui.column(5, ui.output_ui("bm_select_ui")),
                ui.column(4, ui.input_radio_buttons("chart_mode", "Display Metric:", ["Valuation (£)", "Indexed Growth (%)"], selected="Valuation (£)", inline=True)),
                ui.column(3, ui.input_select("lookback_choice", "Time Horizon:", ["All Time", "3 Years", "2 Years", "1 Year", "6 Months"], selected="All Time"))
            ),
            output_widget("trajectory_chart"),
            ui.tags.div(style="margin-top: 1rem;"),
            ui.tags.h5("📊 Comparative Total Performance & Asset Breakdown Scorecard", style="margin-bottom: 8px;"),
            ui.output_ui("scorecard_table_ui"),
            style="margin-bottom: 2rem;"
        ),

        # 2. Histogram of Value Changes & Daily Returns Distribution
        ui.tags.div(
            ui.tags.h4("📊 Value Changes & Daily Returns Distribution Histogram", style="margin-bottom: 4px;"),
            ui.tags.p(
                "Inspect the historical distribution of daily valuation changes (£) and daily returns (%) for your actual portfolio alongside selected benchmarks.",
                class_="text-muted",
                style="font-size: 0.9rem;"
            ),
            ui.row(
                ui.column(4, ui.input_radio_buttons("hist_metric", "Distribution Variable:", ["Daily Value Change (£)", "Daily Return (%)"], selected="Daily Value Change (£)", inline=True)),
                ui.column(4, ui.input_radio_buttons("hist_barmode", "Histogram Layout:", ["Overlaid", "Side-by-Side (Subplots)"], selected="Overlaid", inline=True)),
                ui.column(4, ui.input_slider("hist_bins", "Number of Bins:", min=15, max=80, value=35, step=5))
            ),
            output_widget("hist_chart"),
            ui.tags.div(style="margin-top: 1rem;"),
            ui.tags.h5("📋 Distribution Summary Statistics", style="margin-bottom: 8px;"),
            ui.output_ui("hist_stats_table_ui"),
            style="margin-bottom: 2rem;"
        ),

        # 3. Benchmark Assets & Linear Combination Management
        ui.tags.div(
            ui.tags.h4("⚙️ Benchmark Asset & Blend Management", style="margin-bottom: 4px;"),
            ui.tags.p(
                "Register new single-ticker benchmarks or weighted linear combinations, or remove existing benchmarks from the database.",
                class_="text-muted",
                style="font-size: 0.9rem;"
            ),
            ui.row(
                ui.column(
                    6,
                    ui.card(
                        ui.card_header("➕ Register Benchmark (Single or Linear Combination)"),
                        ui.input_text(
                            "bm_constituents_input",
                            "Constituent Ticker(s) & Weights (%):",
                            value="CSP1.L: 60, VUKE.L: 40",
                            placeholder="e.g. CSP1.L: 60, VUKE.L: 40 or VWRL.L: 100"
                        ),
                        ui.input_text(
                            "new_bm_name",
                            "Benchmark Name (Optional):",
                            value="",
                            placeholder="Leave blank for fallback name (e.g. CSP1.L_60_VUKE.L_40)"
                        ),
                        ui.input_text(
                            "new_bm_desc",
                            "Description (Optional):",
                            value="",
                            placeholder="e.g. 60% S&P 500, 40% FTSE 100 Equity Blend"
                        ),
                        ui.row(
                            ui.column(6, ui.input_action_button("add_bm_btn", "➕ Add Benchmark", class_="btn-primary w-100")),
                            ui.column(6, ui.input_action_button("recalc_bm_btn", "🔄 Recalculate Shadow Trades", class_="btn-outline-secondary w-100"))
                        ),
                        ui.output_ui("bm_form_status")
                    )
                ),
                ui.column(
                    6,
                    ui.card(
                        ui.card_header("📋 Registered Benchmarks (BENCHMARKS Table)"),
                        ui.output_ui("registered_bms_table_ui"),
                        ui.tags.hr(),
                        ui.tags.h6("🗑️ Remove Benchmark"),
                        ui.output_ui("del_bm_select_ui"),
                        ui.input_action_button("del_bm_btn", "🗑️ Delete Selected Benchmark", class_="btn-outline-danger w-100", style="margin-top: 8px;"),
                        ui.output_ui("bm_del_status")
                    )
                )
            ),
            style="margin-bottom: 1.5rem;"
        ),

        # 4. View Benchmark Shadow Transactions Expander
        ui.accordion(
            ui.accordion_panel(
                "🔍 View Benchmark Shadow Transactions (BENCHMARK_TRANSACTIONS Table)",
                ui.tags.p("Each row represents a shadow trade created with equivalent GBP invested value and constituent weight on the original transaction date.", class_="text-muted", style="font-size: 0.88rem;"),
                ui.output_ui("shadow_tx_table_ui")
            ),
            id="shadow_tx_accordion",
            open=False
        )
    )


@module.server
def tab_benchmarks_server(input, output, session, shared_data: Callable[[], Dict[str, Any]]):
    """Server reactive logic for Benchmarks tab."""
    engine = get_engine()

    # Reactive refresh trigger for benchmark additions/deletions
    bm_version = reactive.value(0)
    bm_status_msg = reactive.value(None)
    del_status_msg = reactive.value(None)

    @reactive.calc
    def benchmark_metadata():
        _ = bm_version()
        bm_info_df = fetch_benchmarks_info(engine=engine)
        available_bms = bm_info_df["BENCHMARK_CODE"].tolist() if not bm_info_df.empty else []
        bm_name_map = dict(zip(bm_info_df["BENCHMARK_CODE"], bm_info_df["NAME"])) if not bm_info_df.empty else {}
        return {
            "bm_info_df": bm_info_df,
            "available_bms": available_bms,
            "bm_name_map": bm_name_map
        }

    @render.ui
    def bm_select_ui():
        meta = benchmark_metadata()
        avail = meta["available_bms"]
        name_map = meta["bm_name_map"]
        choices = {b: f"{b} — {name_map.get(b, b)}" for b in avail}
        return ui.input_selectize(
            "selected_bms",
            "Select Benchmarks to Compare:",
            choices=choices,
            selected=avail,
            multiple=True
        )

    @render.ui
    def del_bm_select_ui():
        meta = benchmark_metadata()
        avail = meta["available_bms"]
        name_map = meta["bm_name_map"]
        choices = {b: f"{b} — {name_map.get(b, b)}" for b in avail}
        return ui.input_select(
            "bm_to_del",
            "Select Benchmark to Remove:",
            choices=choices
        )

    # Add Benchmark Handler
    @reactive.effect
    @reactive.event(input.add_bm_btn)
    def handle_add_benchmark():
        constituents = (input.bm_constituents_input() or "").strip()
        name = (input.new_bm_name() or "").strip()
        desc = (input.new_bm_desc() or "").strip()

        if not constituents:
            bm_status_msg.set(('danger', "Please enter benchmark constituent ticker(s)."))
            return

        try:
            res_add = add_benchmark(
                constituents=constituents,
                name=name if name else None,
                description=desc if desc else None,
                engine=engine
            )
            calculate_and_store_daily_benchmark_values(engine=engine)
            bm_version.set(bm_version() + 1)
            bm_status_msg.set(('success', f"✅ Benchmark <b>{res_add['benchmark_code']}</b> ('{res_add['name']}') successfully registered!"))
        except Exception as ex:
            bm_status_msg.set(('danger', f"Failed to add benchmark: {ex}"))

    # Recalculate Shadow Trades Handler
    @reactive.effect
    @reactive.event(input.recalc_bm_btn)
    def handle_recalc_benchmarks():
        try:
            res_bm = calculate_and_store_daily_benchmark_values(engine=engine)
            bm_version.set(bm_version() + 1)
            bm_status_msg.set(('success', f"✅ Generated {res_bm.get('records_stored', 0)} benchmark valuation points!"))
        except Exception as ex:
            bm_status_msg.set(('danger', f"Failed to recalculate benchmark shadow trades: {ex}"))

    # Delete Benchmark Handler
    @reactive.effect
    @reactive.event(input.del_bm_btn)
    def handle_delete_benchmark():
        bm_code = input.bm_to_del()
        if not bm_code:
            del_status_msg.set(('danger', "Please select a benchmark to delete."))
            return

        try:
            delete_benchmark(bm_code, engine=engine)
            bm_version.set(bm_version() + 1)
            del_status_msg.set(('success', f"✅ Deleted benchmark <b>{bm_code}</b> from the database."))
        except Exception as ex:
            del_status_msg.set(('danger', f"Failed to delete benchmark: {ex}"))

    @render.ui
    def bm_form_status():
        msg = bm_status_msg()
        if not msg:
            return ui.HTML("")
        style_cls, text = msg
        return ui.HTML(f'<div class="alert alert-{style_cls}" style="margin-top: 10px; padding: 8px 12px; font-size: 0.9rem;">{text}</div>')

    @render.ui
    def bm_del_status():
        msg = del_status_msg()
        if not msg:
            return ui.HTML("")
        style_cls, text = msg
        return ui.HTML(f'<div class="alert alert-{style_cls}" style="margin-top: 8px; padding: 8px 12px; font-size: 0.9rem;">{text}</div>')

    @reactive.calc
    def benchmark_history_data():
        _ = bm_version()
        sdata = shared_data()
        asof_date = sdata.get("asof_date", None)
        bm_history_df = fetch_benchmark_values_history(asof_date=asof_date, engine=engine)
        pv_df = fetch_portfolio_values_history(asof_date=asof_date, engine=engine)
        return {
            "bm_history_df": bm_history_df,
            "pv_df": pv_df,
            "asof_date": asof_date
        }

    # 1. Historical Valuation / Total Return Trajectory Chart
    @render_plotly
    def trajectory_chart():
        hdata = benchmark_history_data()
        meta = benchmark_metadata()
        pv_df = hdata["pv_df"]
        bm_history_df = hdata["bm_history_df"]
        bm_name_map = meta["bm_name_map"]

        if pv_df.empty or len(pv_df) <= 1:
            fig = go.Figure()
            fig.update_layout(title="No historical portfolio valuation data available.")
            return fig

        selected_bms = input.selected_bms() or []
        chart_mode = input.chart_mode()
        lookback_choice = input.lookback_choice()

        pv_df_sorted = pv_df.sort_values("DATE").reset_index(drop=True)
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
            fig_pv.add_trace(
                go.Scatter(
                    x=pv_filtered["DATE"],
                    y=port_vals_filtered,
                    name="My Portfolio (Stocks + Cash)",
                    line=dict(color="#1E3A8A", width=3.2),
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>My Portfolio Total: <b>£%{y:,.2f}</b><extra></extra>"
                )
            )

            for idx, bm_code in enumerate(selected_bms):
                bm_sub = bm_history_df[bm_history_df["BENCHMARK_CODE"] == bm_code]
                if not bm_sub.empty:
                    bm_sub = bm_sub.sort_values("DATE").reset_index(drop=True)
                    bm_sub = bm_sub[bm_sub["DATE"] >= cutoff_date]
                    if not bm_sub.empty:
                        color = bm_colors[idx % len(bm_colors)]
                        dash = bm_dash_styles[idx % len(bm_dash_styles)]
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
            return fig_pv

        else:
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
            return fig_pv

    # Scorecard Table
    @render.ui
    def scorecard_table_ui():
        hdata = benchmark_history_data()
        meta = benchmark_metadata()
        pv_df = hdata["pv_df"]
        bm_history_df = hdata["bm_history_df"]
        bm_name_map = meta["bm_name_map"]

        if pv_df.empty or len(pv_df) <= 1:
            return ui.HTML('<div class="text-muted">No valuation history.</div>')

        selected_bms = input.selected_bms() or []
        lookback_choice = input.lookback_choice()

        pv_df_sorted = pv_df.sort_values("DATE").reset_index(drop=True)
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

        scorecard_rows = []

        port_latest_tot = float(port_vals_filtered.iloc[-1])
        port_latest_stk = float(pv_filtered["STOCKS"].iloc[-1]) if "STOCKS" in pv_filtered.columns else port_latest_tot
        port_latest_csh = float(pv_filtered["CASH"].iloc[-1]) if "CASH" in pv_filtered.columns else 0.0
        port_init_val = float(port_vals_filtered.iloc[0]) if float(port_vals_filtered.iloc[0]) > 0 else 1.0
        port_total_ret = ((port_latest_tot / port_init_val) - 1.0) * 100.0

        port_1d = ((port_vals_filtered.iloc[-1] / port_vals_filtered.iloc[-2]) - 1.0) * 100.0 if len(port_vals_filtered) > 1 else 0.0
        idx_30 = max(0, len(port_vals_filtered) - 22)
        port_30d = ((port_vals_filtered.iloc[-1] / port_vals_filtered.iloc[idx_30]) - 1.0) * 100.0 if len(port_vals_filtered) > idx_30 else 0.0
        idx_1y = max(0, len(port_vals_filtered) - 252)
        port_1y = ((port_vals_filtered.iloc[-1] / port_vals_filtered.iloc[idx_1y]) - 1.0) * 100.0 if len(port_vals_filtered) > idx_1y else 0.0

        port_pct_changes = port_vals_filtered.pct_change().dropna()
        port_ann_vol = float(port_pct_changes.std() * np.sqrt(252) * 100.0) if len(port_pct_changes) > 1 else 0.0
        port_cagr = (((port_latest_tot / port_init_val) ** (252.0 / max(len(port_vals_filtered), 1))) - 1.0) * 100.0 if port_init_val > 0 else 0.0
        port_sharpe = (port_cagr / port_ann_vol) if port_ann_vol > 0 else 0.0

        scorecard_rows.append({
            "asset": "💼 My Portfolio",
            "tot": f"£{port_latest_tot:,.2f}",
            "stk": f"£{port_latest_stk:,.2f}",
            "csh": f"£{port_latest_csh:,.2f}",
            "d1": f"{'+' if port_1d >= 0 else ''}{port_1d:.2f}%",
            "d30": f"{'+' if port_30d >= 0 else ''}{port_30d:.2f}%",
            "y1": f"{'+' if port_1y >= 0 else ''}{port_1y:.2f}%",
            "period": f"{'+' if port_total_ret >= 0 else ''}{port_total_ret:.2f}%",
            "vol": f"{port_ann_vol:.2f}%",
            "sharpe": f"{port_sharpe:.2f}"
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
                        "asset": bm_label,
                        "tot": f"£{bm_latest_tot:,.2f}",
                        "stk": f"£{bm_latest_stk:,.2f}",
                        "csh": f"£{bm_latest_csh:,.2f}",
                        "d1": f"{'+' if bm_1d >= 0 else ''}{bm_1d:.2f}%",
                        "d30": f"{'+' if bm_30d >= 0 else ''}{bm_30d:.2f}%",
                        "y1": f"{'+' if bm_1y >= 0 else ''}{bm_1y:.2f}%",
                        "period": f"{'+' if bm_tot_ret >= 0 else ''}{bm_tot_ret:.2f}%",
                        "vol": f"{bm_ann_vol:.2f}%",
                        "sharpe": f"{bm_sharpe:.2f}"
                    })

        rows_html = "".join([
            f"""
            <tr>
                <td><b>{r['asset']}</b></td>
                <td><b>{r['tot']}</b></td>
                <td>{r['stk']}</td>
                <td>{r['csh']}</td>
                <td>{r['d1']}</td>
                <td>{r['d30']}</td>
                <td>{r['y1']}</td>
                <td><b>{r['period']}</b></td>
                <td>{r['vol']}</td>
                <td>{r['sharpe']}</td>
            </tr>
            """ for r in scorecard_rows
        ])

        return ui.HTML(f"""
        <div style="overflow-x: auto;">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Asset / Benchmark</th>
                        <th>Total Value (£)</th>
                        <th>Stock Holdings (£)</th>
                        <th>Dividend Cash (£)</th>
                        <th>1-Day Return</th>
                        <th>30-Day Return</th>
                        <th>1-Year Return</th>
                        <th>Period Return</th>
                        <th>Ann. Volatility</th>
                        <th>Return/Risk</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """)

    # 2. Histogram Distribution Chart & Stats
    @reactive.calc
    def distribution_data_map():
        hdata = benchmark_history_data()
        pv_df = hdata["pv_df"]
        bm_history_df = hdata["bm_history_df"]
        selected_bms = input.selected_bms() or []
        lookback_choice = input.lookback_choice()
        hist_metric = input.hist_metric()

        if pv_df.empty or len(pv_df) <= 1:
            return {}

        pv_df_sorted = pv_df.sort_values("DATE").reset_index(drop=True)
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

        return dist_data

    @render_plotly
    def hist_chart():
        dist_data = distribution_data_map()
        if not dist_data:
            return go.Figure()

        hist_metric = input.hist_metric()
        hist_barmode = input.hist_barmode()
        nbins = int(input.hist_bins())
        bm_colors = ["#F59E0B", "#8B5CF6", "#10B981", "#EC4899", "#06B6D4", "#E11D48", "#64748B"]

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
            return fig_hist

        else:
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
            return fig_sub

    @render.ui
    def hist_stats_table_ui():
        dist_data = distribution_data_map()
        if not dist_data:
            return ui.HTML('<div class="text-muted">No distribution data.</div>')

        hist_metric = input.hist_metric()
        unit_sym = "£" if "£" in hist_metric else ""
        unit_pct = "%" if "%" in hist_metric else ""

        rows_html = []
        for label, series in dist_data.items():
            m = float(series.mean())
            s = float(series.std())
            med = float(series.median())
            mn = float(series.min())
            mx = float(series.max())
            pos_pct = float((series > 0).mean() * 100.0)

            rows_html.append(f"""
            <tr>
                <td><b>{label}</b></td>
                <td>{unit_sym}{m:+,.2f}{unit_pct}</td>
                <td>{unit_sym}{s:,.2f}{unit_pct}</td>
                <td>{unit_sym}{med:+,.2f}{unit_pct}</td>
                <td><span class="text-success">{unit_sym}{mx:+,.2f}{unit_pct}</span></td>
                <td><span class="text-danger">{unit_sym}{mn:+,.2f}{unit_pct}</span></td>
                <td><b>{pos_pct:.1f}%</b></td>
            </tr>
            """)

        return ui.HTML(f"""
        <div style="overflow-x: auto;">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Series</th>
                        <th>Mean Daily</th>
                        <th>Std Dev (Daily Vol)</th>
                        <th>Median</th>
                        <th>Max Gain (Best Day)</th>
                        <th>Max Loss (Worst Day)</th>
                        <th>Positive Days (% Win Rate)</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
        """)

    # Registered Benchmarks Table
    @render.ui
    def registered_bms_table_ui():
        meta = benchmark_metadata()
        bm_df = meta["bm_info_df"]
        if bm_df.empty:
            return ui.HTML('<div class="text-muted">No registered benchmarks in the database.</div>')

        rows_html = "".join([
            f"""
            <tr>
                <td><b>{r['BENCHMARK_CODE']}</b></td>
                <td>{r.get('NAME', '')}</td>
                <td><code>{r.get('CONSTITUENTS_DISPLAY', '')}</code></td>
                <td><small>{r.get('DESCRIPTION', '')}</small></td>
            </tr>
            """ for _, r in bm_df.iterrows()
        ])

        return ui.HTML(f"""
        <div style="overflow-x: auto; max-height: 250px;">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Code</th>
                        <th>Name</th>
                        <th>Constituents</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """)

    # Shadow Transactions Table
    @render.ui
    def shadow_tx_table_ui():
        _ = bm_version()
        bm_tx_all = fetch_benchmark_transactions(engine=engine)
        if bm_tx_all.empty:
            return ui.HTML('<div class="alert alert-info">No benchmark shadow transactions recorded yet.</div>')

        rows_html = "".join([
            f"""
            <tr>
                <td>{r.get('BENCHMARK_CODE', '')}</td>
                <td>{r.get('TICKER', '')}</td>
                <td>{str(r.get('TRANSACTION_DATE', ''))[:10]}</td>
                <td>{float(r.get('QUANTITY', 0)):,.4f}</td>
                <td>£{float(r.get('PRICE_GBP', 0)):,.2f}</td>
                <td><b>£{float(r.get('GBP_VALUE', 0)):,.2f}</b></td>
                <td>{float(r.get('WEIGHT', 0))*100:.1f}%</td>
            </tr>
            """ for _, r in bm_tx_all.iterrows()
        ])

        return ui.HTML(f"""
        <div style="overflow-x: auto; max-height: 350px;">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>Ticker</th>
                        <th>Date</th>
                        <th>Quantity</th>
                        <th>Price (GBP)</th>
                        <th>Invested (GBP)</th>
                        <th>Weight</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """)

