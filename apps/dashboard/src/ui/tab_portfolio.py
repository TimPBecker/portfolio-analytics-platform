"""
Tab 1: Portfolio Holdings, Allocations, Top Movers & Historical Valuation View (Shiny Module).
"""

from typing import List, Optional, Dict, Any, Callable
import pandas as pd
import numpy as np
from shiny import module, ui, render, reactive
from shinywidgets import output_widget, render_plotly
import plotly.graph_objects as go

from portfolio_core.db import (
    fetch_portfolio_values_history
)
from portfolio_core.analytics.statistics import compute_top_position_movers

try:
    from src.ui.theme import PALETTE, get_plotly_layout_defaults, render_metric_card
    from src.services.report_generator import generate_portfolio_pdf_report
except ImportError:
    from apps.dashboard.src.ui.theme import PALETTE, get_plotly_layout_defaults, render_metric_card
    from apps.dashboard.src.services.report_generator import generate_portfolio_pdf_report


@module.ui
def tab_portfolio_ui():
    """UI layout for Portfolio Stock Allocations, Top Movers, and Valuation History."""
    return ui.TagList(
        ui.tags.div(
            ui.tags.div(
                ui.tags.div(
                    ui.tags.h3("💼 Portfolio Holdings & Historical Valuation", style="margin-bottom: 4px;"),
                    ui.tags.p("Overview of stock holdings valuation, top daily movers, weight allocation breakdown, and historical trajectory.", class_="text-muted", style="margin-bottom: 0;"),
                ),
                ui.tags.div(
                    ui.download_button(
                        "download_pdf_report",
                        "📄 Export PDF Report",
                        class_="btn btn-primary",
                        style="white-space: nowrap;"
                    ),
                    style="display: flex; align-items: center;"
                ),
                style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.2rem; flex-wrap: wrap; gap: 12px;"
            )
        ),
        
        # 1. Top Valuation KPIs
        ui.output_ui("kpi_cards_ui"),
        ui.tags.div(style="margin-bottom: 1.5rem;"),

        # 2. Top 10 Movers
        ui.tags.div(
            ui.tags.h4("🚀 Top 10 Movers (Absolute Terms)", style="margin-bottom: 4px;"),
            ui.tags.p(
                "Top 10 holdings with the largest day-over-day value movement, ranked by absolute monetary movement (|Δ Value|).",
                class_="text-muted",
                style="font-size: 0.9rem;"
            ),
            ui.row(
                ui.column(7, ui.output_ui("movers_table_ui")),
                ui.column(5, output_widget("movers_chart"))
            ),
            style="margin-bottom: 1.5rem;"
        ),

        # 3. Allocation Donut & Holdings Table
        ui.tags.div(
            ui.row(
                ui.column(
                    6,
                    ui.tags.h4("🥧 Asset Weight Allocation", style="margin-bottom: 8px;"),
                    output_widget("donut_chart")
                ),
                ui.column(
                    6,
                    ui.tags.h4("📋 Current Portfolio Positions", style="margin-bottom: 8px;"),
                    ui.output_ui("holdings_table_ui")
                )
            ),
            style="margin-bottom: 1.5rem;"
        ),

        # 4. Historical Portfolio Valuation Timeline
        ui.tags.div(
            ui.tags.h4("📈 Portfolio Valuation Trajectory", style="margin-bottom: 8px;"),
            output_widget("trajectory_chart")
        )
    )


@module.server
def tab_portfolio_server(input, output, session, shared_data: Callable[[], Dict[str, Any]]):
    """Server reactive logic for Portfolio tab."""

    @reactive.calc
    def portfolio_data():
        data = shared_data()
        prices_gbp = data.get("prices_gbp", pd.DataFrame())
        positions = data.get("positions", {})
        asof_date = data.get("asof_date", None)

        if prices_gbp.empty or not positions:
            return None

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
            return None

        latest_prices = active_prices.iloc[-1]
        active_pos = {t: float(sh) for t, sh in positions.items() if t in latest_prices and sh > 0}
        pos_values = pd.Series({t: active_pos[t] * float(latest_prices[t]) for t in active_pos})
        total_stock_value = float(pos_values.sum())

        pv_df = fetch_portfolio_values_history(days=365, asof_date=asof_date)

        return {
            "active_prices": active_prices,
            "positions": positions,
            "active_pos": active_pos,
            "pos_values": pos_values,
            "total_stock_value": total_stock_value,
            "asof_date": asof_date,
            "latest_prices": latest_prices,
            "pv_df": pv_df
        }

    # PDF Export Handler
    @render.download(
        filename=lambda: f"Portfolio_Analytics_Report_{str(shared_data().get('asof_date', 'latest'))[:10]}.pdf",
        media_type="application/pdf"
    )
    def download_pdf_report():
        asof_date = shared_data().get("asof_date", None)
        ok, path, pdf_bytes, err = generate_portfolio_pdf_report(asof_date=asof_date)
        if ok and pdf_bytes:
            return pdf_bytes
        raise Exception(f"Failed to generate PDF report: {err}")

    # 1. Top Valuation KPIs
    @render.ui
    def kpi_cards_ui():
        pdata = portfolio_data()
        if not pdata:
            return ui.HTML('<div class="alert alert-warning">Insufficient price data or active positions.</div>')

        pv_df = pdata["pv_df"]
        total_stock_value = pdata["total_stock_value"]
        active_pos = pdata["active_pos"]
        active_prices = pdata["active_prices"]

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

        delta_str = f"{diff_stocks_val:+,.2f} ({diff_stocks_pct:+.2f}%)" if prev_stocks_val is not None else None
        delta_type = "positive" if diff_stocks_val > 0 else ("negative" if diff_stocks_val < 0 else "neutral")
        diff_sign = "+" if diff_stocks_val > 0 else ("-" if diff_stocks_val < 0 else "")

        card1 = render_metric_card(f"Current Value ({curr_date_str})", f"£{curr_stocks_val:,.2f}", delta_str, delta_type)
        card2 = render_metric_card(f"Previous Value ({prev_date_str})", f"£{prev_stocks_val:,.2f}" if prev_stocks_val is not None else "N/A")
        card3 = render_metric_card(
            "Day-over-Day Change",
            f"{diff_sign}£{abs(diff_stocks_val):,.2f}" if prev_stocks_val is not None else "£0.00",
            f"{diff_stocks_pct:+.2f}%" if prev_stocks_val is not None else None,
            delta_type
        )
        card4 = render_metric_card("Active Positions", f"{len(active_pos)} assets")

        return ui.HTML(f"""
        <div class="row g-3">
            <div class="col-md-3 col-sm-6">{card1}</div>
            <div class="col-md-3 col-sm-6">{card2}</div>
            <div class="col-md-3 col-sm-6">{card3}</div>
            <div class="col-md-3 col-sm-6">{card4}</div>
        </div>
        """)

    # 2. Top 10 Movers Table
    @render.ui
    def movers_table_ui():
        pdata = portfolio_data()
        if not pdata:
            return ui.HTML('<div class="text-muted">No movers data available.</div>')

        movers_df = compute_top_position_movers(
            prices_gbp=pdata["active_prices"],
            positions=pdata["positions"],
            asof_date=pdata["asof_date"],
            top_n=10
        )

        if movers_df.empty:
            return ui.HTML('<div class="alert alert-info">No movers data available for the selected dates.</div>')

        rows_html = []
        for _, r in movers_df.iterrows():
            sh = r["SHARES"]
            sh_str = f"{sh:,.2f}" if not sh.is_integer() else f"{sh:,.0f}"
            diff_sign = "+" if r["DIFF_GBP"] > 0 else ("-" if r["DIFF_GBP"] < 0 else "")
            pct_sign = "+" if r["DIFF_PCT"] > 0 else ("-" if r["DIFF_PCT"] < 0 else "")
            px_chg_sign = "+" if r["PRICE_CHG_PCT"] > 0 else ("-" if r["PRICE_CHG_PCT"] < 0 else "")
            color_class = "text-success" if r["DIFF_GBP"] >= 0 else "text-danger"

            rows_html.append(f"""
            <tr>
                <td><b>{r['TICKER']}</b></td>
                <td>{sh_str}</td>
                <td>£{r['PRICE_TODAY_GBP']:,.2f}</td>
                <td><span class="{color_class}">{px_chg_sign}{r['PRICE_CHG_PCT']:.2f}%</span></td>
                <td>£{r['VALUE_TODAY_GBP']:,.2f}</td>
                <td><b class="{color_class}">{diff_sign}£{abs(r['DIFF_GBP']):,.2f}</b></td>
                <td><span class="{color_class}">{pct_sign}{abs(r['DIFF_PCT']):.2f}%</span></td>
                <td>£{r['ABS_DIFF_GBP']:,.2f}</td>
            </tr>
            """)

        table_html = f"""
        <div style="overflow-x: auto; max-height: 400px;">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Shares</th>
                        <th>Price (£)</th>
                        <th>Price Chg</th>
                        <th>Value (£)</th>
                        <th>Day Change (£)</th>
                        <th>Day Change (%)</th>
                        <th>|Change| (£)</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
        """
        return ui.HTML(table_html)

    # 2. Top 10 Movers Chart
    @render_plotly
    def movers_chart():
        pdata = portfolio_data()
        if not pdata:
            return go.Figure()

        movers_df = compute_top_position_movers(
            prices_gbp=pdata["active_prices"],
            positions=pdata["positions"],
            asof_date=pdata["asof_date"],
            top_n=10
        )
        if movers_df.empty:
            return go.Figure()

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
        return fig_movers

    # 3. Allocation Donut Chart
    @render_plotly
    def donut_chart():
        pdata = portfolio_data()
        if not pdata:
            return go.Figure()

        pos_values = pdata["pos_values"]
        total_stock_value = pdata["total_stock_value"]
        top_holdings = pos_values.sort_values(ascending=False)

        # Only display slice labels for major holdings (>5% weight)
        total_val = float(top_holdings.values.sum())
        if total_val > 0:
            weights = top_holdings.values / total_val
            slice_labels = [
                f"<b>{ticker}</b><br>{wt * 100.0:.1f}%" if wt > 0.05 else ""
                for ticker, wt in zip(top_holdings.index, weights)
            ]
        else:
            slice_labels = [""] * len(top_holdings)

        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=top_holdings.index,
                    values=top_holdings.values,
                    hole=0.38,
                    text=slice_labels,
                    textinfo="text",
                    textposition="inside",
                    insidetextorientation="horizontal",
                    textfont=dict(size=12, family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
                    hovertemplate="<b>%{label}</b><br>Value: £%{value:,.2f}<br>Weight: %{percent}<extra></extra>"
                )
            ]
        )
        layout_donut = get_plotly_layout_defaults()
        layout_donut.update(dict(
            title=dict(text=f"Holdings Breakdown (Total: £{total_stock_value:,.2f})", font=dict(size=13, color="#0F172A")),
            height=460,
            margin=dict(l=15, r=15, t=35, b=15),
            showlegend=False
        ))
        fig_donut.update_layout(**layout_donut)
        return fig_donut


    # 3. Current Portfolio Positions Table
    @render.ui
    def holdings_table_ui():
        pdata = portfolio_data()
        if not pdata:
            return ui.HTML('<div class="text-muted">No active positions.</div>')

        pos_values = pdata["pos_values"]
        total_stock_value = pdata["total_stock_value"]
        active_pos = pdata["active_pos"]
        latest_prices = pdata["latest_prices"]
        top_holdings = pos_values.sort_values(ascending=False)

        rows_html = []
        for ticker, val in top_holdings.items():
            sh = active_pos[ticker]
            px_val = float(latest_prices[ticker])
            wt = (val / total_stock_value) * 100.0 if total_stock_value > 0 else 0.0
            sh_str = f"{sh:,.2f}" if not sh.is_integer() else f"{sh:,.0f}"

            rows_html.append(f"""
            <tr>
                <td><b>{ticker}</b></td>
                <td>{sh_str}</td>
                <td>£{px_val:,.2f}</td>
                <td><b>£{val:,.2f}</b></td>
                <td><span class="badge bg-light text-dark" style="font-size: 0.85rem; border: 1px solid #E2E8F0;">{wt:.2f}%</span></td>
            </tr>
            """)

        table_html = f"""
        <div style="overflow-x: auto; max-height: 450px;">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Shares</th>
                        <th>Price (£)</th>
                        <th>Value (£)</th>
                        <th>Weight (%)</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
        """
        return ui.HTML(table_html)

    # 4. Historical Portfolio Valuation Timeline
    @render_plotly
    def trajectory_chart():
        pdata = portfolio_data()
        if not pdata:
            return go.Figure()

        pv_df = pdata["pv_df"]
        if not pv_df.empty and len(pv_df) > 1:
            pv_df_sorted = pv_df.sort_values("DATE").reset_index(drop=True)
            stocks_series = pv_df_sorted["STOCKS"] if "STOCKS" in pv_df_sorted.columns else pv_df_sorted["TOTAL_VALUE"]

            fig_pv = go.Figure()
            fig_pv.add_trace(
                go.Scatter(
                    x=pv_df_sorted["DATE"],
                    y=stocks_series,
                    name="Stock Holdings Value (£)",
                    line=dict(color="#1E3A8A", width=3.0),
                    fill="tozeroy",
                    fillcolor="rgba(30, 58, 138, 0.08)",
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>Stock Holdings Value: <b>£%{y:,.2f}</b><extra></extra>"
                )
            )
            layout_pv = get_plotly_layout_defaults()
            layout_pv.update(dict(
                title=dict(text="Stock Holdings Valuation History (£)", font=dict(size=14, color="#0F172A")),
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            ))
            fig_pv.update_layout(**layout_pv)
            fig_pv.update_yaxes(title_text="Stock Value (£)", tickprefix="£")
            return fig_pv
        else:
            fig = go.Figure()
            fig.update_layout(title="No historical portfolio valuation data available.")
            return fig

