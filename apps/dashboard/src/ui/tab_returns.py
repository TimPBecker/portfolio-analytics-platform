"""
Tab 4: Stock Price Levels, Returns & Distribution Histogram View (Shiny Module).
Interactive querying of individual asset price trajectories, daily return series,
empirical histograms with fitted KDE/Normal/Student-t densities, and statistical normality diagnostics.
"""

from typing import List, Optional, Dict, Any, Callable
import pandas as pd
import numpy as np
from shiny import module, ui, render, reactive
from shinywidgets import output_widget, render_plotly
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
try:
    from src.ui.theme import PALETTE, get_plotly_layout_defaults, render_metric_card
except ImportError:
    from apps.dashboard.src.ui.theme import PALETTE, get_plotly_layout_defaults, render_metric_card



@module.ui
def tab_returns_ui():
    """UI layout for Asset Levels, Returns and Distribution Diagnostics."""
    return ui.TagList(
        ui.tags.div(
            ui.tags.h3("📊 Asset Levels, Returns & Distribution Diagnostics", style="margin-bottom: 4px;"),
            ui.tags.p(
                "Inspect stock price levels, daily return time-series, empirical histograms with fitted probability density functions, and tail-fatness diagnostics.",
                class_="text-muted",
                style="margin-bottom: 1.2rem;"
            )
        ),

        # 1. Controls Bar
        ui.row(
            ui.column(3, ui.output_ui("ticker_select_ui")),
            ui.column(3, ui.input_select("horizon_choice", "Time Horizon (Returns):", ["1 Month", "3 Months", "6 Months", "1 Year (Default)", "2 Years", "All Available", "Custom Range"], selected="1 Year (Default)")),
            ui.column(3, ui.input_select("currency_mode", "Currency Display:", ["GBP Converted (£)", "Native Currency"], selected="GBP Converted (£)")),
            ui.column(3, ui.input_select("return_type", "Return Type:", ["Log Returns (ln)", "Simple Returns (%)"], selected="Log Returns (ln)"))
        ),
        ui.output_ui("custom_date_range_ui"),
        ui.tags.div(style="margin-bottom: 0.6rem;"),

        # Loading Animation Indicator Bar
        ui.tags.div(
            ui.tags.div(
                ui.tags.span(class_="spinner-border spinner-border-sm text-primary", role="status", style="width: 1.1rem; height: 1.1rem; border-width: 0.16em; margin-right: 8px; vertical-align: middle;"),
                ui.tags.span("Computing asset levels, daily return time-series, probability density fits & diagnostics...", style="vertical-align: middle; font-size: 0.88rem; font-weight: 600; color: #1E3A8A;"),
                class_="returns-loading-banner"
            )
        ),
        ui.tags.div(style="margin-bottom: 0.8rem;"),

        # 2. Key Statistical Moments Cards
        ui.output_ui("moments_kpis_ui"),
        ui.tags.div(style="margin-bottom: 1.8rem;"),

        # 3. Stock Level Chart with Moving Averages & Volume
        ui.tags.div(
            ui.tags.h4("📈 Price Level Trajectory & Moving Averages", style="margin-bottom: 8px;"),
            output_widget("price_level_chart"),
            style="margin-bottom: 1.8rem;"
        ),

        # 4. Daily Return Time Series
        ui.tags.div(
            ui.tags.h4("🌊 Daily Return Time Series & Confidence Bands", style="margin-bottom: 8px;"),
            output_widget("returns_ts_chart"),
            style="margin-bottom: 1.8rem;"
        ),

        # 5. Return Distribution Histogram & Fitted Probability Densities
        ui.tags.div(
            ui.tags.h4("📊 Empirical Return Distribution & Density Fitting", style="margin-bottom: 4px;"),
            ui.tags.p("Inspect the return frequency distribution with overlaid Kernel Density Estimation (KDE), Gaussian Normal fit, Student-t fit, and VaR cutoff thresholds.", class_="text-muted", style="font-size: 0.9rem;"),
            ui.row(
                ui.column(6, ui.input_slider("num_bins", "Histogram Bins:", min=15, max=80, value=35, step=5)),
                ui.column(6, ui.input_selectize(
                    "density_overlays",
                    "Overlaid Density Curves:",
                    choices=["Kernel Density (KDE)", "Fitted Normal PDF", "Fitted Student-t PDF"],
                    selected=["Kernel Density (KDE)", "Fitted Normal PDF", "Fitted Student-t PDF"],
                    multiple=True
                ))
            ),
            output_widget("density_chart"),
            style="margin-bottom: 1.8rem;"
        ),

        # 6. Q-Q Diagnostics & Top 5 Shocks
        ui.row(
            ui.column(
                5,
                ui.tags.h5("🔬 Normal Quantile-Quantile (Q-Q) Plot", style="margin-bottom: 8px;"),
                output_widget("qq_chart")
            ),
            ui.column(
                7,
                ui.tags.h5("⚡ Top 5 Best & Worst Single-Day Return Shocks", style="margin-bottom: 8px;"),
                ui.output_ui("shocks_table_ui")
            )
        ),
        ui.tags.div(style="margin-bottom: 2rem;"),

        # 7. Cross-Asset Return Correlation Matrix Heatmap
        ui.tags.div(
            ui.tags.hr(),
            ui.tags.h4("🔥 Cross-Asset Daily Return Correlation Heatmap", style="margin-bottom: 4px;"),
            ui.tags.p("Pearson correlation coefficients across multiple assets over the selected historical lookback window.", class_="text-muted", style="font-size: 0.9rem;"),
            ui.row(
                ui.column(8, ui.output_ui("corr_tickers_select_ui")),
                ui.column(4, ui.input_select("corr_window", "Correlation Lookback Window:", ["3 Months (63 Days)", "6 Months (126 Days)", "1 Year (260 Days)", "2 Years (520 Days)", "All Available"], selected="1 Year (260 Days)"))
            ),
            output_widget("corr_heatmap_chart")
        )
    )


@module.server
def tab_returns_server(input, output, session, shared_data: Callable[[], Dict[str, Any]]):
    """Server reactive logic for Returns tab."""

    @render.ui
    def ticker_select_ui():
        sdata = shared_data()
        tickers = sdata.get("available_tickers", [])
        default_t = "NVDA" if "NVDA" in tickers else (tickers[0] if tickers else "")
        return ui.input_select("selected_ticker", "Select Stock Ticker:", choices=tickers, selected=default_t)

    @render.ui
    def custom_date_range_ui():
        if input.horizon_choice() == "Custom Range":
            return ui.row(
                ui.column(6, ui.input_date("start_date", "Start Date (Asset):")),
                ui.column(6, ui.input_date("end_date", "End Date (Asset):"))
            )
        return ui.HTML("")

    # Load Raw Asset Data
    @reactive.calc
    def asset_data():
        ticker = input.selected_ticker()
        if not ticker:
            return None

        raw_df = fetch_raw_asset_prices(ticker)
        if raw_df.empty:
            return None

        raw_df = raw_df.set_index("DATE").sort_index()
        max_date = raw_df.index.max()
        horizon_choice = input.horizon_choice()

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
        elif horizon_choice == "Custom Range":
            start_date = pd.to_datetime(input.start_date() or (max_date - pd.Timedelta(days=365)))
            max_date = pd.to_datetime(input.end_date() or max_date)
        else:
            start_date = max_date - pd.Timedelta(days=365)

        filtered_df = raw_df.loc[start_date:max_date].copy()
        if len(filtered_df) < 5:
            return None

        currency_mode = input.currency_mode() or "GBP Converted (£)"
        native_currency = filtered_df["CURRENCY"].iloc[-1] if "CURRENCY" in filtered_df.columns else "USD"

        if currency_mode.startswith("GBP"):
            price_series = filtered_df["CLOSE_GBP"]
            price_unit = "£"
            curr_label = "GBP"
        else:
            price_series = filtered_df["CLOSE"]
            price_unit = "" if native_currency in ["GBp", "GBX"] else ("$" if native_currency == "USD" else "€")
            curr_label = native_currency

        return_type = input.return_type() or "Log Returns (ln)"
        method_key = "log" if "Log" in return_type else "simple"
        returns_series = compute_asset_returns(price_series, method=method_key)
        dist_metrics = compute_distribution_metrics(returns_series)

        return {
            "ticker": ticker,
            "filtered_df": filtered_df,
            "price_series": price_series,
            "price_unit": price_unit,
            "curr_label": curr_label,
            "returns_series": returns_series,
            "dist_metrics": dist_metrics,
            "method_key": method_key,
            "horizon_choice": horizon_choice
        }

    # 2. Key Statistical Moments Cards
    @render.ui
    def moments_kpis_ui():
        adata = asset_data()
        if not adata:
            return ui.HTML('<div class="alert alert-warning">No price records found or insufficient data for selected asset and horizon.</div>')

        price_series = adata["price_series"]
        price_unit = adata["price_unit"]
        curr_label = adata["curr_label"]
        dist_metrics = adata["dist_metrics"]
        returns_series = adata["returns_series"]

        ann_vol = dist_metrics.get("vol_annualized_pct", 0.0)
        daily_vol = dist_metrics.get("std_daily_pct", 0.0)

        skew = dist_metrics.get("skewness", 0.0)
        skew_label = "Left-skewed (Negative)" if skew < -0.2 else ("Right-skewed (Positive)" if skew > 0.2 else "Symmetric")

        kurt = dist_metrics.get("kurtosis_excess", 0.0)
        kurt_label = "Fat-Tailed (Leptokurtic)" if kurt > 0.5 else ("Thin-Tailed (Platykurtic)" if kurt < -0.5 else "Mesokurtic")

        is_norm = dist_metrics.get("is_normal", False)
        jb_p = dist_metrics.get("jb_pvalue", 0.0)

        c1 = render_metric_card("Latest Price", f"{price_unit}{price_series.iloc[-1]:,.2f} <small style='color:#64748B;'>{curr_label}</small>", f"Obs: {dist_metrics.get('count', len(returns_series))} days")
        c2 = render_metric_card("Annualized Volatility", f"{ann_vol:.2f}%", f"Daily Std: {daily_vol:.2f}%")
        c3 = render_metric_card("Skewness", f"{skew:+.2f}", skew_label)
        c4 = render_metric_card("Excess Kurtosis", f"{kurt:+.2f}", kurt_label)
        c5 = render_metric_card("Normality (Jarque-Bera)", "Pass" if is_norm else "Reject", f"p-val: {jb_p:.2e}")

        return ui.HTML(f"""
        <div class="row g-3">
            <div class="col">{c1}</div>
            <div class="col">{c2}</div>
            <div class="col">{c3}</div>
            <div class="col">{c4}</div>
            <div class="col">{c5}</div>
        </div>
        """)

    # 3. Stock Level Chart with Moving Averages & Volume
    @render_plotly
    def price_level_chart():
        adata = asset_data()
        if not adata:
            fig = go.Figure()
            layout = get_plotly_layout_defaults()
            layout.update(dict(
                title=dict(text="Loading & computing price level trajectory...", font=dict(size=13, color="#64748B")),
                height=450,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False)
            ))
            fig.update_layout(**layout)
            return fig

        filtered_df = adata["filtered_df"]
        price_series = adata["price_series"]
        price_unit = adata["price_unit"]
        curr_label = adata["curr_label"]
        ticker = adata["ticker"]
        returns_series = adata["returns_series"]

        has_volume = "VOLUME" in filtered_df.columns and filtered_df["VOLUME"].sum() > 0

        if has_volume:
            fig_price = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.04,
                row_heights=[0.75, 0.25]
            )
        else:
            fig_price = go.Figure()

        sma_20 = price_series.rolling(20, min_periods=5).mean()
        sma_50 = price_series.rolling(50, min_periods=10).mean()
        sma_200 = price_series.rolling(200, min_periods=20).mean()

        trace_price = go.Scatter(
            x=price_series.index, y=price_series.values,
            name=f"{ticker} Close ({curr_label})",
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
            title=dict(text=f"<b>{ticker}</b> — Price History ({curr_label})", font=dict(size=14, color="#0F172A")),
            height=450
        ))
        fig_price.update_layout(**layout_p)
        if has_volume:
            fig_price.update_yaxes(title_text=f"Price ({curr_label})", row=1, col=1)
            fig_price.update_yaxes(title_text="Volume", row=2, col=1)
        else:
            fig_price.update_yaxes(title_text=f"Price ({curr_label})")

        return fig_price

    # 4. Daily Return Time Series
    @render_plotly
    def returns_ts_chart():
        adata = asset_data()
        if not adata:
            fig = go.Figure()
            layout = get_plotly_layout_defaults()
            layout.update(dict(
                title=dict(text="Loading daily return time series...", font=dict(size=13, color="#64748B")),
                height=350,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False)
            ))
            fig.update_layout(**layout)
            return fig

        returns_series = adata["returns_series"]
        ticker = adata["ticker"]
        method_key = adata["method_key"]

        fig_rets = go.Figure()
        std_val = float(returns_series.std())
        mean_val = float(returns_series.mean())

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

        fig_rets.add_hline(y=(mean_val + 2 * std_val) * 100.0, line_dash="dash", line_color="#D97706", line_width=1.2, annotation_text=f"+2σ (+{(mean_val + 2 * std_val)*100:.1f}%)", annotation_position="top right")
        fig_rets.add_hline(y=(mean_val - 2 * std_val) * 100.0, line_dash="dash", line_color="#D97706", line_width=1.2, annotation_text=f"-2σ ({(mean_val - 2 * std_val)*100:.1f}%)", annotation_position="bottom right")
        fig_rets.add_hline(y=0.0, line_color="#64748B", line_width=1.0)

        layout_rets = get_plotly_layout_defaults()
        layout_rets.update(dict(
            title=dict(text=f"<b>{ticker}</b> — Daily Returns Time Series ({'Log' if method_key == 'log' else 'Percentage'})", font=dict(size=14, color="#0F172A")),
            height=350
        ))
        fig_rets.update_layout(**layout_rets)
        fig_rets.update_yaxes(title_text="Daily Return (%)", ticksuffix="%")
        return fig_rets

    # 5. Return Distribution Histogram & Fitted Density Curves
    @render_plotly
    def density_chart():
        adata = asset_data()
        if not adata:
            fig = go.Figure()
            layout = get_plotly_layout_defaults()
            layout.update(dict(
                title=dict(text="Computing empirical distribution & fitting density curves...", font=dict(size=13, color="#64748B")),
                height=460,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False)
            ))
            fig.update_layout(**layout)
            return fig

        returns_series = adata["returns_series"]
        ticker = adata["ticker"]
        dist_metrics = adata["dist_metrics"]
        num_bins = int(input.num_bins() or 35)
        density_overlays = input.density_overlays() or []

        x_grid, kde_y, norm_y, t_y = generate_density_curves(returns_series)

        fig_density = go.Figure()

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

        if "Kernel Density (KDE)" in density_overlays and len(x_grid) > 0:
            fig_density.add_trace(
                go.Scatter(
                    x=x_grid * 100.0, y=kde_y / 100.0,
                    name="Kernel Density (KDE)",
                    line=dict(color="#1E3A8A", width=2.6),
                    hovertemplate="KDE Density: %{y:.4f}<extra></extra>"
                )
            )

        if "Fitted Normal PDF" in density_overlays and len(x_grid) > 0:
            fig_density.add_trace(
                go.Scatter(
                    x=x_grid * 100.0, y=norm_y / 100.0,
                    name=f"Fitted Normal (μ={dist_metrics.get('mean_daily_pct', 0):.2f}%, σ={dist_metrics.get('std_daily_pct', 0):.2f}%)",
                    line=dict(color="#DC2626", width=2.2, dash="dash"),
                    hovertemplate="Normal Fit Density: %{y:.4f}<extra></extra>"
                )
            )

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

        p05 = dist_metrics.get("p05_pct", 0.0)
        p01 = dist_metrics.get("p01_pct", 0.0)

        fig_density.add_vline(x=p05, line_dash="dashdot", line_color="#F59E0B", line_width=1.8, annotation_text=f"95% VaR ({p05:.2f}%)", annotation_position="top left")
        fig_density.add_vline(x=p01, line_dash="dashdot", line_color="#DC2626", line_width=1.8, annotation_text=f"99% VaR ({p01:.2f}%)", annotation_position="top left")

        layout_dens = get_plotly_layout_defaults()
        layout_dens.update(dict(
            title=dict(text=f"<b>{ticker}</b> — Empirical Return Histogram & Probability Density Fits", font=dict(size=14, color="#0F172A")),
            height=460,
        ))
        fig_density.update_layout(**layout_dens)
        fig_density.update_xaxes(title_text="Daily Return (%)", ticksuffix="%")
        fig_density.update_yaxes(title_text="Probability Density")
        return fig_density

    # 6. Q-Q Diagnostics & Top 5 Shocks Table
    @render_plotly
    def qq_chart():
        adata = asset_data()
        if not adata:
            fig = go.Figure()
            layout = get_plotly_layout_defaults()
            layout.update(dict(
                title=dict(text="Calculating Q-Q normality diagnostics...", font=dict(size=12, color="#64748B")),
                height=380,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False, showticklabels=False)
            ))
            fig.update_layout(**layout)
            return fig

        returns_series = adata["returns_series"]
        osm, osr, slope, intercept = compute_qq_plot_data(returns_series)

        if len(osm) == 0:
            return go.Figure()

        fig_qq = go.Figure()
        fig_qq.add_trace(
            go.Scatter(
                x=osm, y=osr * 100.0,
                mode="markers",
                name="Sample Quantiles",
                marker=dict(color="#1E3A8A", size=5, opacity=0.75)
            )
        )
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
        return fig_qq

    @render.ui
    def shocks_table_ui():
        adata = asset_data()
        if not adata:
            return ui.HTML('<div class="text-muted">No data.</div>')

        returns_series = adata["returns_series"]
        std_val = float(returns_series.std())

        best_5 = returns_series.sort_values(ascending=False).head(5)
        worst_5 = returns_series.sort_values(ascending=True).head(5)

        shock_records = []
        for d, r in best_5.items():
            d_str = pd.to_datetime(d).strftime("%Y-%m-%d")
            impact_str = f"{abs(r) / std_val:.1f}σ" if std_val > 0 else "-"
            shock_records.append({"date": d_str, "type": "Gain 🟢", "ret": f"{r * 100.0:+.2f}%", "mag": impact_str, "cls": "text-success"})
        for d, r in worst_5.items():
            d_str = pd.to_datetime(d).strftime("%Y-%m-%d")
            impact_str = f"{abs(r) / std_val:.1f}σ" if std_val > 0 else "-"
            shock_records.append({"date": d_str, "type": "Loss 🔴", "ret": f"{r * 100.0:+.2f}%", "mag": impact_str, "cls": "text-danger"})

        rows_html = "".join([
            f"""
            <tr>
                <td>{r['date']}</td>
                <td>{r['type']}</td>
                <td><b class="{r['cls']}">{r['ret']}</b></td>
                <td>{r['mag']}</td>
            </tr>
            """ for r in shock_records
        ])

        return ui.HTML(f"""
        <div style="overflow-x: auto; max-height: 380px;">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Type</th>
                        <th>Daily Return</th>
                        <th>Magnitude</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """)

    # 7. Cross-Asset Return Correlation Matrix Heatmap
    @render.ui
    def corr_tickers_select_ui():
        sdata = shared_data()
        prices_gbp = sdata.get("prices_gbp", pd.DataFrame())
        available_tickers = sdata.get("available_tickers", [])
        default_corr = [t for t in available_tickers if t in prices_gbp.columns][:15]
        return ui.input_selectize(
            "corr_tickers",
            "Select Assets for Correlation Matrix:",
            choices=list(prices_gbp.columns),
            selected=default_corr,
            multiple=True
        )

    @render_plotly
    def corr_heatmap_chart():
        sdata = shared_data()
        prices_gbp = sdata.get("prices_gbp", pd.DataFrame())
        corr_tickers = input.corr_tickers() or []
        corr_window = input.corr_window() or "1 Year (260 Days)"

        if prices_gbp.empty or len(corr_tickers) < 2:
            fig = go.Figure()
            fig.update_layout(title="Please select at least 2 assets to generate the correlation matrix.")
            return fig

        window_days = 260
        if "3 Months" in corr_window:
            window_days = 63
        elif "6 Months" in corr_window:
            window_days = 126
        elif "2 Years" in corr_window:
            window_days = 520
        elif "All Available" in corr_window:
            window_days = len(prices_gbp)

        valid_tickers = [t for t in corr_tickers if t in prices_gbp.columns]
        if len(valid_tickers) < 2:
            return go.Figure()

        p_sub = prices_gbp[valid_tickers].iloc[-window_days:]
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
            height=min(650, max(380, len(valid_tickers) * 32 + 100)),
            margin=dict(l=40, r=40, t=30, b=40)
        ))
        fig_corr.update_layout(**layout_corr)
        return fig_corr
