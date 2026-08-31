"""
Portfolio Risk & Volatility Analytics Dashboard (Main Entrypoint).
Interactive web application powered by Shiny for Python, Plotly, and SQLAlchemy.
"""

from typing import Dict, List, Tuple, Any
import pandas as pd
from shiny import App, ui, render, reactive
from shinywidgets import output_widget

from portfolio_core.config import config
from portfolio_core.db import (
    get_engine,
    test_db_connection,
    fetch_available_tickers,
    fetch_historical_prices_gbp,
    fetch_portfolio_positions,
    fetch_available_var_dates
)
try:
    from src.ui.theme import custom_css_header, render_metric_card
    from src.ui.tab_portfolio import tab_portfolio_ui, tab_portfolio_server
    from src.ui.tab_benchmarks import tab_benchmarks_ui, tab_benchmarks_server
    from src.ui.tab_var import tab_var_ui, tab_var_server
    from src.ui.tab_returns import tab_returns_ui, tab_returns_server
    from src.ui.tab_volatility import tab_volatility_ui, tab_volatility_server
    from src.ui.tab_transactions import tab_transactions_ui, tab_transactions_server
except ImportError:
    from apps.dashboard.src.ui.theme import custom_css_header, render_metric_card
    from apps.dashboard.src.ui.tab_portfolio import tab_portfolio_ui, tab_portfolio_server
    from apps.dashboard.src.ui.tab_benchmarks import tab_benchmarks_ui, tab_benchmarks_server
    from apps.dashboard.src.ui.tab_var import tab_var_ui, tab_var_server
    from apps.dashboard.src.ui.tab_returns import tab_returns_ui, tab_returns_server
    from apps.dashboard.src.ui.tab_volatility import tab_volatility_ui, tab_volatility_server
    from apps.dashboard.src.ui.tab_transactions import tab_transactions_ui, tab_transactions_server

app_title = config.ui_config.get("app_title", "Portfolio Risk & Volatility Analytics")

# -----------------------------------------------------------------------------
# 1. UI Definition
# -----------------------------------------------------------------------------
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.tags.div(
            ui.tags.h3("📈 Risk Analytics", style="margin-bottom: 2px; color: #0F172A; font-size: 1.2rem; font-weight: 700;"),
            ui.tags.p("Quantitative Portfolio Risk, Volatility & Return Modeling", class_="text-muted", style="font-size: 0.82rem; margin-bottom: 12px;"),
            ui.tags.hr(style="margin: 8px 0;"),
            ui.output_ui("sidebar_db_status_ui"),
            ui.tags.div(style="margin-top: 12px;"),
            ui.tags.h6("⚙️ Environment Details", style="font-size: 0.88rem; font-weight: 600; color: #334155;"),
            ui.output_ui("sidebar_env_details_ui"),
            ui.tags.div(style="margin-top: 12px;"),
            ui.input_action_button("btn_refresh_cache", "🔄 Refresh Data Cache", class_="btn-sm btn-outline-secondary w-100"),
            ui.tags.hr(style="margin: 12px 0;"),
            ui.tags.h6("📚 Quick Guide", style="font-size: 0.88rem; font-weight: 600; color: #334155;"),
            ui.tags.ul(
                ui.tags.li(ui.tags.b("Holdings:"), " Allocation & historical valuation trajectory."),
                ui.tags.li(ui.tags.b("Benchmarking:"), " Opportunity cost vs S&P 500 / FTSE 100."),
                ui.tags.li(ui.tags.b("VaR Spectrum:"), " Volatility-scaled tail risk (1% to 99%)."),
                ui.tags.li(ui.tags.b("Returns & Hist:"), " Stock levels, returns & KDE / t-fits."),
                ui.tags.li(ui.tags.b("Rolling Vol:"), " Multi-estimator EWMA vs rolling windows."),
                ui.tags.li(ui.tags.b("Trades:"), " Record trades with live Yahoo quote lookup."),
                style="font-size: 0.78rem; color: #64748B; padding-left: 1.1rem; margin-bottom: 0;"
            ),
            ui.tags.p("Developed with Shiny for Python & Plotly", class_="text-muted", style="font-size: 0.72rem; margin-top: 12px;")
        ),
        title=None,
        width=300,
        open="closed"
    ),
    custom_css_header(),
    ui.busy_indicators.use(spinners=True, pulse=True, fade=True),

    # Top Header & KPI Strip
    ui.tags.div(
        ui.tags.h2("📈 Portfolio Risk & Volatility Analytics", style="margin-bottom: 4px; font-weight: 700; color: #0F172A; font-size: 1.6rem;"),
        ui.tags.p("Interactive quantitative risk suite for rolling volatilities, tail risk percentiles, and return distribution modeling.", class_="text-muted", style="margin-bottom: 1rem;"),
        ui.output_ui("top_kpi_strip_ui"),
        style="margin-bottom: 1.2rem;"
    ),

    # Main Navigation Tabs
    ui.navset_tab(
        ui.nav_panel("💼 Portfolio Holdings & Valuation", tab_portfolio_ui("tab_portfolio")),
        ui.nav_panel("🎯 Benchmarking", tab_benchmarks_ui("tab_benchmarks")),
        ui.nav_panel("🛡️ Value-at-Risk Spectrum", tab_var_ui("tab_var")),
        ui.nav_panel("📊 Levels, Returns & Histogram", tab_returns_ui("tab_returns")),
        ui.nav_panel("📈 Rolling Volatility", tab_volatility_ui("tab_volatility")),
        ui.nav_panel("📝 Transaction Entry", tab_transactions_ui("tab_transactions")),
        id="main_navset"
    ),
    title=app_title,
    fillable=False
)


# -----------------------------------------------------------------------------
# 2. Server Definition
# -----------------------------------------------------------------------------
def server(input, output, session):
    cache_trigger = reactive.value(0)

    @reactive.effect
    @reactive.event(input.btn_refresh_cache)
    def _handle_refresh():
        cache_trigger.set(cache_trigger() + 1)

    # Core Reactive Data Loader
    @reactive.calc
    def core_market_data() -> Dict[str, Any]:
        _ = cache_trigger()
        engine = get_engine()
        db_connected, db_msg = test_db_connection(engine=engine)
        tickers = fetch_available_tickers(engine=engine)
        prices_gbp = fetch_historical_prices_gbp(engine=engine)
        positions = fetch_portfolio_positions(engine=engine)
        var_dates = fetch_available_var_dates(engine=engine)
        latest_date_str = str(prices_gbp.index[-1])[:10] if not prices_gbp.empty else "N/A"

        return {
            "db_connected": db_connected,
            "db_msg": db_msg,
            "available_tickers": tickers,
            "prices_gbp": prices_gbp,
            "positions": positions,
            "var_dates": var_dates,
            "asof_date": latest_date_str
        }

    # Sidebar DB Status
    @render.ui
    def sidebar_db_status_ui():
        cdata = core_market_data()
        if cdata["db_connected"]:
            return ui.HTML('<div class="alert alert-success py-1 px-2" style="font-size: 0.82rem; margin-bottom: 0;">🟢 <b>Database Connected</b></div>')
        else:
            return ui.HTML(f'<div class="alert alert-danger py-1 px-2" style="font-size: 0.82rem; margin-bottom: 0;">🔴 <b>DB Failed:</b> {cdata["db_msg"]}</div>')

    # Sidebar Environment Details
    @render.ui
    def sidebar_env_details_ui():
        db_cfg = config.db_config
        return ui.HTML(f"""
        <div style="font-size: 0.8rem; color: #64748B;">
            <div>• Backend: <code>{db_cfg.get('type', 'mariadb')}</code></div>
            <div>• Host: <code>{db_cfg.get('host', 'local')}</code></div>
            <div>• Database: <code>{db_cfg.get('database', 'stocks')}</code></div>
        </div>
        """)

    # Top KPI Header Strip
    @render.ui
    def top_kpi_strip_ui():
        cdata = core_market_data()
        prices_gbp = cdata["prices_gbp"]
        tickers = cdata["available_tickers"]
        positions = cdata["positions"]
        latest_date = cdata["asof_date"]

        c1 = render_metric_card("Latest Market Date", latest_date)
        c2 = render_metric_card("Active Tickers in DB", f"{len(tickers)} assets")
        c3 = render_metric_card("Tracked Holdings", f"{len(positions)} positions")

        return ui.HTML(f"""
        <div class="row g-3">
            <div class="col-md-4">{c1}</div>
            <div class="col-md-4">{c2}</div>
            <div class="col-md-4">{c3}</div>
        </div>
        """)

    # Initialize All 6 Tab Servers
    tab_portfolio_server("tab_portfolio", shared_data=core_market_data)
    tab_benchmarks_server("tab_benchmarks", shared_data=core_market_data)
    tab_var_server("tab_var", shared_data=core_market_data)
    tab_returns_server("tab_returns", shared_data=core_market_data)
    tab_volatility_server("tab_volatility", shared_data=core_market_data)
    tab_transactions_server("tab_transactions", shared_data=core_market_data)


# -----------------------------------------------------------------------------
# 3. Shiny App Instance
# -----------------------------------------------------------------------------
app = App(app_ui, server)
