"""
Tab 6: Transaction Management & Trade Entry View (Shiny Module).
Allows recording new buy/sell transactions with automatic ID incrementation
and live Yahoo Finance closing price lookup.
"""

from datetime import date
from typing import Optional, Dict, Any, Callable
import pandas as pd
from shiny import module, ui, render, reactive

from portfolio_core.db import (
    get_engine,
    get_next_transaction_id,
    record_transaction,
    fetch_all_transactions,
    fetch_portfolio_positions,
    query_yahoo_close_price,
    fetch_and_store_ticker,
    fetch_and_store_fx_rate,
    calculate_and_store_daily_portfolio_values,
    calculate_and_store_daily_benchmark_values,
    get_latest_allowed_market_date
)
try:
    from src.ui.theme import PALETTE, render_metric_card
except ImportError:
    from apps.dashboard.src.ui.theme import PALETTE, render_metric_card



@module.ui
def tab_transactions_ui():
    """UI layout for Transaction Management and Trade Entry."""
    return ui.TagList(
        ui.tags.div(
            ui.tags.h3("📝 Record Portfolio Transactions", style="margin-bottom: 4px;"),
            ui.tags.p(
                "Enter new stock trade transactions into the database with real-time Yahoo Finance price lookup and automatic ID incrementation.",
                class_="text-muted",
                style="margin-bottom: 1.2rem;"
            )
        ),

        # 1. Trade Entry Form & Quote Preview
        ui.row(
            # Column 1: Trade Form
            ui.column(
                6,
                ui.tags.div(
                    ui.tags.h4("➕ New Trade Entry", style="margin-bottom: 12px;"),
                    ui.output_ui("next_id_ui"),
                    ui.row(
                        ui.column(6, ui.input_text("ticker_input", "Ticker Symbol:", value="NVDA", placeholder="e.g. NVDA, STAN.L")),
                        ui.column(6, ui.input_date("tx_date", "Transaction Date:", value=get_latest_allowed_market_date()))
                    ),

                    ui.row(
                        ui.column(6, ui.input_numeric("qty_input", "Quantity / Shares:", value=10.0, min=0.0001, step=1.0)),
                        ui.column(6, ui.input_radio_buttons("action", "Action:", {"BUY": "BUY (Add Shares)", "SELL": "SELL (Reduce Shares)"}, selected="BUY"))
                    ),
                    ui.input_checkbox("sync_prices", "Sync latest historical prices for this ticker into ASSET_PRICES", value=True),
                    ui.tags.div(style="margin-top: 10px;"),
                    ui.input_action_button("submit_btn", "💾 Record Transaction", class_="btn-primary w-100"),
                    ui.tags.div(style="margin-top: 10px;"),
                    ui.output_ui("tx_submit_result_ui"),
                    class_="metric-card",
                    style="margin-bottom: 1.5rem;"
                )
            ),
            # Column 2: Yahoo Live Price Preview
            ui.column(
                6,
                ui.tags.div(
                    ui.tags.h4("🔍 Yahoo Finance Market Quote & FX Translation", style="margin-bottom: 12px;"),
                    ui.output_ui("yahoo_preview_ui"),
                    class_="metric-card",
                    style="margin-bottom: 1.5rem;"
                )
            )
        ),

        ui.tags.div(style="margin-bottom: 1.5rem;"),

        # 2. Historical Transactions & Current Net Positions
        ui.row(
            ui.column(
                7,
                ui.tags.h4("📋 Recorded Transactions History", style="margin-bottom: 8px;"),
                ui.output_ui("recorded_tx_table_ui")
            ),
            ui.column(
                5,
                ui.tags.h4("📊 Current Net Share Positions", style="margin-bottom: 8px;"),
                ui.output_ui("net_positions_table_ui")
            )
        )
    )


@module.server
def tab_transactions_server(input, output, session, shared_data: Callable[[], Dict[str, Any]]):
    """Server reactive logic for Transactions tab."""
    refresh_trigger = reactive.value(0)
    submit_status = reactive.value(None)

    @render.ui
    def next_id_ui():
        _ = refresh_trigger()
        try:
            next_id = get_next_transaction_id()
            return ui.HTML(f"""
            <div style="margin-bottom: 12px;">
                <label class="form-label" style="font-size: 0.85rem; font-weight: 600; color: #64748B;">Transaction ID (Auto-Incremented):</label>
                <input class="form-control" type="text" value="#{next_id}" disabled readonly style="background-color: #F1F5F9; font-weight: 600;">
            </div>
            """)
        except Exception:
            return ui.HTML("")

    # Live Yahoo Quote Preview
    @reactive.calc
    def quote_preview_data():
        ticker = (input.ticker_input() or "").strip().upper()
        tx_date_val = input.tx_date() or date.today()
        qty = float(input.qty_input() or 0.0)

        if not ticker:
            return None

        quote_info = query_yahoo_close_price(ticker=ticker, target_date=tx_date_val)
        return {"quote_info": quote_info, "ticker": ticker, "qty": qty, "date": tx_date_val}

    @render.ui
    def yahoo_preview_ui():
        pdata = quote_preview_data()
        if not pdata:
            return ui.HTML('<div class="text-muted">Enter a ticker symbol to look up its market closing price and GBP translated quote.</div>')

        quote_info = pdata["quote_info"]
        ticker = pdata["ticker"]
        qty = pdata["qty"]

        if quote_info.get("status") != "success":
            msg = quote_info.get("message", f"Could not retrieve price for {ticker}.")
            return ui.HTML(f'<div class="alert alert-warning">{msg}</div>')

        px = float(quote_info["close_price"])
        curr = str(quote_info.get("currency", "USD"))
        found_dt = str(quote_info["found_date"])
        is_exact = bool(quote_info["is_exact_date"])
        fx_rate = float(quote_info.get("fx_rate_to_gbp", 1.0))
        fx_desc = str(quote_info.get("fx_description", ""))
        px_gbp = float(quote_info.get("close_price_gbp", px * fx_rate))

        total_val_orig = px * qty
        total_val_gbp = px_gbp * qty

        curr_sym = "£" if curr == "GBP" else ("p" if curr in ["GBp", "GBX"] else ("$" if curr == "USD" else ("€" if curr == "EUR" else f"{curr} ")))
        orig_quote_str = f"{px:,.2f} {curr}" if curr in ["GBp", "GBX"] else f"{curr_sym}{px:,.2f} {curr}"
        orig_val_str = f"{total_val_orig:,.2f} {curr}" if curr in ["GBp", "GBX"] else f"{curr_sym}{total_val_orig:,.2f} {curr}"

        fx_banner = (
            f'<div class="alert alert-info py-2" style="font-size: 0.85rem;">💱 <b>FX Rate:</b> 1 {curr} = <b>£{fx_rate:,.4f} GBP</b><br><i>{fx_desc}</i></div>'
            if curr != "GBP" else
            '<div class="alert alert-success py-2" style="font-size: 0.85rem;">💱 <b>Currency:</b> Native GBP (FX Rate: 1.00)</div>'
        )

        date_badge = '<span class="badge bg-success">Exact Match</span>' if is_exact else '<span class="badge bg-warning text-dark">Nearest Prior Market Day</span>'

        return ui.HTML(f"""
        <div>
            <div class="row g-2 mb-3">
                <div class="col-6">
                    <div class="p-2 border rounded" style="background-color: #F8FAFC;">
                        <small class="text-muted d-block">Price in GBP ({ticker})</small>
                        <b style="font-size: 1.1rem; color: #1E3A8A;">£{px_gbp:,.2f} GBP</b>
                    </div>
                </div>
                <div class="col-6">
                    <div class="p-2 border rounded" style="background-color: #F8FAFC;">
                        <small class="text-muted d-block">Original Quote ({curr})</small>
                        <b style="font-size: 1.1rem; color: #0F172A;">{orig_quote_str}</b>
                    </div>
                </div>
            </div>

            {fx_banner}

            <div class="mb-3" style="font-size: 0.85rem; color: #64748B;">
                📅 <b>Trading Date Found:</b> <code>{found_dt}</code> {date_badge}
            </div>

            <hr style="margin: 8px 0;">

            <div class="row g-2">
                <div class="col-6">
                    <div class="p-2 border rounded" style="background-color: #F1F5F9;">
                        <small class="text-muted d-block">Est. Total Value (GBP)</small>
                        <b style="font-size: 1.15rem; color: #10B981;">£{total_val_gbp:,.2f} GBP</b>
                        <small class="text-muted d-block" style="font-size: 0.75rem;">{qty:,.4f} shs @ £{px_gbp:,.2f}</small>
                    </div>
                </div>
                <div class="col-6">
                    <div class="p-2 border rounded" style="background-color: #F1F5F9;">
                        <small class="text-muted d-block">Est. Value ({curr})</small>
                        <b style="font-size: 1.15rem; color: #0F172A;">{orig_val_str}</b>
                        <small class="text-muted d-block" style="font-size: 0.75rem;">{qty:,.4f} shs @ {orig_quote_str}</small>
                    </div>
                </div>
            </div>
        </div>
        """)

    # Submission Effect
    @reactive.effect
    @reactive.event(input.submit_btn)
    def _handle_tx_submit():
        ticker = (input.ticker_input() or "").strip().upper()
        if not ticker:
            submit_status.set({"status": "error", "msg": "Please enter a valid ticker symbol."})
            return

        tx_date_val = input.tx_date() or date.today()
        qty = float(input.qty_input() or 0.0)
        action_val = input.action() or "BUY"
        signed_qty = qty if action_val == "BUY" else -abs(qty)
        sync_prices = bool(input.sync_prices())

        try:
            next_id = get_next_transaction_id()
            rec_res = record_transaction(
                ticker=ticker,
                transaction_date=tx_date_val,
                quantity=signed_qty,
                transaction_id=next_id
            )

            if sync_prices:
                try:
                    fetch_and_store_ticker(ticker=ticker)
                    fetch_and_store_fx_rate(from_curr="USD")
                    calculate_and_store_daily_portfolio_values(backfill_days=30)
                    calculate_and_store_daily_benchmark_values()
                except Exception as e:
                    pass

            submit_status.set({
                "status": "success",
                "msg": f"✅ Successfully recorded Transaction #{rec_res['id']}: {signed_qty:+,.4f} shares of {ticker} on {tx_date_val}!"
            })
            refresh_trigger.set(refresh_trigger() + 1)
        except Exception as ex:
            submit_status.set({"status": "error", "msg": f"Failed to record transaction: {ex}"})

    @render.ui
    def tx_submit_result_ui():
        st_val = submit_status()
        if not st_val:
            return ui.HTML("")
        cls = "alert alert-success" if st_val["status"] == "success" else "alert alert-danger"
        return ui.HTML(f'<div class="{cls} py-2" style="font-size: 0.9rem;">{st_val["msg"]}</div>')

    # Recorded Transactions Table
    @render.ui
    def recorded_tx_table_ui():
        _ = refresh_trigger()
        tx_df = fetch_all_transactions(limit=100)
        if tx_df.empty:
            return ui.HTML('<div class="text-muted">No transaction records found in the database.</div>')

        rows_html = "".join([
            f"""
            <tr>
                <td><b>#{int(r['ID'])}</b></td>
                <td><b>{str(r['TICKER'])}</b></td>
                <td>{str(r['TRANSACTION_DATE'])}</td>
                <td><span class="badge {'bg-success' if float(r['QUANTITY']) > 0 else 'bg-danger'}">{'BUY' if float(r['QUANTITY']) > 0 else 'SELL'}</span></td>
                <td>{f"+{float(r['QUANTITY']):,.2f}" if float(r['QUANTITY']) > 0 else f"{float(r['QUANTITY']):,.2f}"}</td>
            </tr>
            """ for _, r in tx_df.iterrows()
        ])

        return ui.HTML(f"""
        <div style="overflow-x: auto; max-height: 400px;">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Ticker</th>
                        <th>Date</th>
                        <th>Type</th>
                        <th>Quantity</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """)

    # Current Net Positions Table
    @render.ui
    def net_positions_table_ui():
        _ = refresh_trigger()
        positions = fetch_portfolio_positions()
        if not positions:
            return ui.HTML('<div class="text-muted">No active positions currently held.</div>')

        rows_html = "".join([
            f"""
            <tr>
                <td><b>{t}</b></td>
                <td><b>{sh:,.2f}</b></td>
            </tr>
            """ for t, sh in positions.items()
        ])

        return ui.HTML(f"""
        <div style="overflow-x: auto; max-height: 400px;">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Net Shares</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """)


