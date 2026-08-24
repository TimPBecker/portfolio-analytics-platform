"""
Portfolio Reporting and Telegram Notification Module for Portfolio-Analytics-Platform.
Generates multi-panel financial visual reports (Portfolio Valuations & Value-at-Risk)
and delivers them with formatted summary captions to configured Telegram recipients.
"""

import io
import os
import math
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import requests
from sqlalchemy import text

# Headless matplotlib backend for headless server / container environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


def fetch_recent_portfolio_values(
    days: int = 14,
    asof_date: Optional[str] = None,
    engine=None
) -> pd.DataFrame:
    """
    Fetches the most recent `days` daily portfolio valuation records from PORTFOLIO_VALUES
    up to `asof_date` (or latest available date if omitted).
    Returns chronologically sorted DataFrame (oldest to newest).
    """
    from portfolio_core.db import get_engine, create_all_tables
    engine = engine or get_engine()
    create_all_tables(engine)

    if asof_date:
        asof_str = pd.to_datetime(asof_date).strftime("%Y-%m-%d")
        query = """
            SELECT `DATE`, `TOTAL_VALUE`, `STOCKS`, `CASH`, `CURRENCY`
            FROM `PORTFOLIO_VALUES`
            WHERE `DATE` <= :asof
            ORDER BY `DATE` DESC
            LIMIT :limit
        """
        params = {"limit": max(1, days), "asof": asof_str}
    else:
        query = """
            SELECT `DATE`, `TOTAL_VALUE`, `STOCKS`, `CASH`, `CURRENCY`
            FROM `PORTFOLIO_VALUES`
            ORDER BY `DATE` DESC
            LIMIT :limit
        """
        params = {"limit": max(1, days)}

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    if df.empty:
        return pd.DataFrame(columns=["DATE", "TOTAL_VALUE", "STOCKS", "CASH", "CURRENCY"])

    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE", ascending=True).reset_index(drop=True)
    return df


def fetch_recent_var_metrics(
    days: int = 130,
    confidence_levels: Optional[List[float]] = None,
    asof_date: Optional[str] = None,
    engine=None
) -> pd.DataFrame:
    """
    Fetches historical Value-at-Risk records from PORTFOLIO_VAR for the last `days` unique valuation dates
    up to `asof_date` (or latest available date if omitted).
    Filters by specified confidence_levels (default: [0.01, 0.05, 0.95, 0.99]).
    Returns chronologically sorted DataFrame.
    """
    from portfolio_core.db import get_engine, create_all_tables
    engine = engine or get_engine()
    create_all_tables(engine)

    confidence_levels = confidence_levels or [0.01, 0.05, 0.95, 0.99]
    cls_str = ", ".join(str(float(c)) for c in confidence_levels)

    # 1. Fetch the last `days` distinct dates present in PORTFOLIO_VAR up to asof_date
    if asof_date:
        asof_str = pd.to_datetime(asof_date).strftime("%Y-%m-%d")
        dates_query = """
            SELECT DISTINCT `DATE`
            FROM `PORTFOLIO_VAR`
            WHERE `DATE` <= :asof
            ORDER BY `DATE` DESC
            LIMIT :limit
        """
        params = {"limit": max(1, days), "asof": asof_str}
    else:
        dates_query = """
            SELECT DISTINCT `DATE`
            FROM `PORTFOLIO_VAR`
            ORDER BY `DATE` DESC
            LIMIT :limit
        """
        params = {"limit": max(1, days)}

    with engine.connect() as conn:
        dates_res = conn.execute(text(dates_query), params).scalars().all()

    if not dates_res:
        return pd.DataFrame(columns=["DATE", "METHOD", "CONFIDENCE_LEVEL", "PORTFOLIO_VALUE_GBP", "VAR_GBP", "VAR_PCT", "CVAR_GBP"])

    target_dates = [str(d) for d in dates_res]
    dates_str = ", ".join(f"'{d}'" for d in target_dates)

    # 2. Fetch records for those dates and confidence levels
    data_query = f"""
        SELECT `DATE`, `METHOD`, `CONFIDENCE_LEVEL`, `PORTFOLIO_VALUE_GBP`, `VAR_GBP`, `VAR_PCT`, `CVAR_GBP`, `CVAR_PCT`
        FROM `PORTFOLIO_VAR`
        WHERE `DATE` IN ({dates_str})
        AND `CONFIDENCE_LEVEL` IN ({cls_str})
        ORDER BY `DATE` ASC, `METHOD` ASC, `CONFIDENCE_LEVEL` ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(data_query), conn)

    if df.empty:
        return pd.DataFrame(columns=["DATE", "METHOD", "CONFIDENCE_LEVEL", "PORTFOLIO_VALUE_GBP", "VAR_GBP", "VAR_PCT", "CVAR_GBP"])

    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE", ascending=True).reset_index(drop=True)
    return df


def fetch_top_risk_contributors(
    top_n: int = 5,
    method_pattern: str = "%Vol-Scaled%",
    asof_date: Optional[str] = None,
    engine=None
) -> pd.DataFrame:
    """
    Fetches the top `top_n` risk contributor positions for the specified model (default: Vol-Scaled VaR)
    from PORTFOLIO_RISK_CONTRIBUTIONS as of the specified date (or latest date <= asof_date).
    Ranked in descending order of absolute Shapley VaR contribution.
    """
    from portfolio_core.db import get_engine, create_all_tables
    engine = engine or get_engine()
    create_all_tables(engine)

    with engine.connect() as conn:
        if asof_date:
            asof_str = pd.to_datetime(asof_date).strftime("%Y-%m-%d")
            target_date = conn.execute(
                text("SELECT MAX(`DATE`) FROM `PORTFOLIO_RISK_CONTRIBUTIONS` WHERE `METHOD` LIKE :m AND `DATE` <= :d"),
                {"m": method_pattern, "d": asof_str}
            ).scalar()
        else:
            target_date = conn.execute(
                text("SELECT MAX(`DATE`) FROM `PORTFOLIO_RISK_CONTRIBUTIONS` WHERE `METHOD` LIKE :m"),
                {"m": method_pattern}
            ).scalar()

        if not target_date:
            return pd.DataFrame(columns=[
                "TICKER", "POSITION_VALUE_GBP", "WEIGHT_PCT",
                "SHAPLEY_VAR_GBP", "SHAPLEY_VAR_PCT",
                "STANDALONE_VAR_GBP", "DIVERSIFICATION_BENEFIT_GBP"
            ])

        query = """
            SELECT `TICKER`, `POSITION_VALUE_GBP`, `WEIGHT_PCT`,
                   `SHAPLEY_VAR_GBP`, `SHAPLEY_VAR_PCT`,
                   `STANDALONE_VAR_GBP`, `DIVERSIFICATION_BENEFIT_GBP`,
                   `CONFIDENCE_LEVEL`, `METHOD`, `DATE`
            FROM `PORTFOLIO_RISK_CONTRIBUTIONS`
            WHERE `DATE` = :d AND `METHOD` LIKE :m
            ORDER BY ABS(`SHAPLEY_VAR_GBP`) DESC
            LIMIT :limit
        """
        df = pd.read_sql(
            text(query),
            conn,
            params={"d": str(target_date), "m": method_pattern, "limit": max(1, top_n)}
        )

    return df


def fetch_dividends_for_date(
    asof_date: Optional[str] = None,
    engine=None
) -> pd.DataFrame:
    """
    Fetches dividend payments recorded for a specific date (or latest valuation date)
    from the CASHFLOWS table, including native currency amounts and converted GBP values.
    Returns a DataFrame with columns:
    ['DATE', 'TICKER', 'TYPE', 'SHARES', 'DIVIDEND_PER_SHARE', 'AMOUNT', 'CURRENCY', 'AMOUNT_GBP']
    """
    from portfolio_core.db import get_engine, create_all_tables
    engine = engine or get_engine()
    create_all_tables(engine)

    with engine.connect() as conn:
        if not asof_date:
            asof_date = conn.execute(text("SELECT MAX(`DATE`) FROM `PORTFOLIO_VALUES`")).scalar()
            if not asof_date:
                asof_date = conn.execute(text("SELECT MAX(`DATE`) FROM `CASHFLOWS` WHERE `TYPE` = 'DIVIDEND'")).scalar()

        if not asof_date:
            return pd.DataFrame(columns=[
                "DATE", "TICKER", "TYPE", "SHARES", "DIVIDEND_PER_SHARE", "AMOUNT", "CURRENCY", "AMOUNT_GBP"
            ])

        target_date = str(asof_date)[:10]

        query = """
            SELECT `DATE`, `TICKER`, `TYPE`, `SHARES`, `DIVIDEND_PER_SHARE`, `AMOUNT`, `CURRENCY`, `AMOUNT_GBP`
            FROM `CASHFLOWS`
            WHERE `DATE` = :d AND `TYPE` = 'DIVIDEND'
            ORDER BY `AMOUNT_GBP` DESC, `TICKER` ASC
        """
        df = pd.read_sql(text(query), conn, params={"d": target_date})

    if df.empty:
        return pd.DataFrame(columns=[
            "DATE", "TICKER", "TYPE", "SHARES", "DIVIDEND_PER_SHARE", "AMOUNT", "CURRENCY", "AMOUNT_GBP"
        ])

    return df


def format_dividend_line(row: Any) -> str:
    """
    Formats an individual dividend payment entry for Telegram HTML caption.
    Translates native currencies (USD, EUR, GBp/GBX, etc.) into GBP.
    """
    ticker = str(row["TICKER"])
    shares = float(row["SHARES"])
    dps = float(row["DIVIDEND_PER_SHARE"])
    amt = float(row["AMOUNT"])
    curr = str(row["CURRENCY"]).strip()
    amt_gbp = float(row["AMOUNT_GBP"])

    shares_str = f"{shares:,.0f}" if shares.is_integer() else f"{shares:,.2f}"

    if curr in ["GBp", "GBX", "GBp_PENCE", "gbp", "gbx"]:
        dps_str = f"{dps:.2f}p"
        amt_str = f"{amt:,.0f}p" if amt.is_integer() else f"{amt:,.2f}p"
        return f"• <b>{ticker}:</b> {shares_str} shs @ {dps_str} ({amt_str}) → <b>£{amt_gbp:,.2f}</b>"
    elif curr.upper() == "GBP":
        dps_str = f"£{dps:,.4f}" if dps < 1.0 else f"£{dps:,.2f}"
        return f"• <b>{ticker}:</b> {shares_str} shs @ {dps_str} → <b>£{amt_gbp:,.2f}</b>"
    elif curr.upper() == "USD":
        dps_str = f"${dps:,.4f}" if dps < 1.0 else f"${dps:,.2f}"
        amt_str = f"${amt:,.2f}"
        return f"• <b>{ticker}:</b> {shares_str} shs @ {dps_str} ({amt_str}) → <b>£{amt_gbp:,.2f}</b>"
    elif curr.upper() == "EUR":
        dps_str = f"€{dps:,.4f}" if dps < 1.0 else f"€{dps:,.2f}"
        amt_str = f"€{amt:,.2f}"
        return f"• <b>{ticker}:</b> {shares_str} shs @ {dps_str} ({amt_str}) → <b>£{amt_gbp:,.2f}</b>"
    else:
        dps_str = f"{dps:,.4f} {curr}"
        amt_str = f"{amt:,.2f} {curr}"
        return f"• <b>{ticker}:</b> {shares_str} shs @ {dps_str} ({amt_str}) → <b>£{amt_gbp:,.2f}</b>"


def fetch_top_position_movers(
    top_n: int = 5,
    asof_date: Optional[str] = None,
    engine=None
) -> pd.DataFrame:
    """
    Computes the day-over-day change in position market value (in GBP) for all held assets
    between the reporting date and the preceding available market date.
    Returns the top `top_n` positions ranked in descending order of absolute value difference (|ΔValue|).
    """
    from portfolio_core.db import get_engine, create_all_tables, fetch_historical_prices_gbp, fetch_portfolio_positions
    engine = engine or get_engine()
    create_all_tables(engine)

    prices_gbp = fetch_historical_prices_gbp(asof_date=asof_date, engine=engine)
    if prices_gbp.empty or len(prices_gbp) < 2:
        return pd.DataFrame(columns=[
            "TICKER", "VALUE_TODAY_GBP", "VALUE_PREV_GBP", "DIFF_GBP", "DIFF_PCT",
            "ABS_DIFF_GBP", "SHARES_TODAY", "PRICE_TODAY_GBP", "PRICE_PREV_GBP"
        ])

    today_date = str(prices_gbp.index[-1])
    prev_date = str(prices_gbp.index[-2])

    today_prices = prices_gbp.iloc[-1]
    prev_prices = prices_gbp.iloc[-2]

    today_positions = fetch_portfolio_positions(asof=today_date, engine=engine)
    prev_positions = fetch_portfolio_positions(asof=prev_date, engine=engine)

    all_tickers = sorted(list(set(today_positions.keys()) | set(prev_positions.keys())))
    rows = []
    for ticker in all_tickers:
        sh_today = float(today_positions.get(ticker, 0.0))
        sh_prev = float(prev_positions.get(ticker, 0.0))

        if sh_today == 0.0 and sh_prev == 0.0:
            continue

        p_today = float(today_prices.get(ticker, 0.0)) if ticker in today_prices else 0.0
        p_prev = float(prev_prices.get(ticker, 0.0)) if ticker in prev_prices else 0.0

        val_today = sh_today * p_today
        val_prev = sh_prev * p_prev
        diff_gbp = val_today - val_prev
        abs_diff = abs(diff_gbp)
        diff_pct = ((val_today / val_prev) - 1.0) * 100.0 if val_prev > 0 else (100.0 if val_today > 0 else 0.0)

        rows.append({
            "TICKER": ticker,
            "VALUE_TODAY_GBP": round(val_today, 2),
            "VALUE_PREV_GBP": round(val_prev, 2),
            "DIFF_GBP": round(diff_gbp, 2),
            "DIFF_PCT": round(diff_pct, 2),
            "ABS_DIFF_GBP": round(abs_diff, 2),
            "SHARES_TODAY": sh_today,
            "PRICE_TODAY_GBP": round(p_today, 4),
            "PRICE_PREV_GBP": round(p_prev, 4),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "TICKER", "VALUE_TODAY_GBP", "VALUE_PREV_GBP", "DIFF_GBP", "DIFF_PCT",
            "ABS_DIFF_GBP", "SHARES_TODAY", "PRICE_TODAY_GBP", "PRICE_PREV_GBP"
        ])

    df = pd.DataFrame(rows)
    df = df.sort_values("ABS_DIFF_GBP", ascending=False).head(max(1, top_n)).reset_index(drop=True)
    return df


def format_top_mover_line(row: Any) -> str:
    """
    Formats an individual top value mover entry for Telegram HTML caption.
    """
    ticker = str(row["TICKER"])
    val_today = float(row["VALUE_TODAY_GBP"])
    diff_gbp = float(row["DIFF_GBP"])
    diff_pct = float(row["DIFF_PCT"])

    diff_sign = "+" if diff_gbp > 0 else ("-" if diff_gbp < 0 else "")
    diff_fmt = f"{diff_sign}£{abs(diff_gbp):,.2f}" if diff_gbp != 0 else "£0.00"

    pct_sign = "+" if diff_pct > 0 else ("-" if diff_pct < 0 else "")
    pct_fmt = f"{pct_sign}{abs(diff_pct):.2f}%" if diff_pct != 0 else "0.00%"

    return f"• <b>{ticker}:</b> £{val_today:,.2f} ({diff_fmt} / {pct_fmt})"


def generate_portfolio_report_chart(
    portfolio_df: pd.DataFrame,
    var_df: pd.DataFrame,
    top_contributors_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None
) -> bytes:
    """
    Generates a financial visual report chart:
    1. Top Subplot: Last Week's Stock Holdings Value (GBP).
    2. Middle Subplot: Last 2 Weeks Return Percentiles (1%, 5%, 95%, 99%) for Historical and Vol-Scaled models.
    3. Bottom Subplot (Optional): Formatted Table of Top Risk Contributors for Vol-Scaled VaR.

    Returns PNG image bytes and optionally saves to `output_path`.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    has_table = top_contributors_df is not None and not top_contributors_df.empty

    if has_table:
        fig, (ax1, ax2, ax3) = plt.subplots(
            nrows=3, ncols=1,
            figsize=(11, 12.5),
            dpi=180,
            gridspec_kw={"height_ratios": [1.15, 1.25, 0.75]}
        )
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(11, 9.5), dpi=180)
        ax3 = None

    fig.patch.set_facecolor("#FAFAFC")

    # =====================================================================
    # Subplot 1: Last Week Stock Holdings Valuation
    # =====================================================================
    ax1.set_facecolor("#FFFFFF")
    if not portfolio_df.empty:
        dates_p = portfolio_df["DATE"]
        stk_val = portfolio_df["STOCKS"]

        ax1.plot(dates_p, stk_val, label="Stock Holdings Value", color="#1F77B4", linewidth=2.5, marker="o", markersize=5)

        # Annotate Latest Stock Valuation
        latest_row = portfolio_df.iloc[-1]
        first_row = portfolio_df.iloc[0]
        latest_stk = latest_row["STOCKS"]
        first_stk = first_row["STOCKS"]
        period_return = ((latest_stk / first_stk) - 1.0) * 100.0 if first_stk > 0 else 0.0
        ret_sign = "+" if period_return >= 0 else ""

        ax1.annotate(
            f"Latest: £{latest_stk:,.2f} ({ret_sign}{period_return:.2f}%)",
            xy=(latest_row["DATE"], latest_stk),
            xytext=(-15, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1F77B4", ec="none", alpha=0.85),
            color="white",
            fontsize=9,
            fontweight="bold",
            ha="right"
        )

        min_y = stk_val.min() * 0.98
        max_y = stk_val.max() * 1.02
        ax1.set_ylim(bottom=max(0, min_y), top=max_y)

    ax1.set_title("Stock Holdings Valuation — Last 2 Weeks (GBP)", fontsize=13, fontweight="bold", pad=10, color="#1A1A2E")
    ax1.set_ylabel("Value (£)", fontsize=10, fontweight="bold", color="#333333")
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("£{x:,.0f}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax1.grid(True, linestyle=":", alpha=0.6, color="#CCCCCC")
    if not portfolio_df.empty:
        ax1.legend(loc="upper left", frameon=True, facecolor="#F8F9FA", edgecolor="#E2E8F0", fontsize=9)

    # =====================================================================
    # Subplot 2: Return Percentiles (1%, 5%, 95%, 99%)
    # =====================================================================
    ax2.set_facecolor("#FFFFFF")
    if not var_df.empty:
        model_styles = {
            ("Historical Simulation", 0.01): {"color": "#2CA02C", "linestyle": "-", "label": "Hist 1% (Gain)"},
            ("Historical Simulation", 0.05): {"color": "#66BB6A", "linestyle": "-", "label": "Hist 5% (Gain)"},
            ("Historical Simulation", 0.95): {"color": "#4A90E2", "linestyle": "-", "label": "Hist 95% (Loss)"},
            ("Historical Simulation", 0.99): {"color": "#D0021B", "linestyle": "-", "label": "Hist 99% (Loss)"},
            ("Vol-Scaled VaR (EWMA Volatility (λ=0.94))", 0.01): {"color": "#FF7F0E", "linestyle": "--", "label": "Vol-Scaled 1% (Gain)"},
            ("Vol-Scaled VaR (EWMA Volatility (λ=0.94))", 0.05): {"color": "#FFA726", "linestyle": "--", "label": "Vol-Scaled 5% (Gain)"},
            ("Vol-Scaled VaR (EWMA Volatility (λ=0.94))", 0.95): {"color": "#26A69A", "linestyle": "--", "label": "Vol-Scaled 95% (Loss)"},
            ("Vol-Scaled VaR (EWMA Volatility (λ=0.94))", 0.99): {"color": "#9013FE", "linestyle": "--", "label": "Vol-Scaled 99% (Loss)"},
        }

        # Use subtle markers only if data points are few (<= 20 days)
        num_unique_dates = len(var_df["DATE"].unique())
        use_marker = num_unique_dates <= 20
        marker_style = "o" if use_marker else None
        marker_size = 3.5 if use_marker else 0

        for (method, cl), group in var_df.groupby(["METHOD", "CONFIDENCE_LEVEL"]):
            style = model_styles.get((method, round(cl, 2)), {
                "color": "#666666",
                "linestyle": ":",
                "label": f"{method} ({cl*100:.0f}%)"
            })
            ax2.plot(
                group["DATE"],
                group["VAR_GBP"],
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.6,
                marker=marker_style,
                markersize=marker_size
            )

        ax2.axhline(0, color="#888888", linestyle=":", linewidth=1.0, alpha=0.8)
        ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter("£{x:,.0f}"))
        ax2.legend(loc="upper left", frameon=True, facecolor="#F8F9FA", edgecolor="#E2E8F0", fontsize=8, ncol=4)

    # Dynamic timeline duration label in title
    num_unique_dates = len(var_df["DATE"].unique()) if not var_df.empty else 0
    if num_unique_dates >= 100:
        time_label = "6-Month Timeline"
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    elif num_unique_dates >= 25:
        time_label = "1-Month Timeline"
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    else:
        time_label = f"Last {max(1, num_unique_dates)} Days"
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    ax2.set_title(f"1-Day Return Percentiles (1%, 5%, 95%, 99%) — {time_label} (GBP Exposure)", fontsize=13, fontweight="bold", pad=10, color="#1A1A2E")
    ax2.set_ylabel("P&L Exposure (£)", fontsize=10, fontweight="bold", color="#333333")
    ax2.grid(True, linestyle=":", alpha=0.6, color="#CCCCCC")

    # =====================================================================
    # Subplot 3: Top Risk Contributors Table (Vol-Scaled VaR)
    # =====================================================================
    if has_table and ax3 is not None:
        ax3.set_facecolor("#FAFAFC")
        ax3.axis("off")

        table_data = []
        for _, r in top_contributors_df.iterrows():
            pos_v = float(r["POSITION_VALUE_GBP"])
            wt = float(r["WEIGHT_PCT"])
            s_var = float(r["SHAPLEY_VAR_GBP"])
            s_pct = float(r["SHAPLEY_VAR_PCT"])
            st_var = float(r["STANDALONE_VAR_GBP"]) if not pd.isna(r.get("STANDALONE_VAR_GBP")) else 0.0
            div_b = float(r["DIVERSIFICATION_BENEFIT_GBP"]) if not pd.isna(r.get("DIVERSIFICATION_BENEFIT_GBP")) else 0.0

            table_data.append([
                str(r["TICKER"]),
                f"£{pos_v:,.2f}",
                f"{wt:.2f}%",
                f"-£{abs(s_var):,.2f}",
                f"{s_pct:.2f}%",
                f"-£{abs(st_var):,.2f}",
                f"£{div_b:,.2f}"
            ])

        headers = ["Ticker", "Position Value", "Portfolio Weight", "Shapley VaR", "Risk Share", "Standalone VaR", "Div. Benefit"]
        tbl = ax3.table(
            cellText=table_data,
            colLabels=headers,
            loc="center",
            cellLoc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.6)

        for (row_idx, col_idx), cell in tbl.get_celld().items():
            if row_idx == 0:
                cell.set_facecolor("#1A1A2E")
                cell.set_text_props(color="white", weight="bold")
            elif row_idx % 2 == 0:
                cell.set_facecolor("#F1F5F9")
            else:
                cell.set_facecolor("#FFFFFF")

        ax3.set_title(
            f"Top {len(top_contributors_df)} Risk Contributors — Volatility-Scaled VaR",
            fontsize=12,
            fontweight="bold",
            pad=12,
            color="#1A1A2E"
        )

    plt.tight_layout(pad=2.2)

    # Export to memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    image_bytes = buf.getvalue()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(image_bytes)

    return image_bytes


def format_telegram_caption(
    portfolio_df: pd.DataFrame,
    var_df: pd.DataFrame,
    top_contributors_df: Optional[pd.DataFrame] = None,
    dividends_df: Optional[pd.DataFrame] = None,
    top_movers_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Constructs a rich HTML caption summarizing daily valuations, top 5 daily value movers
    (|ΔValue| vs previous day), dividends received on the reporting date (with translations into GBP),
    latest VaR metrics for 1%, 5%, 95%, and 99% percentiles (separate for Historical and Vol-Scaled models),
    and a breakdown of the Top 5 Vol-Scaled VaR risk contributors.
    """
    if portfolio_df.empty:
        return "📊 <b>Portfolio Daily Report</b>\n<i>No valuation data available.</i>"

    latest_p = portfolio_df.iloc[-1]
    date_str = latest_p["DATE"].strftime("%Y-%m-%d") if hasattr(latest_p["DATE"], "strftime") else str(latest_p["DATE"])
    stk_val = float(latest_p["STOCKS"])
    curr = str(latest_p.get("CURRENCY", "GBP"))

    # Day-over-day change in stock holdings
    if len(portfolio_df) > 1:
        prev_p = portfolio_df.iloc[-2]
        prev_stk = float(prev_p["STOCKS"])
        dod_chg = stk_val - prev_stk
        dod_pct = ((stk_val / prev_stk) - 1.0) * 100.0 if prev_stk > 0 else 0.0
        dod_sign = "+" if dod_chg > 0 else ("-" if dod_chg < 0 else "")
        chg_fmt = f"{dod_sign}£{abs(dod_chg):,.2f}" if dod_chg != 0 else "£0.00"
        dod_pct_sign = "+" if dod_pct > 0 else ("-" if dod_pct < 0 else "")
        dod_str = f" ({chg_fmt} / {dod_pct_sign}{abs(dod_pct):.2f}%)"
    else:
        dod_str = ""

    # 1-Week change in stock holdings
    if len(portfolio_df) >= 5:
        w_p = portfolio_df.iloc[0]
        w_stk = float(w_p["STOCKS"])
        w_chg = stk_val - w_stk
        w_pct = ((stk_val / w_stk) - 1.0) * 100.0 if w_stk > 0 else 0.0
        w_sign = "+" if w_chg > 0 else ("-" if w_chg < 0 else "")
        w_chg_fmt = f"{w_sign}£{abs(w_chg):,.2f}" if w_chg != 0 else "£0.00"
        w_pct_sign = "+" if w_pct > 0 else ("-" if w_pct < 0 else "")
        w_str = f"\n• <b>1-Week Change:</b> {w_chg_fmt} ({w_pct_sign}{abs(w_pct):.2f}%)"
    else:
        w_str = ""

    caption_lines = [
        "📊 <b>Daily Stock Holdings & Risk Report</b>",
        f"📅 <i>Valuation Date: {date_str}</i>",
        "",
        "📈 <b>Stock Holdings Valuation:</b>",
        f"• <b>Stock Holdings:</b> £{stk_val:,.2f} {curr}{dod_str}{w_str}",
    ]

    # Top Daily Value Movers Section (|ΔValue| from previous day)
    if top_movers_df is not None and not top_movers_df.empty:
        caption_lines.append("")
        caption_lines.append(f"🔥 <b>Top {len(top_movers_df)} Daily Value Movers (|ΔValue|):</b>")
        for _, r in top_movers_df.iterrows():
            caption_lines.append(format_top_mover_line(r))

    # Dividends Section (Dividends paid on reporting day)
    if dividends_df is not None and not dividends_df.empty:
        total_div_gbp = float(dividends_df["AMOUNT_GBP"].sum())
        caption_lines.append("")
        caption_lines.append("💵 <b>Dividends Paid Today:</b>")
        for _, r in dividends_df.iterrows():
            caption_lines.append(format_dividend_line(r))
        if len(dividends_df) > 1:
            caption_lines.append(f"• <b>Total Dividends:</b> <b>£{total_div_gbp:,.2f}</b>")
    else:
        caption_lines.append("")
        caption_lines.append("💵 <b>Dividends Paid Today:</b> None")

    # Percentile Highlights (1%, 5%, 95%, 99% separate for Historical and Vol-Scaled)
    if not var_df.empty:
        latest_var_date = var_df["DATE"].max()
        today_var = var_df[var_df["DATE"] == latest_var_date].copy()

        # Historical Simulation
        hist_df = today_var[today_var["METHOD"] == "Historical Simulation"].sort_values("CONFIDENCE_LEVEL")
        if not hist_df.empty:
            caption_lines.append("")
            caption_lines.append("🏛️ <b>Historical Simulation Percentiles:</b>")
            for _, r in hist_df.iterrows():
                cl = float(r["CONFIDENCE_LEVEL"])
                pct_label = f"{round(cl * 100.0)}%"
                v_gbp = float(r["VAR_GBP"])
                v_pct = float(r.get("VAR_PCT", 0.0))
                sign_str = "+" if v_gbp > 0 else ("-" if v_gbp < 0 else "")
                pct_sign = "+" if v_pct > 0 else ("-" if v_pct < 0 else "")
                caption_lines.append(f"• <b>{pct_label} Percentile:</b> {sign_str}£{abs(v_gbp):,.2f} ({pct_sign}{abs(v_pct):.2f}%)")

        # Vol-Scaled VaR
        vol_df = today_var[today_var["METHOD"].str.contains("Vol-Scaled")].sort_values("CONFIDENCE_LEVEL")
        if not vol_df.empty:
            caption_lines.append("")
            caption_lines.append("⚡ <b>Vol-Scaled VaR Percentiles:</b>")
            for _, r in vol_df.iterrows():
                cl = float(r["CONFIDENCE_LEVEL"])
                pct_label = f"{round(cl * 100.0)}%"
                v_gbp = float(r["VAR_GBP"])
                v_pct = float(r.get("VAR_PCT", 0.0))
                sign_str = "+" if v_gbp > 0 else ("-" if v_gbp < 0 else "")
                pct_sign = "+" if v_pct > 0 else ("-" if v_pct < 0 else "")
                caption_lines.append(f"• <b>{pct_label} Percentile:</b> {sign_str}£{abs(v_gbp):,.2f} ({pct_sign}{abs(v_pct):.2f}%)")

    # Top 5 Risk Contributors Breakdown
    if top_contributors_df is not None and not top_contributors_df.empty:
        caption_lines.append("")
        caption_lines.append(f"🎯 <b>Top {len(top_contributors_df)} Vol-Scaled VaR Risk Contributors:</b>")
        for _, r in top_contributors_df.iterrows():
            t = r["TICKER"]
            pos_v = float(r["POSITION_VALUE_GBP"])
            wt = float(r["WEIGHT_PCT"])
            s_var = abs(float(r["SHAPLEY_VAR_GBP"]))
            s_pct = float(r["SHAPLEY_VAR_PCT"])
            caption_lines.append(
                f"• <b>{t}:</b> £{pos_v:,.2f} ({wt:.1f}%) → VaR: -£{s_var:,.2f} (<b>{s_pct:.1f}%</b>)"
            )

    caption_lines.append("")
    caption_lines.append("🤖 <i>Automated update by the Portfolio-Analytics-Platform</i>")
    return "\n".join(caption_lines)


def send_telegram_photo(
    token: str,
    chat_id: str,
    photo_bytes: bytes,
    caption: str,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Sends a photo with HTML caption to a Telegram chat using the Telegram Bot API.
    If the caption exceeds Telegram's 1024 character limit for sendPhoto, delivers
    the photo and sends the complete caption text via sendMessage (which supports up to 4096 characters).
    """
    if not token:
        raise ValueError("Telegram Bot Token is required to send notifications.")
    if not chat_id:
        raise ValueError("Telegram chat_id is required.")

    clean_cid = str(chat_id).strip()
    photo_url = f"https://api.telegram.org/bot{token}/sendPhoto"
    files = {"photo": ("portfolio_report.png", photo_bytes, "image/png")}

    if len(caption) <= 1024:
        data = {
            "chat_id": clean_cid,
            "caption": caption,
            "parse_mode": "HTML"
        }
        resp = requests.post(photo_url, data=data, files=files, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    else:
        # Photo caption limit is 1024 chars: send photo first, then full caption via sendMessage
        data = {"chat_id": clean_cid}
        resp_photo = requests.post(photo_url, data=data, files=files, timeout=timeout)
        resp_photo.raise_for_status()

        msg_url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp_msg = requests.post(
            msg_url,
            json={"chat_id": clean_cid, "text": caption, "parse_mode": "HTML"},
            timeout=timeout
        )
        resp_msg.raise_for_status()
        return resp_msg.json()


def fetch_available_dates(engine=None) -> List[str]:
    """
    Returns a sorted list of all available historical calculation dates (YYYY-MM-DD)
    present in the PORTFOLIO_VALUES table.
    """
    from portfolio_core.db import get_engine, create_all_tables
    engine = engine or get_engine()
    create_all_tables(engine)
    with engine.connect() as conn:
        dates_res = conn.execute(
            text("SELECT DISTINCT `DATE` FROM `PORTFOLIO_VALUES` ORDER BY `DATE` ASC")
        ).scalars().all()
    return [pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates_res if d is not None]


def generate_report(
    asof_date: Optional[str] = None,
    portfolio_days: int = 14,
    var_days: int = 130,
    confidence_levels: Optional[List[float]] = None,
    top_risk_contributors_n: int = 5,
    top_movers_n: int = 5,
    engine=None,
    output_chart_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a financial visual report chart and summary caption for a specific asof_date
    (or the latest available date if omitted) using existing data in the database.
    Does not require sending to Telegram.
    """
    from portfolio_core.db import get_engine, create_all_tables
    engine = engine or get_engine()
    create_all_tables(engine)

    # 1. Fetch historical data up to asof_date
    portfolio_df = fetch_recent_portfolio_values(days=portfolio_days, asof_date=asof_date, engine=engine)
    var_df = fetch_recent_var_metrics(days=var_days, confidence_levels=confidence_levels, asof_date=asof_date, engine=engine)

    latest_date_str = None
    if asof_date:
        latest_date_str = pd.to_datetime(asof_date).strftime("%Y-%m-%d")
    elif not portfolio_df.empty:
        latest_date_str = portfolio_df.iloc[-1]["DATE"].strftime("%Y-%m-%d") if hasattr(portfolio_df.iloc[-1]["DATE"], "strftime") else str(portfolio_df.iloc[-1]["DATE"])

    top_contrib_df = fetch_top_risk_contributors(
        top_n=top_risk_contributors_n,
        method_pattern="%Vol-Scaled%",
        asof_date=latest_date_str,
        engine=engine
    )

    dividends_df = fetch_dividends_for_date(asof_date=latest_date_str, engine=engine)
    top_movers_df = fetch_top_position_movers(top_n=top_movers_n, asof_date=latest_date_str, engine=engine)

    # 2. Render chart
    chart_bytes = generate_portfolio_report_chart(
        portfolio_df=portfolio_df,
        var_df=var_df,
        top_contributors_df=top_contrib_df,
        output_path=output_chart_path
    )

    # 3. Format caption
    caption = format_telegram_caption(
        portfolio_df=portfolio_df,
        var_df=var_df,
        top_contributors_df=top_contrib_df,
        dividends_df=dividends_df,
        top_movers_df=top_movers_df
    )

    total_div_gbp = float(dividends_df["AMOUNT_GBP"].sum()) if not dividends_df.empty else 0.0
    clean_md_caption = (
        caption.replace("<b>", "**")
        .replace("</b>", "**")
        .replace("<i>", "*")
        .replace("</i>", "*")
    )

    return {
        "asof_date": latest_date_str,
        "chart_bytes": chart_bytes,
        "caption": caption,
        "clean_markdown_caption": clean_md_caption,
        "chart_size_bytes": len(chart_bytes),
        "output_chart_path": output_chart_path,
        "dividends_records": dividends_df.to_dict(orient="records"),
        "dividends_total_gbp": total_div_gbp,
        "top_movers_records": top_movers_df.to_dict(orient="records"),
        "portfolio_df": portfolio_df,
        "var_df": var_df,
        "top_contributors_df": top_contrib_df,
    }


def send_telegram_report(
    recipients: List[str],
    token: Optional[str] = None,
    asof_date: Optional[str] = None,
    portfolio_days: int = 14,
    var_days: int = 130,
    confidence_levels: Optional[List[float]] = None,
    top_risk_contributors_n: int = 5,
    top_movers_n: int = 5,
    engine=None,
    output_chart_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Full reporting pipeline:
    1. Fetches recent portfolio values, VaR metrics, top Vol-Scaled VaR risk contributors,
       top daily value movers (|ΔValue| vs previous day), and reporting day dividend payouts from the database
       as of `asof_date` (or latest date).
    2. Renders the multi-panel chart (including the Top 5 Risk Contributors table).
    3. Builds the HTML formatted caption with valuation, top movers, dividends, risk breakdown, and contributors.
    4. Delivers the chart and caption to all configured Telegram recipient chat IDs.
    """
    bot_token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise ValueError(
            "Telegram Bot Token not found. Please set TELEGRAM_BOT_TOKEN in your .env file."
        )

    clean_recipients = [str(r).strip() for r in recipients if str(r).strip()]
    if not clean_recipients:
        raise ValueError("No Telegram recipients configured. Please specify at least one chat ID in config.yaml.")

    # 1. Generate report data and chart
    report = generate_report(
        asof_date=asof_date,
        portfolio_days=portfolio_days,
        var_days=var_days,
        confidence_levels=confidence_levels,
        top_risk_contributors_n=top_risk_contributors_n,
        top_movers_n=top_movers_n,
        engine=engine,
        output_chart_path=output_chart_path
    )

    # 2. Broadcast to recipients
    success = []
    failed = []
    for cid in clean_recipients:
        try:
            res = send_telegram_photo(token=bot_token, chat_id=cid, photo_bytes=report["chart_bytes"], caption=report["caption"])
            success.append(cid)
        except Exception as e:
            failed.append({"chat_id": cid, "error": str(e)})

    return {
        "status": "Delivered" if not failed else ("Partial" if success else "Failed"),
        "asof_date": report["asof_date"],
        "recipients_total": len(clean_recipients),
        "delivered_recipients": success,
        "failed_recipients": failed,
        "chart_size_bytes": report["chart_size_bytes"],
        "caption": report["caption"],
        "clean_markdown_caption": report["clean_markdown_caption"],
        "dividends_records": report["dividends_records"],
        "dividends_total_gbp": report["dividends_total_gbp"],
        "top_movers_records": report["top_movers_records"],
    }


def generate_reports_for_dates(
    dates: List[str],
    recipients: Optional[List[str]] = None,
    token: Optional[str] = None,
    send_telegram: bool = False,
    output_dir: Optional[str] = None,
    portfolio_days: int = 14,
    var_days: int = 130,
    confidence_levels: Optional[List[float]] = None,
    top_risk_contributors_n: int = 5,
    top_movers_n: int = 5,
    engine=None
) -> Dict[str, Dict[str, Any]]:
    """
    Generates reports for a sequence or list of historical dates.
    Optionally saves PNG charts to `output_dir` and broadcasts to `recipients` via Telegram.
    """
    results = {}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for d in dates:
        date_str = pd.to_datetime(d).strftime("%Y-%m-%d")
        chart_path = os.path.join(output_dir, f"portfolio_report_{date_str}.png") if output_dir else None

        if send_telegram:
            res = send_telegram_report(
                recipients=recipients or [],
                token=token,
                asof_date=date_str,
                portfolio_days=portfolio_days,
                var_days=var_days,
                confidence_levels=confidence_levels,
                top_risk_contributors_n=top_risk_contributors_n,
                top_movers_n=top_movers_n,
                engine=engine,
                output_chart_path=chart_path
            )
        else:
            res = generate_report(
                asof_date=date_str,
                portfolio_days=portfolio_days,
                var_days=var_days,
                confidence_levels=confidence_levels,
                top_risk_contributors_n=top_risk_contributors_n,
                top_movers_n=top_movers_n,
                engine=engine,
                output_chart_path=chart_path
            )
        results[date_str] = res

    return results


if __name__ == "__main__":
    import argparse
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate financial visual reports (and optional Telegram broadcast) using historical database data."
    )
    parser.add_argument("--date", "--asof-date", dest="asof_date", type=str, help="Specific historical as-of date (YYYY-MM-DD).")
    parser.add_argument("--dates", nargs="+", help="List of historical dates (YYYY-MM-DD).")
    parser.add_argument("--start-date", type=str, help="Start date for range of historical reports (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, help="End date for range of historical reports (YYYY-MM-DD).")
    parser.add_argument("--send-telegram", action="store_true", help="Send generated reports to configured Telegram recipients.")
    parser.add_argument("--recipients", nargs="+", help="Override Telegram recipient chat IDs.")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save generated chart PNGs (default: current directory).")
    parser.add_argument("--list-dates", action="store_true", help="List all available valuation dates present in the database.")

    args = parser.parse_args()

    from portfolio_core.db import get_engine
    engine = get_engine()

    if args.list_dates:
        avail_dates = fetch_available_dates(engine=engine)
        print(f"Available historical valuation dates in database ({len(avail_dates)} total):")
        for d in avail_dates:
            print(f"  - {d}")
        exit(0)

    # Resolve dates
    target_dates = []
    if args.dates:
        target_dates.extend(args.dates)
    elif args.asof_date:
        target_dates.append(args.asof_date)
    elif args.start_date and args.end_date:
        all_avail = fetch_available_dates(engine=engine)
        s = pd.to_datetime(args.start_date).strftime("%Y-%m-%d")
        e = pd.to_datetime(args.end_date).strftime("%Y-%m-%d")
        target_dates = [d for d in all_avail if s <= d <= e]
        if not target_dates:
            print(f"No available dates found between {s} and {e} in the database.")
            exit(1)
    else:
        # Default: latest date
        target_dates = [None]

    # Resolve Telegram configuration if sending
    recipients = args.recipients
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f) or {}
        rep_cfg = cfg.get("ops", {}).get("portfolio_telegram_report", {}).get("config", {})
        if not recipients:
            recipients = rep_cfg.get("recipients", [])
        if not token:
            token = rep_cfg.get("bot_token") or os.getenv("TELEGRAM_BOT_TOKEN")

    print(f"Generating report(s) for {len(target_dates)} date(s)...")
    for d in target_dates:
        date_label = d or "latest"
        print(f"\n--- Processing Date: {date_label} ---")
        chart_file = os.path.join(args.output_dir, f"portfolio_report_{d or 'latest'}.png")
        if args.send_telegram:
            if not recipients:
                print("Error: --send-telegram specified but no recipients found in config.yaml or --recipients.")
                exit(1)
            res = send_telegram_report(
                recipients=recipients,
                token=token,
                asof_date=d,
                output_chart_path=chart_file,
                engine=engine
            )
            print(f"Status: {res['status']} (Delivered: {len(res['delivered_recipients'])}, Failed: {len(res['failed_recipients'])})")
            print(f"Chart saved to: {chart_file}")
        else:
            res = generate_report(
                asof_date=d,
                output_chart_path=chart_file,
                engine=engine
            )
            print(f"Chart saved to: {chart_file}")
            print(f"Caption summary:\n{res['clean_markdown_caption']}")

