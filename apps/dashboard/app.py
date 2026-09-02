"""
Portfolio Risk Analytics Dashboard (Main Entrypoint).
Interactive web application powered by Streamlit, Plotly, and SQLAlchemy.
"""

import os
import concurrent.futures
import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from portfolio_core.config import config, is_dev_environment
from portfolio_core.db import (
    get_engine,
    test_db_connection,
    fetch_available_tickers,
    fetch_historical_prices_gbp,
    fetch_portfolio_positions,
    fetch_available_var_dates,
    fetch_portfolio_values_history,
    fetch_benchmarks_info,
    fetch_benchmark_values_history,
    fetch_all_transactions,
    fetch_raw_asset_prices
)
try:
    from src.ui.theme import inject_custom_css, ensure_sidebar_collapsed
    from src.ui.tab_volatility import render_tab_volatility
    from src.ui.tab_var import render_tab_var
    from src.ui.tab_returns import render_tab_returns
    from src.ui.tab_portfolio import render_tab_portfolio
    from src.ui.tab_benchmarks import render_tab_benchmarks
    from src.ui.tab_transactions import render_tab_transactions
except ImportError:
    from apps.dashboard.src.ui.theme import inject_custom_css, ensure_sidebar_collapsed
    from apps.dashboard.src.ui.tab_volatility import render_tab_volatility
    from apps.dashboard.src.ui.tab_var import render_tab_var
    from apps.dashboard.src.ui.tab_returns import render_tab_returns
    from apps.dashboard.src.ui.tab_portfolio import render_tab_portfolio
    from apps.dashboard.src.ui.tab_benchmarks import render_tab_benchmarks
    from apps.dashboard.src.ui.tab_transactions import render_tab_transactions


# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
try:
    st.set_page_config(
        page_title=config.ui_config.get("app_title", "Portfolio Risk Analytics"),
        page_icon=config.ui_config.get("app_icon", "📈"),
        layout="wide",
        initial_sidebar_state="collapsed"
    )
except Exception:
    pass

# Inject custom styling & auto-collapse sidebar
try:
    inject_custom_css()
    ensure_sidebar_collapsed()
except Exception:
    pass


# -----------------------------------------------------------------------------
# 2. Cached Parallel Data Loader (Option 1: ThreadPool Pre-calculation)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def load_cached_data_parallel(db_name: str, _engine: Optional[Any] = None) -> Dict[str, Any]:
    """
    Loads and caches core market prices, positions, and pre-fetches all tab data
    concurrently across background worker threads for maximum responsiveness.
    """
    engine = _engine
    if engine is None:
        try:
            engine = get_engine(database=db_name)
        except Exception:
            from portfolio_core.db import get_test_engine
            engine = get_test_engine(fallback_to_sqlite=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        f_tickers = executor.submit(fetch_available_tickers, engine=engine)
        f_prices = executor.submit(fetch_historical_prices_gbp, engine=engine)
        f_positions = executor.submit(fetch_portfolio_positions, engine=engine)
        f_var_dates = executor.submit(fetch_available_var_dates, engine=engine)
        f_pv = executor.submit(fetch_portfolio_values_history, days=None, engine=engine)
        f_bm_info = executor.submit(fetch_benchmarks_info, engine=engine)
        f_bm_history = executor.submit(fetch_benchmark_values_history, engine=engine)
        f_tx = executor.submit(fetch_all_transactions, limit=100, engine=engine)

        tickers = f_tickers.result()
        prices_gbp = f_prices.result()
        positions = f_positions.result()
        var_dates = f_var_dates.result()
        pv_df = f_pv.result()
        bm_info_df = f_bm_info.result()
        bm_history_df = f_bm_history.result()
        transactions_df = f_tx.result()

    # Pre-fetch raw prices for the default selected ticker in Tab 4
    raw_cache: Dict[str, pd.DataFrame] = {}
    default_ticker = "NVDA" if "NVDA" in tickers else (tickers[0] if tickers else None)
    if default_ticker:
        try:
            raw_cache[default_ticker] = fetch_raw_asset_prices(default_ticker, engine=engine)
        except Exception:
            pass

    return {
        "prices_gbp": prices_gbp,
        "tickers": tickers,
        "positions": positions,
        "var_dates": var_dates,
        "pv_df": pv_df,
        "bm_info_df": bm_info_df,
        "bm_history_df": bm_history_df,
        "transactions_df": transactions_df,
        "raw_prices_cache": raw_cache
    }


def load_cached_data(db_name: str, _engine: Optional[Any] = None) -> Tuple[pd.DataFrame, List[str], Dict[str, float], List[str]]:
    """Legacy helper returning tuple for backward compatibility."""
    bundle = load_cached_data_parallel(db_name, _engine=_engine)
    return bundle["prices_gbp"], bundle["tickers"], bundle["positions"], bundle["var_dates"]


# -----------------------------------------------------------------------------
# 3. Sidebar: Database Status, Selection & Application Metadata
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 Risk Analytics")
    st.caption("Quantitative Portfolio Risk, Volatility & Return Modeling")
    st.divider()

    # Database Selection Dropdown
    st.markdown("### 🗄️ Database")
    is_dev = is_dev_environment()
    db_cfg = config.db_config
    configured_dbs = db_cfg.get("databases", ["stocks_dev" if is_dev else "stocks"])
    if not isinstance(configured_dbs, list):
        configured_dbs = [configured_dbs]

    if is_dev:
        # Strictly filter to databases ending with _dev
        configured_dbs = [d for d in configured_dbs if d.endswith("_dev")]
        if not configured_dbs:
            configured_dbs = ["stocks_dev"]

    # Active default database: check environment variable DB_NAME, then config default
    default_db = os.getenv("DB_NAME") or db_cfg.get("database", "stocks_dev" if is_dev else "stocks")
    if is_dev and not default_db.endswith("_dev"):
        default_db = f"{default_db}_dev"

    # Ensure default_db is present in options list
    if default_db in configured_dbs:
        db_options = configured_dbs
        default_index = configured_dbs.index(default_db)
    else:
        db_options = [default_db] + [d for d in configured_dbs if d != default_db]
        default_index = 0

    selected_db = st.selectbox(
        "Active Database:",
        options=db_options,
        index=default_index,
        help="Select the database to query and manage. In development mode, only databases ending in '_dev' are permitted."
    )

    if is_dev:
        st.caption("🔒 Dev Mode: Restricted strictly to `*_dev` databases.")
        if not selected_db.endswith("_dev"):
            st.error(f"🚫 Security Restriction: In development mode, connections to non-dev database '{selected_db}' are blocked.")
            st.stop()

    # Active DB Engine
    active_engine = get_engine(database=selected_db)

    # DB Connection Status Check
    db_connected, db_msg = test_db_connection(engine=active_engine)
    if db_connected:
        st.success(f"🟢 Connected to `{selected_db}`")
    else:
        st.error(f"🔴 DB Connection Failed (`{selected_db}`): {db_msg}")

    st.markdown("### ⚙️ Environment Details")
    st.markdown(f"- **DB Backend:** `{db_cfg.get('type', 'mariadb')}`")
    st.markdown(f"- **Host:** `{db_cfg.get('host', 'local')}`")
    st.markdown(f"- **Database:** `{selected_db}`")

    # Refresh Data Cache Button
    if st.button("🔄 Refresh Data Cache", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("### 📚 Quick Guide")
    st.markdown(
        """
        - **Rolling Volatility:** Dynamic standard deviation & EWMA across custom horizons.
        - **VaR Spectrum:** Volatility-Scaled vs Historical Simulation tail risk from 1% to 99%.
        - **Levels & Returns:** Stock prices, daily return series, and distribution histograms with KDE & normality tests.
        - **Portfolio:** Weights breakdown and correlation matrix.
        """
    )
    st.caption("Developed with Streamlit & Plotly")


# -----------------------------------------------------------------------------
# 4. Main Application Layout & Tabs
# -----------------------------------------------------------------------------
st.title("📈 Portfolio Risk Analytics")
st.markdown("Interactive quantitative risk suite for rolling volatilities, tail risk percentiles, and return distribution modeling.")

# Load core data and precalculate tab datasets for selected database concurrently
with st.spinner(f"Connecting to database '{selected_db}' and pre-calculating tab analytics in parallel..."):
    data_bundle = load_cached_data_parallel(selected_db)

prices_gbp = data_bundle["prices_gbp"]
available_tickers = data_bundle["tickers"]
positions = data_bundle["positions"]
var_dates = data_bundle["var_dates"]
pv_df = data_bundle["pv_df"]
bm_info_df = data_bundle["bm_info_df"]
bm_history_df = data_bundle["bm_history_df"]
transactions_df = data_bundle["transactions_df"]
raw_prices_cache = data_bundle["raw_prices_cache"]

if prices_gbp.empty:
    st.error(f"No historical market price data found in database '{selected_db}'. Please ensure the database is accessible and populated.")
    st.stop()

# Header status strip
latest_date_str = str(prices_gbp.index[-1])[:10]
col_h1, col_h2, col_h3 = st.columns(3)
with col_h1:
    st.metric("Latest Market Date", latest_date_str)
with col_h2:
    st.metric("Active Tickers in DB", f"{len(available_tickers)} assets")
with col_h3:
    st.metric("Tracked Holdings", f"{len(positions)} positions")

st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💼 Portfolio Holdings & Valuation",
    "🎯 Benchmarking",
    "🛡️ Value-at-Risk Spectrum",
    "📊 Levels, Returns & Histogram",
    "📈 Rolling Volatility",
    "📝 Transaction Entry"
])

with tab1:
    render_tab_portfolio(
        prices_gbp=prices_gbp,
        positions=positions,
        asof_date=latest_date_str,
        engine=active_engine,
        db_name=selected_db,
        pv_df=pv_df
    )

with tab2:
    render_tab_benchmarks(
        prices_gbp=prices_gbp,
        asof_date=latest_date_str,
        engine=active_engine,
        bm_info_df=bm_info_df,
        bm_history_df=bm_history_df,
        pv_df=pv_df
    )

with tab3:
    render_tab_var(
        prices_gbp=prices_gbp,
        positions=positions,
        asof_date=latest_date_str,
        engine=active_engine
    )

with tab4:
    render_tab_returns(
        prices_gbp=prices_gbp,
        available_tickers=available_tickers,
        engine=active_engine,
        raw_prices_cache=raw_prices_cache
    )

with tab5:
    render_tab_volatility(
        prices_gbp=prices_gbp,
        available_tickers=available_tickers,
        raw_prices_cache=raw_prices_cache
    )

with tab6:
    render_tab_transactions(
        engine=active_engine,
        transactions_df=transactions_df,
        positions_dict=positions
    )
