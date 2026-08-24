"""
Portfolio Risk & Volatility Analytics Dashboard (Main Entrypoint).
Interactive web application powered by Streamlit, Plotly, and SQLAlchemy.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple

from portfolio_core.config import config
from portfolio_core.db import (
    get_engine,
    test_db_connection,
    fetch_available_tickers,
    fetch_historical_prices_gbp,
    fetch_portfolio_positions,
    fetch_available_var_dates
)
from src.ui.theme import inject_custom_css, ensure_sidebar_collapsed
from src.ui.tab_volatility import render_tab_volatility
from src.ui.tab_var import render_tab_var
from src.ui.tab_returns import render_tab_returns
from src.ui.tab_portfolio import render_tab_portfolio


# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=config.ui_config.get("app_title", "Portfolio Risk & Volatility Analytics"),
    page_icon=config.ui_config.get("app_icon", "📈"),
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject custom styling & auto-collapse sidebar
inject_custom_css()
ensure_sidebar_collapsed()


# -----------------------------------------------------------------------------
# 2. Cached Data Loaders
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def load_cached_data() -> Tuple[pd.DataFrame, List[str], Dict[str, float], List[str]]:
    """Loads and caches market prices, tickers, positions, and risk dates."""
    engine = get_engine()
    tickers = fetch_available_tickers(engine)
    prices_gbp = fetch_historical_prices_gbp(engine=engine)
    positions = fetch_portfolio_positions(engine=engine)
    var_dates = fetch_available_var_dates(engine)
    return prices_gbp, tickers, positions, var_dates


# -----------------------------------------------------------------------------
# 3. Sidebar: Database Status & Application Metadata
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 Risk Analytics")
    st.caption("Quantitative Portfolio Risk, Volatility & Return Modeling")
    st.divider()

    # DB Connection Status Check
    db_connected, db_msg = test_db_connection()
    if db_connected:
        st.success("🟢 Database Connected")
    else:
        st.error(f"🔴 DB Connection Failed: {db_msg}")

    st.markdown("### ⚙️ Environment Details")
    db_cfg = config.db_config
    st.markdown(f"- **DB Backend:** `{db_cfg.get('type', 'mariadb')}`")
    st.markdown(f"- **Host:** `{db_cfg.get('host', 'local')}`")
    st.markdown(f"- **Database:** `{db_cfg.get('database', 'stocks')}`")

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
st.title("📈 Portfolio Risk & Volatility Analytics")
st.markdown("Interactive quantitative risk suite for rolling volatilities, tail risk percentiles, and return distribution modeling.")

# Load core data
with st.spinner("Connecting to database and loading market data..."):
    prices_gbp, available_tickers, positions, var_dates = load_cached_data()

if prices_gbp.empty:
    st.error("No historical market price data found. Please ensure the database is accessible and populated.")
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
tab1, tab2, tab3, tab4 = st.tabs([
    "💼 Portfolio & Correlation",
    "🛡️ Value-at-Risk Spectrum",
    "📊 Levels, Returns & Histogram",
    "📈 Rolling Volatility"
])

with tab1:
    render_tab_portfolio(
        prices_gbp=prices_gbp,
        positions=positions,
        asof_date=latest_date_str
    )

with tab2:
    render_tab_var(
        prices_gbp=prices_gbp,
        positions=positions,
        asof_date=latest_date_str
    )

with tab3:
    render_tab_returns(
        prices_gbp=prices_gbp,
        available_tickers=available_tickers
    )

with tab4:
    render_tab_volatility(
        prices_gbp=prices_gbp,
        available_tickers=available_tickers
    )
