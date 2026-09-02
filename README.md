# 🏛️ Portfolio Analytics Platform (Monorepo)

An institutional-grade portfolio risk analytics, automated data pipeline, and interactive web dashboard platform.

---

## 🏗️ Monorepo Architecture

```
portfolio-analytics-platform/
├── packages/
│   └── portfolio-core/             # 🧠 Shared Core Library (Single Source of Truth)
│       ├── src/portfolio_core/
│       │   ├── config.py           # Configuration & .env loader
│       │   ├── db.py               # SQLAlchemy connection pooling, schemas, & queries
│       │   └── analytics/          # Pure financial mathematics
│       │       ├── volatility.py   # Rolling sample, EWMA (λ=0.94), Parkinson, scaling factors
│       │       ├── var.py          # Historical VaR, Vol-Scaled VaR, CVaR, Shapley Attribution
│       │       └── statistics.py   # Returns, KDE, moments, Jarque-Bera normality tests
│       └── tests/                  # Core analytics and database unit tests
│
├── apps/
│   ├── pipeline/                   # ⚙️ Dagster & Dask Ingestion & Materialization Engine
│   │   ├── repo.py                 # Dagster jobs, assets, ops, schedules
│   │   ├── reporting.py            # Automated multi-panel PDF/chart Telegram reporter
│   │   └── tests/                  # Pipeline reporting integration tests
│   │
│   └── dashboard/                  # 📊 Streamlit + Plotly Interactive Risk Dashboard
│       ├── app.py                  # Main dashboard application entrypoint
│       └── src/ui/                 # Modular UI tabs
│           ├── tab_volatility.py   # Rolling volatilities & multi-estimator comparison
│           ├── tab_var.py          # Full VaR spectrum (1%-99%), Shapley attribution & scenarios
│           ├── tab_returns.py      # Price levels, returns series, histogram & normality
│           └── tab_portfolio.py    # Holdings allocation, valuation timeline & correlation matrix
│
├── docker-compose.yml              # Multi-container local/production deployment
├── pyproject.toml                  # Monorepo workspace configuration
└── .env.example                    # Template for database & API credentials
```

---

## 🚀 Quickstart & Installation

### 1. Clone & Set Up Environment

```bash
cd /home/tim/Projects/portfolio-analytics-platform
python3 -m venv .venv
source .venv/bin/activate

# Install the shared core library in editable mode
pip install -e packages/portfolio-core

# Install pipeline & dashboard dependencies
pip install -r apps/pipeline/requirements.txt
pip install -r apps/dashboard/requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your secrets (database password and optional Telegram bot token):
# DB_PASSWORD=your_db_password_here
# TELEGRAM_BOT_TOKEN=your_telegram_bot_token_optional
```

> **Note:** Non-sensitive database settings (host, port, user, default database, available databases) and risk model parameters are configured in `apps/pipeline/config.yaml` and `apps/dashboard/config.yaml`. Any values in `.env` (such as `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_NAME`, `DB_DATABASES`) can also be used as environment variable overrides.

---

## 🧪 Running Unit & Integration Tests

Run the complete test suite across all packages and apps:

```bash
pytest -v
```

Run tests for a specific component:

```bash
# Test shared analytics & database core (28 tests)
pytest -v packages/portfolio-core/tests

# Test Dagster reporting module (21 tests)
pytest -v apps/pipeline/tests

# Test Dashboard integration (10 tests)
pytest -v apps/dashboard/tests
```

---

## 🖥️ Running the Applications

### Launch Streamlit Dashboard

```bash
cd apps/dashboard
streamlit run app.py
```
*Access in browser at `http://localhost:8501`.*

### Launch Dagster Pipeline Webserver

```bash
cd apps/pipeline
dagster dev -f repo.py -p 3000
```
*Access Dagster UI at `http://localhost:3000`.*

---

## 🐳 Docker Deployment

Run all services together with Docker Compose:

```bash
docker compose up --build -d
```
