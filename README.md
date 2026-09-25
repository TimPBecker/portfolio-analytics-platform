# 🏛️ Portfolio Analytics Platform (Monorepo)

An institutional-grade portfolio risk analytics, automated data pipeline, and interactive web dashboard platform.

---

## 🏗️ Monorepo Architecture

```
portfolio-analytics-platform/
├── packages/
│   └── portfolio-core/             # 🧠 Shared Core Library (Single Source of Truth)
│       ├── src/portfolio_core/
│       │   ├── config.py           # Configuration, YAML loader & .env credentials
│       │   ├── db.py               # SQLAlchemy connection pooling, 13 automated schemas & queries
│       │   └── analytics/          # Pure financial mathematics
│       │       ├── volatility.py   # Rolling sample, EWMA (λ=0.94), Parkinson, scaling factors
│       │       ├── var.py          # Historical VaR, Vol-Scaled VaR, CVaR, Shapley Attribution
│       │       ├── empirical_cdf.py# Non-parametric empirical CDF percentile & scenario estimation
│       │       ├── backtesting.py  # Kupiec POF, Christoffersen independence & binomial calibration tests
│       │       └── statistics.py   # Returns, KDE, moments, Jarque-Bera normality tests
│       └── tests/                  # Core analytics and database unit tests
│
├── apps/
│   ├── pipeline/                   # ⚙️ Dagster & Dask Ingestion & Materialization Engine
│   │   ├── repo.py                 # Dagster jobs, assets, ops, schedules & backfill definitions
│   │   ├── reporting.py            # Automated multi-panel PDF/chart Telegram reporter
│   │   ├── config.yaml             # Pipeline execution parameters & DB config
│   │   └── tests/                  # Pipeline reporting integration tests
│   │
│   └── dashboard/                  # 📊 Streamlit + Plotly Interactive Risk Dashboard
│       ├── app.py                  # Main dashboard application entrypoint & cached data loader
│       ├── config.yaml             # Dashboard connection & risk modeling parameters
│       └── src/ui/                 # Modular UI tabs
│           ├── tab_portfolio.py    # Holdings allocation, valuation timeline & correlation matrix
│           ├── tab_benchmarks.py   # Centralized benchmarking, shadow portfolio replication & comparison
│           ├── tab_var.py          # Full VaR spectrum (1%-99%), Shapley attribution & scenarios
│           ├── tab_backtesting.py  # VaR model validation, outlier clustering & diagnostic tests
│           ├── tab_returns.py      # Price levels, returns series, histogram & normality
│           ├── tab_volatility.py   # Rolling volatilities & multi-estimator comparison
│           ├── tab_transactions.py # Buy/sell transaction recording with live Yahoo quote lookup
│           └── tab_market_data.py  # Market data coverage overview, custom time series ingestion & refresh
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
```

Edit `.env` to match your selected database backend (see [Database Setup & Configuration](#-database-setup--configuration) below for details).

---

## 🗄️ Database Setup & Configuration

The platform supports two database backends:
1. **SQLite**: Recommended for fast, single-machine local development and lightweight testing with zero server setup.
2. **MariaDB / MySQL**: Recommended for institutional, multi-container, production, and networked environments.

---

### ❓ Do Users Need to Create Database Tables?

> [!IMPORTANT]
> **NO, you do NOT need to create any database tables manually!**
> The platform manages all table creation automatically. You do NOT need to write SQL DDL or execute migration scripts.

When the application runs (launching the Streamlit dashboard, running a Dagster pipeline asset, recording a transaction, or running tests), the platform automatically executes `create_all_tables(engine)` with `CREATE TABLE IF NOT EXISTS` across your database.

If you wish to explicitly pre-initialize or verify all tables from the command line upfront, run:

```bash
# Initialize all 13 tables in the active database:
python -m portfolio_core.db --init-tables
```

Or programmatically in Python:

```python
from portfolio_core.db import create_all_tables
create_all_tables()
```

#### Managed Database Tables (13 Tables)

| Table | Description |
| :--- | :--- |
| `TRANSACTIONS` | Portfolio trade ledger (Buy/Sell order history, dates, and share quantities). |
| `ASSET_PRICES` | Daily historical asset prices (OHLCV, dividends, stock splits, currency). |
| `FX_RATES` | Daily foreign exchange rates to GBP with forward/back-filling. |
| `CASHFLOWS` | Computed dividend payout cashflows per held position. |
| `CASHACCOUNT` | Multi-currency cash account balances and cumulative valuations in GBP. |
| `PORTFOLIO_VALUES` | Daily total portfolio valuation history (Stocks, Cash, and Total Value in GBP). |
| `PORTFOLIO_VAR` | Value-at-Risk calculations across confidence levels (1%-99%) and multiple risk models. |
| `PORTFOLIO_SCENARIO_PNL` | Simulated scenario profit-and-loss distributions across risk models. |
| `PORTFOLIO_RISK_CONTRIBUTIONS` | Game-theoretic Shapley Value risk attributions per asset position. |
| `BENCHMARKS` | Benchmark definitions and constituent asset weights (single or linear combinations). |
| `BENCHMARK_TRANSACTIONS` | Shadow transactions automatically replicating portfolio cash deployments on trade dates. |
| `BENCHMARK_VALUES` | Daily mark-to-market valuations for all configured benchmark portfolios. |
| `PROCESSED_DATES` | Audit ledger tracking all successfully completed trading days and data ingestion dates. |

---

### Option A: SQLite Setup (Fastest & Simplest)

SQLite requires **zero external database server installation**. The database file is created automatically on disk upon first connection.

#### Which Configurations are NOT Needed for SQLite?
When using SQLite, server connection parameters and credentials are **NOT needed and are ignored**:
- `DB_PASSWORD`: **Not needed** (leave empty or omit from `.env`)
- `DB_USER`: **Not needed**
- `DB_HOST`: **Not needed**
- `DB_PORT`: **Not needed**

#### Configuration Instructions:
1. In `apps/dashboard/config.yaml` and `apps/pipeline/config.yaml`:
   ```yaml
   db:
     type: "sqlite"
     sqlite_path: "stocks.s3db"
   ```
2. Or configure via `.env` overrides:
   ```bash
   DB_TYPE=sqlite
   SQLITE_PATH=stocks.s3db
   ```

> [!NOTE]
> **Strict SQLite Extension Rule**: The SQLite database path must always use the `.s3db` extension (e.g. `stocks.s3db`). This ensures:
> - The database file is automatically excluded from git tracking via `.gitignore`.
> - In development mode (`PORTFOLIO_ENV=development`) and test mode (`PORTFOLIO_ENV=test`), the system automatically resolves to `<name>_dev.s3db` or in-memory `:memory:` to safeguard your production data.

---

### Option B: MariaDB / MySQL Setup (Production & Multi-Container)

MariaDB / MySQL is suited for multi-container Docker deployments, multi-user access, or networked environments.

#### One-Time Database Instance Creation
Before running the application against a MariaDB server for the first time, the database instances themselves must exist on your server. Log in to your MariaDB server as an administrator (e.g., via `mariadb -u root -p`) and create the production and development databases:

```sql
-- 1. Create production and development database instances
CREATE DATABASE IF NOT EXISTS stocks;
CREATE DATABASE IF NOT EXISTS stocks_dev;

-- 2. Create the application user and grant permissions
CREATE USER IF NOT EXISTS 'stocks'@'%' IDENTIFIED BY 'your_secure_password';
GRANT ALL PRIVILEGES ON stocks.* TO 'stocks'@'%';
GRANT ALL PRIVILEGES ON stocks_dev.* TO 'stocks'@'%';
FLUSH PRIVILEGES;
```

*(Note: You only need to create the database containers `stocks` and `stocks_dev`. Do **NOT** create tables manually; the platform creates all tables automatically).*

#### Configuration Instructions:
1. Specify connection parameters in `apps/dashboard/config.yaml` and `apps/pipeline/config.yaml`:
   ```yaml
   db:
     type: "mariadb"
     host: "192.168.178.40"   # or "localhost"
     port: 3306
     user: "stocks"
     database: "stocks"
     databases:
       - "stocks"
   ```
2. Add your secret password to `.env`:
   ```bash
   DB_PASSWORD=your_secure_password
   ```
3. Optional environment overrides in `.env`:
   ```bash
   DB_HOST=192.168.178.40
   DB_PORT=3306
   DB_USER=stocks
   DB_NAME=stocks
   ```

---

### 🛡️ Environment Safety & Database Isolation

The platform enforces strict safety rules across environments via the `PORTFOLIO_ENV` variable:

| Environment | Setting | Database Target | Behavior |
| :--- | :--- | :--- | :--- |
| **Production** | `PORTFOLIO_ENV=production` | `stocks` | Live production operations, portfolio valuations, and real trades. |
| **Development** | `PORTFOLIO_ENV=development` | `stocks_dev` / `stocks_dev.s3db` | Development and staging. Connecting to production `stocks` is strictly blocked by safety assertions. |
| **Testing** | `PORTFOLIO_ENV=test` or `TEST_MODE=1` | `stocks_dev` / `:memory:` | Automated tests, test runners, and diagnostic scripts strictly target dev databases or on-demand SQLite databases. Production `stocks` is never touched. |

---

## 🛠️ Database CLI & Maintenance Utilities

The shared core library includes built-in command-line maintenance tools:

```bash
# 1. Initialize all database tables without running data ingestion:
python -m portfolio_core.db --init-tables

# 2. Run full end-to-end data ingestion, backfilling, and portfolio valuation:
python -m portfolio_core.db

# 3. Clear all computed pipeline tables (strictly preserves manual TRANSACTIONS):
python -m portfolio_core.db --clear-tables

# 4. Prune records after a specific date (preview dry-run):
python -m portfolio_core.db --delete-after 2026-09-24 --dry-run

# 5. Prune records after a specific date (execute deletion):
python -m portfolio_core.db --delete-after 2026-09-24
```

---

## 🧪 Running Unit & Integration Tests

The repository includes a comprehensive test suite with over **100 automated tests** covering core analytics, empirical distributions, VaR backtesting diagnostics, benchmark synchronization, database isolation, Dagster reporting, and Streamlit UI components:

```bash
# Run the complete test suite:
pytest -v

# Run tests for shared analytics & database core:
pytest -v packages/portfolio-core/tests

# Run tests for Dagster pipeline reporting:
pytest -v apps/pipeline/tests

# Run tests for Streamlit dashboard UI:
pytest -v apps/dashboard/tests
```

> [!TIP]
> All tests automatically adhere to safe database isolation (`PORTFOLIO_ENV=test`, `TEST_MODE=1`), utilizing isolated SQLite `:memory:` instances or the MariaDB `stocks_dev` database.

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
